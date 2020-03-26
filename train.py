#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import args

from datetime import datetime
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from utils import prepare_hierarchical_targets, get_model_name, save_model, init_stats, log_model_data, save_stats, \
    get_teacher_prob, clip_gradients, cyclical_lr, get_model_path, get_ground_truth_captions, show_versions

# (Needed to handle Vocabulary pickle)
from vocabulary import get_vocab
from dataset import get_loader, DatasetParams
from model.encoder_decoder import ModelParams, EncoderDecoder, DecoderRNN
from model.loss import HierarchicalXEntropyLoss, SharedEmbeddingXentropyLoss
from vocabulary import caption_ids_to_words, caption_ids_ext_to_words, paragraph_ids_to_words

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    print('WARNING: tensorboardx module not found. '
          'Install it if you want to have support for advanced logging :-)')
    SummaryWriter = None

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration now in main()
device = None


def do_validate(model, valid_loader, criterion, scorers, vocab, teacher_p, args, params, stats, epoch,
                sc_activated=False, gts_sc_val=None):
    begin = datetime.now()
    model.eval()

    gts = gts_sc_val if sc_activated and gts_sc_val is not None else {}
    res = {}

    total_loss = 0

    if params.hierarchical_model:
        total_loss_sent = 0
        total_loss_word = 0

    num_batches = 0
    for i, data in enumerate(valid_loader):
        if params.hierarchical_model:
            (images, captions, lengths, image_ids, features,
             sorting_order, last_sentence_indicator) = data
        else:
            (images, captions, lengths, image_ids, features) = data
            sorting_order = None

        if len(scorers) > 0 and not sc_activated:
            for j in range(captions.shape[0]):
                jid = image_ids[j]
                if jid not in gts:
                    gts[jid] = []
                if params.hierarchical_model:
                    gts[jid].append(paragraph_ids_to_words(captions[j, :], vocab).lower())
                else:
                    # obs! this should be caption_ids_ext_to_words()
                    gts[jid].append(caption_ids_to_words(captions[j, :], vocab, skip_start_token=True).lower())

        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)

        if params.rnn_hidden_init == 'from_features':
            # Subtract one from all lengths to match new target lengths:
            lengths = [x - 1 if x > 0 else x for x in lengths]
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
        else:
            if params.hierarchical_model:
                targets = prepare_hierarchical_targets(last_sentence_indicator, args.max_sentences, lengths, captions,
                                                       device)
            else:
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        init_features = features[0].to(device) if len(features) > 0 and features[0] is not None else None
        persist_features = features[1].to(device) if len(features) > 1 and features[1] is not None else None

        with torch.no_grad():
            outputs = model(images, init_features, captions, lengths, persist_features, teacher_p, args.teacher_forcing,
                            sorting_order)

            if len(scorers) > 0:
                # Generate a caption from the image
                sampled_batch = model.sample(images, init_features, persist_features,
                                             trigram_penalty_alpha=args.trigram_penalty_alpha,
                                             max_seq_length=20, start_token_id=vocab('<start>'))
                sampled_ids_batch = sampled_batch

        if args.share_embedding_weights:
            # Weights of (HxH) projection matrix used for regularizing
            # models that share embedding weights
            projection = model.decoder.projection.weight
            loss = criterion(projection, outputs, targets)
        elif sc_activated:
            sample_len = captions.size(1) if args.self_critical_loss in ['mixed', 'mixed_with_face'] else 20
            with torch.no_grad():
                sampled_seq, sampled_log_probs, outputs = model.sample(images, init_features, persist_features,
                                                                       max_seq_length=sample_len,
                                                                       start_token_id=vocab('<start>'),
                                                                       trigram_penalty_alpha=args.trigram_penalty_alpha,
                                                                       stochastic_sampling=True, output_logprobs=True,
                                                                       output_outputs=True)
                sampled_seq = model.decoder.alt_prob_to_tensor(sampled_seq, device=device)

                greedy_sampled_seq = model.sample(images, init_features, persist_features,
                                                  trigram_penalty_alpha=args.trigram_penalty_alpha,
                                                  max_seq_length=sample_len, start_token_id=vocab('<start>'),
                                                  stochastic_sampling=False)
                greedy_sampled_seq = model.decoder.alt_prob_to_tensor(greedy_sampled_seq, device=device)

            if args.self_critical_loss in ['sc', 'sc_with_diversity', 'sc_with_relative_diversity', 'sc_with_bleu_diversity', 'sc_with_repetition']:
                loss = criterion(sampled_seq, sampled_log_probs, greedy_sampled_seq,
                                 [gts[i] for i in image_ids], scorers, vocab)
            elif args.self_critical_loss in ['mixed']:
                loss = criterion(sampled_seq, sampled_log_probs, outputs, greedy_sampled_seq,
                                 [gts[i] for i in image_ids], scorers, vocab, targets, lengths,
                                 gamma_ml_rl=args.gamma_ml_rl)
            elif args.self_critical_loss in ['mixed_with_face']:
                loss = criterion(sampled_seq, sampled_log_probs, outputs, greedy_sampled_seq,
                                 [gts[i] for i in image_ids], scorers, vocab, captions, targets, lengths,
                                 gamma_ml_rl=args.gamma_ml_rl)
            else:
                raise ValueError('Invalid self-critical loss')
        else:
            loss = criterion(outputs, targets)

        if params.hierarchical_model:
            _, loss_sent, _, loss_word = criterion.item_terms()
            total_loss_sent += float(loss_sent)
            total_loss_word += float(loss_word)

        total_loss += loss.item()
        num_batches += 1

        if len(scorers) > 0:
            for j in range(len(sampled_ids_batch)):
                jid = image_ids[j]
                if params.hierarchical_model:
                    res[jid] = [paragraph_ids_to_words(sampled_ids_batch[j], vocab).lower()]
                else:
                    res[jid] = [caption_ids_ext_to_words(sampled_ids_batch[j], vocab, skip_start_token=True).lower()]

        # Used for testing:
        if i + 1 == args.num_batches:
            break

    model.train()

    end = datetime.now()

    for score_name, scorer in scorers.items():
        if score_name == 'CIDEr-D':
            continue
        score = scorer.compute_score(gts, res)[0]
        print('Validation', score_name, score)
        stats['validation_' + score_name.lower()] = score

    val_loss = total_loss / num_batches
    stats['validation_loss'] = val_loss

    if params.hierarchical_model:
        stats['val_loss_sentence'] = total_loss_sent / num_batches
        stats['val_loss_word'] = total_loss_word / num_batches
    print('Epoch {} validation duration: {}, validation average loss: {:.4f}'.format(epoch + 1, end - begin, val_loss))
    return val_loss


def main(args):
    if args.model_name is not None:
        print('Preparing to train model: {}'.format(args.model_name))

    global device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    sc_will_happen = args.self_critical_from_epoch != -1

    if args.validate is None and args.lr_scheduler == 'ReduceLROnPlateau':
        print('ERROR: you need to enable validation in order to use default lr_scheduler (ReduceLROnPlateau)')
        print('Hint: use something like --validate=coco:val2017')
        sys.exit(1)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    scorers = {}
    if args.validation_scoring is not None or sc_will_happen:
        assert not (args.validation_scoring is None and sc_will_happen), "Please provide a metric when using self-critical training"
        for s in args.validation_scoring.split(','):
            s = s.lower().strip()
            if s == 'cider':
                from eval.cider import Cider
                scorers['CIDEr'] = Cider()
            if s == 'ciderd':
                from eval.ciderD.ciderD import CiderD
                scorers['CIDEr-D'] = CiderD(df=args.cached_words)

    ########################
    # Set Model parameters #
    ########################

    # Store parameters gotten from arguments separately:
    arg_params = ModelParams.fromargs(args)

    print("Model parameters inferred from command arguments: ")
    print(arg_params)
    start_epoch = 0

    ###############################
    # Load existing model state   #
    # and update Model parameters #
    ###############################

    state = None

    if args.load_model:
        try:
            state = torch.load(args.load_model, map_location=device)
        except AttributeError:
            print('WARNING: Old model found. Please use model_update.py in the model before executing this script.')
            exit(1)
        new_external_features = arg_params.features.external

        params = ModelParams(state, arg_params=arg_params)
        if len(new_external_features) and params.features.external != new_external_features:
            print('WARNING: external features changed: ', params.features.external, new_external_features)
            print('Updating feature paths...')
            params.update_ext_features(new_external_features)
        start_epoch = state['epoch']
        print('Loaded model {} at epoch {}'.format(args.load_model, start_epoch))
    else:
        params = arg_params
        params.command_history = []

    if params.rnn_hidden_init == 'from_features' and params.skip_start_token:
        print("ERROR: Please remove --skip_start_token if you want to use image features "
              " to initialize hidden and cell states. <start> token is needed to trigger "
              " the process of sequence generation, since we don't have image features "
              " embedding as the first input token.")
        sys.exit(1)

    # Force set the following hierarchical model parameters every time:
    if arg_params.hierarchical_model:
        params.hierarchical_model = True
        params.max_sentences = arg_params.max_sentences
        params.weight_sentence_loss = arg_params.weight_sentence_loss
        params.weight_word_loss = arg_params.weight_word_loss
        params.dropout_stopping = arg_params.dropout_stopping
        params.dropout_fc = arg_params.dropout_fc
        params.coherent_sentences = arg_params.coherent_sentences
        params.coupling_alpha = arg_params.coupling_alpha
        params.coupling_beta = arg_params.coupling_beta

    assert args.replace or \
        not os.path.isdir(os.path.join(args.output_root, args.model_path, get_model_name(args, params))) or \
        not (args.load_model and not args.validate_only), \
        '{} already exists. If you want to replace it or resume training please use --replace flag. ' \
        'If you want to validate a loaded model without training it, use --validate_only flag.'  \
        'Otherwise specify a different model name using --model_name flag.'\
        .format(os.path.join(args.output_root, args.model_path, get_model_name(args, params)))

    if args.load_model:
        print("Final model parameters (loaded model + command arguments): ")
        print(params)

    ##############################
    # Load dataset configuration #
    ##############################

    dataset_configs = DatasetParams(args.dataset_config_file)

    if args.dataset is None and not args.validate_only:
        print('ERROR: No dataset selected!')
        print('Please supply a training dataset with the argument --dataset DATASET')
        print('The following datasets are configured in {}:'.format(args.dataset_config_file))
        for ds, _ in dataset_configs.config.items():
            if ds not in ('DEFAULT', 'generic'):
                print(' ', ds)
        sys.exit(1)

    if args.validate_only:
        if args.load_model is None:
            print('ERROR: for --validate_only you need to specify a model to evaluate using --load_model MODEL')
            sys.exit(1)
    else:
        dataset_params = dataset_configs.get_params(args.dataset)

        for i in dataset_params:
            i.config_dict['no_tokenize'] = args.no_tokenize
            i.config_dict['show_tokens'] = args.show_tokens
            i.config_dict['skip_start_token'] = params.skip_start_token

            if params.hierarchical_model:
                i.config_dict['hierarchical_model'] = True
                i.config_dict['max_sentences'] = params.max_sentences
                i.config_dict['crop_regions'] = False

    if args.validate is not None:
        validation_dataset_params = dataset_configs.get_params(args.validate)
        for i in validation_dataset_params:
            i.config_dict['no_tokenize'] = args.no_tokenize
            i.config_dict['show_tokens'] = args.show_tokens
            i.config_dict['skip_start_token'] = params.skip_start_token

            if params.hierarchical_model:
                i.config_dict['hierarchical_model'] = True
                i.config_dict['max_sentences'] = params.max_sentences
                i.config_dict['crop_regions'] = False

    #######################
    # Load the vocabulary #
    #######################

    # For pre-trained models attempt to obtain
    # saved vocabulary from the model itself:
    if args.load_model and params.vocab is not None:
        print("Loading vocabulary from the model file:")
        vocab = params.vocab
    else:
        if args.vocab is None:
            print("ERROR: You must specify the vocabulary to be used for training using "
                  "--vocab flag.\nTry --vocab AUTO if you want the vocabulary to be "
                  "either generated from the training dataset or loaded from cache.")
            sys.exit(1)
        print("Loading / generating vocabulary:")
        vocab = get_vocab(args, dataset_params)

    print('Size of the vocabulary is {}'.format(len(vocab)))

    ##########################
    # Initialize data loader #
    ##########################

    ext_feature_sets = [params.features.external, params.persist_features.external]
    if not args.validate_only:
        print('Loading dataset: {} with {} workers'.format(args.dataset, args.num_workers))
        if params.skip_start_token:
            print("Skipping the use of <start> token...")
        data_loader, ef_dims = get_loader(dataset_params, vocab, transform, args.batch_size,
                                          shuffle=True, num_workers=args.num_workers,
                                          ext_feature_sets=ext_feature_sets,
                                          skip_images=not params.has_internal_features(),
                                          verbose=args.verbose, unique_ids=sc_will_happen)
        if sc_will_happen:
            gts_sc = get_ground_truth_captions(data_loader.dataset)

    gts_sc_valid = None
    if args.validate is not None:
        valid_loader, ef_dims = get_loader(validation_dataset_params, vocab, transform,
                                           args.batch_size, shuffle=True,
                                           num_workers=args.num_workers,
                                           ext_feature_sets=ext_feature_sets,
                                           skip_images=not params.has_internal_features(),
                                           verbose=args.verbose)
        gts_sc_valid = get_ground_truth_captions(valid_loader.dataset) if sc_will_happen else None

    #########################################
    # Setup (optional) TensorBoardX logging #
    #########################################

    writer = None
    if args.tensorboard:
        if SummaryWriter is not None:
            model_name = get_model_name(args, params)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            log_dir = os.path.join(args.output_root, 'log_tb/{}_{}'.format(model_name, timestamp))
            writer = SummaryWriter(log_dir=log_dir)
            print("INFO: Logging TensorBoardX events to {}".format(log_dir))
        else:
            print("WARNING: SummaryWriter object not available. "
                  "Hint: Please install TensorBoardX using pip install tensorboardx")

    ######################
    # Build the model(s) #
    ######################

    # Set per parameter learning rate here, if supplied by the user:

    if args.lr_word_decoder is not None:
        if not params.hierarchical_model:
            print("ERROR: Setting word decoder learning rate currently supported in Hierarchical Model only.")
            sys.exit(1)

        lr_dict = {'word_decoder': args.lr_word_decoder}
    else:
        lr_dict = {}

    model = EncoderDecoder(params, device, len(vocab), state, ef_dims, lr_dict=lr_dict)

    ######################
    # Optimizer and loss #
    ######################

    sc_activated = False
    opt_params = model.get_opt_params()

    # Loss and optimizer
    if params.hierarchical_model:
        criterion = HierarchicalXEntropyLoss(weight_sentence_loss=params.weight_sentence_loss,
                                             weight_word_loss=params.weight_word_loss)
    elif args.share_embedding_weights:
        criterion = SharedEmbeddingXentropyLoss(param_lambda=0.15)
    else:
        criterion = nn.CrossEntropyLoss()

    if sc_will_happen:  # save it for later
        if args.self_critical_loss == 'sc':
            from model.loss import SelfCriticalLoss
            rl_criterion = SelfCriticalLoss()
        elif args.self_critical_loss == 'sc_with_diversity':
            from model.loss import SelfCriticalWithDiversityLoss
            rl_criterion = SelfCriticalWithDiversityLoss()
        elif args.self_critical_loss == 'sc_with_relative_diversity':
            from model.loss import SelfCriticalWithRelativeDiversityLoss
            rl_criterion = SelfCriticalWithRelativeDiversityLoss()
        elif args.self_critical_loss == 'sc_with_bleu_diversity':
            from model.loss import SelfCriticalWithBLEUDiversityLoss
            rl_criterion = SelfCriticalWithBLEUDiversityLoss()
        elif args.self_critical_loss == 'sc_with_repetition':
            from model.loss import SelfCriticalWithRepetitionLoss
            rl_criterion = SelfCriticalWithRepetitionLoss()
        elif args.self_critical_loss == 'mixed':
            from model.loss import MixedLoss
            rl_criterion = MixedLoss()
        elif args.self_critical_loss == 'mixed_with_face':
            from model.loss import MixedWithFACELoss
            rl_criterion = MixedWithFACELoss(vocab_size=len(vocab))
        elif args.self_critical_loss in ['sc_with_penalty', 'sc_with_penalty_throughout', 'sc_masked_tokens']:
            raise ValueError('Deprecated loss, use \'sc\' loss')
        else:
            raise ValueError('Invalid self-critical loss')

        print('Selected self-critical loss is', rl_criterion)

        if start_epoch >= args.self_critical_from_epoch:
            criterion = rl_criterion
            sc_activated = True
            print('Self-critical loss training begins')

    # When using CyclicalLR, default learning rate should be always 1.0
    if args.lr_scheduler == 'CyclicalLR':
        default_lr = 1.
    else:
        default_lr = 0.001

    if sc_activated:
        optimizer = torch.optim.Adam(opt_params, lr=args.learning_rate if args.learning_rate else 5e-5, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=default_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(opt_params, lr=default_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(opt_params, lr=default_lr, weight_decay=args.weight_decay)
    else:
        print('ERROR: unknown optimizer:', args.optimizer)
        sys.exit(1)

    # We don't want to initialize the optimizer if we are transfering
    # the language model from the regular model to hierarchical model
    transfer_language_model = False

    if arg_params.hierarchical_model and state and not state.get('hierarchical_model'):
        transfer_language_model = True

    # Set optimizer state to the one found in a loaded model, unless
    # we are doing a transfer learning step from flat to hierarchical model,
    # or we are using self-critical loss,
    # or the number of unique parameter groups has changed, or the user
    # has explicitly told us *not to* reuse optimizer parameters from before
    if state and not transfer_language_model and not sc_activated and not args.optimizer_reset:
        # Check that number of parameter groups is the same
        if len(optimizer.param_groups) == len(state['optimizer']['param_groups']):
            optimizer.load_state_dict(state['optimizer'])

    # override lr if set explicitly in arguments -
    # 1) Global learning rate:
    if args.learning_rate:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        params.learning_rate = args.learning_rate
    else:
        params.learning_rate = default_lr

    # 2) Parameter-group specific learning rate:
    if args.lr_word_decoder is not None:
        # We want to give user an option to set learning rate for word_decoder
        # separately. Other exceptions can be added as needed:
        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'word_decoder':
                param_group['lr'] = args.lr_word_decoder
                break

    if args.validate is not None and args.lr_scheduler == 'ReduceLROnPlateau':
        print('Using ReduceLROnPlateau learning rate scheduler')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=2)
    elif args.lr_scheduler == 'StepLR':
        print('Using StepLR learning rate scheduler with step_size {}'.format(args.lr_step_size))
        # Decrease the learning rate by the factor of gamma at every
        # step_size epochs (for example every 5 or 10 epochs):
        step_size = args.lr_step_size
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.5, last_epoch=-1)
    elif args.lr_scheduler == 'CyclicalLR':
        print("Using Cyclical learning rate scheduler, lr range: [{},{}]".format(args.lr_cyclical_min,
                                                                                 args.lr_cyclical_max))

        step_size = len(data_loader)
        clr = cyclical_lr(step_size, min_lr=args.lr_cyclical_min, max_lr=args.lr_cyclical_max)
        n_groups = len(optimizer.param_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr] * n_groups)
    elif args.lr_scheduler is not None:
        print('ERROR: Invalid learing rate scheduler specified: {}'.format(args.lr_scheduler))
        sys.exit(1)

    ###################
    # Train the model #
    ###################

    stats_postfix = None
    if args.validate_only:
        stats_postfix = args.validate
    if args.load_model:
        all_stats = init_stats(args, params, postfix=stats_postfix)
    else:
        all_stats = {}

    if args.force_epoch:
        start_epoch = args.force_epoch - 1

    if not args.validate_only:
        total_step = len(data_loader)
        print('Start training with start_epoch={:d} num_epochs={:d} num_batches={:d} ...'.
              format(start_epoch, args.num_epochs, args.num_batches))

    if args.teacher_forcing != 'always':
        print('\t k: {}'.format(args.teacher_forcing_k))
        print('\t beta: {}'.format(args.teacher_forcing_beta))
    print('Optimizer:', optimizer)

    if args.validate_only:
        stats = {}
        teacher_p = 1.0
        if args.teacher_forcing != 'always':
            print('WARNING: teacher_forcing!=always, not yet implemented for --validate_only mode')

        epoch = start_epoch - 1
        if str(epoch + 1) in all_stats.keys() and args.skip_existing_validations:
            print('WARNING: epoch {} already validated, skipping...'.format(epoch + 1))
            return

        val_loss = do_validate(model, valid_loader, criterion, scorers, vocab, teacher_p, args, params, stats, epoch,
                               sc_activated, gts_sc_valid)
        all_stats[str(epoch + 1)] = stats
        save_stats(args, params, all_stats, postfix=stats_postfix)
    else:
        for epoch in range(start_epoch, args.num_epochs):
            stats = {}
            begin = datetime.now()

            total_loss = 0

            if params.hierarchical_model:
                total_loss_sent = 0
                total_loss_word = 0

            num_batches = 0
            vocab_counts = {'cnt': 0, 'max': 0, 'min': 9999,
                            'sum': 0, 'unk_cnt': 0, 'unk_sum': 0}

            # If start self critical training
            if not sc_activated and sc_will_happen and epoch >= args.self_critical_from_epoch:
                if all_stats:
                    best_ep, best_cider = max([(ep, all_stats[ep]['validation_cider']) for ep in all_stats],
                                              key=lambda x: x[1])
                    print('Loading model from epoch', best_ep, 'which has the better score with', best_cider)
                    state = torch.load(get_model_path(args, params, int(best_ep)))
                    model = EncoderDecoder(params, device, len(vocab), state, ef_dims, lr_dict=lr_dict)
                    opt_params = model.get_opt_params()

                optimizer = torch.optim.Adam(opt_params, lr=5e-5, weight_decay=args.weight_decay)
                criterion = rl_criterion
                print('Self-critical loss training begins')
                sc_activated = True

            for i, data in enumerate(data_loader):

                if params.hierarchical_model:
                    (images, captions, lengths, image_ids, features, sorting_order,
                     last_sentence_indicator) = data
                    sorting_order = sorting_order.to(device)
                else:
                    (images, captions, lengths, image_ids, features) = data

                if epoch == 0:
                    unk = vocab('<unk>')
                    for j in range(captions.shape[0]):
                        # Flatten the caption in case it's a paragraph
                        # this is harmless for regular captions too:
                        xl = captions[j, :].view(-1)
                        xw = xl > unk
                        xu = xl == unk
                        xwi = sum(xw).item()
                        xui = sum(xu).item()
                        vocab_counts['cnt'] += 1
                        vocab_counts['sum'] += xwi
                        vocab_counts['max'] = max(vocab_counts['max'], xwi)
                        vocab_counts['min'] = min(vocab_counts['min'], xwi)
                        vocab_counts['unk_cnt'] += xui > 0
                        vocab_counts['unk_sum'] += xui
                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)

                # Remove <start> token from targets if we are initializing the RNN
                # hidden state from image features:
                if params.rnn_hidden_init == 'from_features' and not params.hierarchical_model:
                    # Subtract one from all lengths to match new target lengths:
                    lengths = [x - 1 if x > 0 else x for x in lengths]
                    targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
                else:
                    if params.hierarchical_model:
                        targets = prepare_hierarchical_targets(last_sentence_indicator, args.max_sentences, lengths,
                                                               captions, device)
                    else:
                        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                        sorting_order = None

                init_features = features[0].to(device) if len(features) > 0 and features[0] is not None else None
                persist_features = features[1].to(device) if len(features) > 1 and features[1] is not None else None

                # Forward, backward and optimize
                # Calculate the probability whether to use teacher forcing or not:

                # Iterate over batches:
                iteration = (epoch - start_epoch) * len(data_loader) + i

                teacher_p = get_teacher_prob(args.teacher_forcing_k, iteration, args.teacher_forcing_beta)

                # Allow model to log values at the last batch of the epoch
                writer_data = None
                if writer and (i == len(data_loader) - 1 or i == args.num_batches - 1):
                    writer_data = {'writer': writer, 'epoch': epoch + 1}

                sample_len = captions.size(1) if args.self_critical_loss in ['mixed', 'mixed_with_face'] else 20
                if sc_activated:
                    sampled_seq, sampled_log_probs, outputs = model.sample(images, init_features, persist_features,
                                                                           max_seq_length=sample_len,
                                                                           start_token_id=vocab('<start>'),
                                                                           trigram_penalty_alpha=args.trigram_penalty_alpha,
                                                                           stochastic_sampling=True,
                                                                           output_logprobs=True, output_outputs=True)
                    sampled_seq = model.decoder.alt_prob_to_tensor(sampled_seq, device=device)
                else:
                    outputs = model(images, init_features, captions, lengths, persist_features, teacher_p,
                                    args.teacher_forcing, sorting_order, writer_data=writer_data)

                if args.share_embedding_weights:
                    # Weights of (HxH) projection matrix used for regularizing
                    # models that share embedding weights
                    projection = model.decoder.projection.weight
                    loss = criterion(projection, outputs, targets)
                elif sc_activated:
                    # get greedy decoding baseline
                    model.eval()
                    with torch.no_grad():
                        greedy_sampled_seq = model.sample(images, init_features, persist_features,
                                                          max_seq_length=sample_len, start_token_id=vocab('<start>'),
                                                          trigram_penalty_alpha=args.trigram_penalty_alpha,
                                                          stochastic_sampling=False)
                        greedy_sampled_seq = model.decoder.alt_prob_to_tensor(greedy_sampled_seq, device=device)
                    model.train()

                    if args.self_critical_loss in ['sc', 'sc_with_diversity', 'sc_with_relative_diversity',
                                                   'sc_with_bleu_diversity', 'sc_with_repetition']:
                        loss, advantage = criterion(sampled_seq, sampled_log_probs, greedy_sampled_seq,
                                                    [gts_sc[i] for i in image_ids], scorers, vocab, return_advantage=True)
                    elif args.self_critical_loss in ['mixed']:
                        loss, advantage = criterion(sampled_seq, sampled_log_probs, outputs, greedy_sampled_seq,
                                                    [gts_sc[i] for i in image_ids], scorers, vocab, targets, lengths,
                                                    gamma_ml_rl=args.gamma_ml_rl, return_advantage=True)
                    elif args.self_critical_loss in ['mixed_with_face']:
                        loss, advantage = criterion(sampled_seq, sampled_log_probs, outputs, greedy_sampled_seq,
                                                    [gts_sc[i] for i in image_ids], scorers, vocab, captions, targets,
                                                    lengths, gamma_ml_rl=args.gamma_ml_rl, return_advantage=True)
                    else:
                        raise ValueError('Invalid self-critical loss')

                    if writer is not None and i % 100 == 0:
                        writer.add_scalar('training_loss', loss.item(), epoch * len(data_loader) + i)
                        writer.add_scalar('advantage', advantage, epoch * len(data_loader) + i)
                        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(data_loader) + i)
                else:
                    loss = criterion(outputs, targets)

                model.zero_grad()
                loss.backward()

                # Clip gradients if desired:
                if args.grad_clip is not None:
                    # grad_norms = [x.grad.data.norm(2) for x in opt_params]
                    # batch_max_grad = np.max(grad_norms)
                    # if batch_max_grad > 10.0:
                    #     print('WARNING: gradient norms larger than 10.0')

                    # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.1)
                    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.1)
                    clip_gradients(optimizer, args.grad_clip)

                # Update weights:
                optimizer.step()

                # CyclicalLR requires us to update LR at every minibatch:
                if args.lr_scheduler == 'CyclicalLR':
                    scheduler.step()

                total_loss += loss.item()

                num_batches += 1

                if params.hierarchical_model:
                    _, loss_sent, _, loss_word = criterion.item_terms()
                    total_loss_sent += float(loss_sent)
                    total_loss_word += float(loss_word)

                # Print log info
                if (i + 1) % args.log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, '
                          'Perplexity: {:5.4f}'.
                          format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item(), np.exp(loss.item())))
                    sys.stdout.flush()

                    if params.hierarchical_model:
                        weight_sent, loss_sent, weight_word, loss_word = criterion.item_terms()
                        print('Sentence Loss: {:.4f}, '
                              'Word Loss: {:.4f}'.
                              format(float(loss_sent), float(loss_word)))
                        sys.stdout.flush()

                if i + 1 == args.num_batches:
                    break

            end = datetime.now()

            stats['training_loss'] = total_loss / num_batches

            if params.hierarchical_model:
                stats['loss_sentence'] = total_loss_sent / num_batches
                stats['loss_word'] = total_loss_word / num_batches

            print('Epoch {} duration: {}, average loss: {:.4f}'.
                  format(epoch + 1, end - begin, stats['training_loss']))

            save_model(args, params, model.encoder, model.decoder, optimizer, epoch, vocab)

            if epoch == 0:
                vocab_counts['avg'] = vocab_counts['sum'] / vocab_counts['cnt']
                vocab_counts['unk_cnt_per'] = 100 * vocab_counts['unk_cnt'] / vocab_counts['cnt']
                vocab_counts['unk_sum_per'] = 100 * vocab_counts['unk_sum'] / vocab_counts['sum']
                # print(vocab_counts)
                print(('Training data contains {sum} words in {cnt} captions (avg. {avg:.1f} w/c)' +
                       ' with {unk_sum} <unk>s ({unk_sum_per:.1f}%)' +
                       ' in {unk_cnt} ({unk_cnt_per:.1f}%) captions').format(**vocab_counts))

            ############################################
            # Validation loss and learning rate update #
            ############################################

            if args.validate is not None and (epoch + 1) % args.validation_step == 0:
                val_loss = do_validate(model, valid_loader, criterion, scorers, vocab, teacher_p, args, params, stats,
                                       epoch, sc_activated, gts_sc_valid)

                if args.lr_scheduler == 'ReduceLROnPlateau':
                    scheduler.step(val_loss)
            elif args.lr_scheduler == 'StepLR':
                scheduler.step()

            all_stats[str(epoch + 1)] = stats
            save_stats(args, params, all_stats, writer=writer)

            if writer is not None:
                # Log model data to tensorboard
                log_model_data(params, model, epoch + 1, writer)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    show_versions()
    
    args = args.parse_args()

    begin = datetime.now()
    print('Started training at {}'.format(begin))

    models = args.load_model
    if models is None:
        models = [None]
    if len(models) > 1:
        print('INFO: Iterating over {} models'.format(len(models)))

    for load_model in models:
        args.load_model = load_model
        if args.profiler:
            import cProfile

            cProfile.run('main(args=args)', filename='train.prof')
        else:
            main(args=args)

    end = datetime.now()
    print('Training ended at {}. Total training time: {}'.format(end, end - begin))
