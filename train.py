#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import sys
import json

from datetime import datetime
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# (Needed to handle Vocabulary pickle)
from vocabulary import Vocabulary, get_vocab, get_vocab_from_txt
from data_loader import get_loader, DatasetParams
from model import ModelParams, EncoderDecoder, SpatialAttentionEncoderDecoder

# Device configuration now in main()
device = None


def feats_to_str(feats):
    return '+'.join(feats.internal + [os.path.splitext(os.path.basename(f))[0]
                                      for f in feats.external])


# This is to print the float without exponential-notation, and without trailing zeros.
# Normal formatting, e.g.: '{:f}'.format(0.01) produces "0.010000"
def f2s(f):
    return '{:0.16f}'.format(f).rstrip('0')


def get_model_name(args, params):
    """Create model name"""

    if args.model_name is not None:
        model_name = args.model_name
    else:
        bn = args.model_basename

        feat_spec = feats_to_str(params.features)
        if params.has_persist_features():
            feat_spec += '-' + feats_to_str(params.persist_features)

        model_name = ('{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.
                      format(bn, params.embed_size, params.hidden_size, params.num_layers,
                             params.batch_size, args.optimizer, f2s(params.learning_rate),
                             f2s(args.weight_decay), params.dropout, params.encoder_dropout,
                             feat_spec))
    return model_name


def save_model(args, params, encoder, decoder, optimizer, epoch, vocab):
    model_name = get_model_name(args, params)

    state = {
        'epoch': epoch + 1,
        # Attention models can in principle be trained without an encoder:
        'encoder': encoder.state_dict() if encoder is not None else None,
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'embed_size': params.embed_size,
        'hidden_size': params.hidden_size,
        'num_layers': params.num_layers,
        'batch_size': params.batch_size,
        'learning_rate': params.learning_rate,
        'dropout': params.dropout,
        'encoder_dropout': params.encoder_dropout,
        'features': params.features,
        'persist_features': params.persist_features,
        'attention': params.attention,
        'vocab': vocab
    }

    file_name = 'ep{}.model'.format(epoch + 1)

    model_path = os.path.join(args.model_path, model_name, file_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(state, model_path)
    print('Saved model as {}'.format(model_path))
    if args.verbose:
        print(params)


def stats_filename(args, params):
    model_name = get_model_name(args, params)
    model_dir = os.path.join(args.model_path, model_name)
    return os.path.join(model_dir, 'train_stats.json')


def init_stats(args, params):
    filename = stats_filename(args, params)
    if os.path.exists(filename):
        with open(filename, 'r') as fp:
            return json.load(fp)
    else:
        return dict()


def save_stats(args, params, all_stats):
    filename = stats_filename(args, params)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as outfile:
        json.dump(all_stats, outfile, indent=2)


def find_matching_model(args, params):
    """Get a model file matching the parameters given with the latest trained epoch"""
    print('Attempting to resume from latest epoch matching supplied '
          'parameters...')
    # Get a matching filename without the epoch part
    model_name = get_model_name(args, params)

    # Files matching model:
    full_path_prefix = os.path.join(args.model_path, model_name)
    matching_files = glob.glob(full_path_prefix + '*.model')

    print("Looking for: {}".format(full_path_prefix + '*.model'))

    # get a file name with a largest matching epoch:
    file_regex = full_path_prefix + '/ep([0-9]*).model'
    r = re.compile(file_regex)
    last_epoch = 0

    for file in matching_files:
        m = r.match(file)
        if m:
            matched_epoch = int(m.group(1))
            if matched_epoch > last_epoch:
                last_epoch = matched_epoch

    model_file_path = None
    if last_epoch:
        model_file_name = 'ep{}.model'.format(last_epoch)
        model_file_path = os.path.join(full_path_prefix, model_file_name)
        print('Found matching model: {}'.format(args.load_model))
    else:
        print("Warning: Failed to intelligently resume...")

    return model_file_path


def get_teacher_prob(k, i, beta=1):
    """Inverse sigmoid sampling scheduler determines the probability
    with which teacher forcing is turned off, more info here:
    https://arxiv.org/pdf/1506.03099.pdf"""
    if k == 0:
        return 1.0

    i = i * beta
    p = k / (k + np.exp(i / k))

    return p


def main(args):
    global device
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.cpu else 'cpu')

    if args.validate is None and args.lr_scheduler:
        print('ERROR: you need to enable validation in order to use the lr_scheduler')
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

    state = None

    # Get dataset parameters and vocabulary wrapper:
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, vocab_path = dataset_configs.get_params(args.dataset,
                                                            vocab_path=args.vocab_path)
    for i in dataset_params:
        i.config_dict['no_tokenize'] = args.no_tokenize
        i.config_dict['show_tokens'] = args.show_tokens

    vocab_is_txt = re.search('\\.txt$', vocab_path)
    vocab = get_vocab_from_txt(vocab_path) if vocab_is_txt else get_vocab(vocab_path)
    print('Size of the vocabulary is {}'.format(len(vocab)))

    if False:
        vocl = vocab.get_list()
        with open('vocab-dump.txt', 'w') as vocf:
            print('\n'.join(vocl), file=vocf)

    if args.validate is not None:
        validation_dataset_params, _ = dataset_configs.get_params(args.validate)
        for i in validation_dataset_params:
            i.config_dict['no_tokenize'] = args.no_tokenize
            i.config_dict['show_tokens'] = args.show_tokens

    params = ModelParams.fromargs(args)
    print(params)
    start_epoch = 0

    # Intelligently resume from the newest trained epoch matching
    # supplied configuration:
    if args.resume:
        args.load_model = find_matching_model(args, params)

    if args.load_model:
        state = torch.load(args.load_model)
        external_features = params.features.external
        params = ModelParams(state)
        if params.features.external != external_features:
            print('WARNING: external features changed: ',
                  params.features.external, external_features)
            print('Updating feature paths...')
            params.update_ext_features(args.features)
        start_epoch = state['epoch']
        print('Loading model {} at epoch {}.'.format(args.load_model,
                                                     start_epoch))
        print(params)

    if args.force_epoch:
        start_epoch = args.force_epoch - 1

    # Build data loader
    print('Loading dataset: {} with {} workers'.format(args.dataset, args.num_workers))
    ext_feature_sets = [params.features.external, params.persist_features.external]
    data_loader, ef_dims = get_loader(dataset_params, vocab, transform, args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      ext_feature_sets=ext_feature_sets,
                                      skip_images=not params.has_internal_features(),
                                      verbose=args.verbose)

    if args.validate is not None:
        valid_loader, _ = get_loader(validation_dataset_params, vocab, transform,
                                     args.batch_size, shuffle=True,
                                     num_workers=args.num_workers,
                                     ext_feature_sets=ext_feature_sets,
                                     skip_images=not params.has_internal_features(),
                                     verbose=args.verbose)

    # Build the models
    if args.attention is None:
        _Model = EncoderDecoder
    else:
        _Model = SpatialAttentionEncoderDecoder

    model = _Model(params, device, len(vocab), state, ef_dims)

    opt_params = model.get_opt_params()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    default_lr = 0.001
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=default_lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(opt_params, lr=default_lr,
                                        weight_decay=args.weight_decay)
    else:
        print('ERROR: unknown optimizer:', args.optimizer)
        sys.exit(1)

    if state:
        optimizer.load_state_dict(state['optimizer'])

    if args.learning_rate:  # override lr if set explicitly in arguments
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        params.learning_rate = args.learning_rate
    else:
        params.learning_rate = default_lr

    if args.validate is not None and args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                                               patience=2)

    # Train the models
    total_step = len(data_loader)
    if args.load_model:
        all_stats = init_stats(args, params)
    else:
        all_stats = {}

    print('Start training with num_epochs={:d} num_batches={:d} ...'.
          format(args.num_epochs, args.num_batches))
    if args.teacher_forcing != 'always':
        print('\t k: {}'.format(args.teacher_forcing_k))
        print('\t beta: {}'.format(args.teacher_forcing_beta))
    print('Optimizer:', optimizer)
    for epoch in range(start_epoch, args.num_epochs):
        stats = {}
        begin = datetime.now()
        total_loss = 0
        num_batches = 0
        for i, (images, captions, lengths, _, features) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths,
                                           batch_first=True)[0]
            init_features = features[0].to(device)if len(features) > 0 and \
                features[0] is not None else None
            persist_features = features[1].to(device) if len(features) > 1 and \
                features[1] is not None else None

            # Forward, backward and optimize
            # Calculate the probability whether to use teacher forcing or not:

            # Iterate over batches:
            iteration = (epoch - start_epoch) * len(data_loader) + i

            teacher_p = get_teacher_prob(args.teacher_forcing_k, iteration,
                                         args.teacher_forcing_beta)

            outputs = model(images, init_features, captions, lengths, persist_features,
                            teacher_p, args.teacher_forcing)

            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()

            # grad_norms = [x.grad.data.norm(2) for x in opt_params]
            # batch_max_grad = np.max(grad_norms)
            # if batch_max_grad > 10.0:
            #     print('WARNING: gradient norms larger than 10.0')

            # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.1)
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.1)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print log info
            if (i + 1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, '
                      'Perplexity: {:5.4f}'.
                      format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item(),
                             np.exp(loss.item())))
                sys.stdout.flush()

            if i + 1 == args.num_batches:
                break

        end = datetime.now()

        stats['training_loss'] = total_loss / num_batches
        print('Epoch {} duration: {}, average loss: {:.4f}.'.format(epoch + 1, end - begin,
                                                                    stats['training_loss']))
        save_model(args, params, model.encoder, model.decoder, optimizer, epoch, vocab)

        if args.validate is not None and (epoch + 1) % args.validation_step == 0:
            begin = datetime.now()
            model.eval()

            total_loss = 0
            num_batches = 0
            for i, (images, captions, lengths, _, features) in enumerate(valid_loader):
                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths,
                                               batch_first=True)[0]
                init_features = features[0].to(device)if len(features) > 0 and \
                    features[0] is not None else None
                persist_features = features[1].to(device) if len(features) > 1 and \
                    features[1] is not None else None

                # Forward, backward and optimize
                with torch.no_grad():
                    outputs = model(images, init_features, captions, lengths, persist_features)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                # Used for testing:
                if i + 1 == args.num_batches:
                    break

            model.train()

            end = datetime.now()
            val_loss = total_loss / num_batches
            stats['validation_loss'] = val_loss
            print('Epoch {} validation duration: {}, validation average loss: {:.4f}.'.format(
                epoch + 1, end - begin, val_loss))

            if args.lr_scheduler:
                scheduler.step(val_loss)

        all_stats[epoch + 1] = stats
        save_stats(args, params, all_stats)


if __name__ == '__main__':
    default_dataset = 'coco:train2014'
    default_features = 'resnet152'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=default_dataset,
                        help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--load_model', type=str,
                        help='existing model, for continuing training')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_basename', type=str, default='model',
                        help='base name for model snapshot filenames')
    parser.add_argument('--model_path', type=str, default='models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='path for vocabulary wrapper')
    parser.add_argument('--tmp_dir_prefix', type=str,
                        default='image_captioning',
                        help='where in /tmp folder to store project data')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--resume', action="store_true",
                        help="Resume from largest epoch checkpoint matching \
                        current parameters")
    parser.add_argument('--verbose', action="store_true", help="Increase verbosity")
    parser.add_argument('--profiler', action="store_true", help="Run in profiler")
    parser.add_argument('--cpu', action="store_true",
                        help="Use CPU even when GPU is available")

    # Model parameters
    parser.add_argument('--features', type=str, default=default_features,
                        help='features to use as the initial input for the '
                        'caption generator, given as comma separated list, '
                        'multiple features are concatenated, '
                        'features ending with .npy are assumed to be '
                        'precalculated features read from the named npy file, '
                        'example: "resnet152,foo.npy"')
    parser.add_argument('--persist_features', type=str,
                        help='features accessible in all caption generation '
                        'steps, given as comma separated list')
    parser.add_argument('--attention', type=str,
                        help='type of attention mechanism to use '
                        ' currently supported types: None, spatial')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout for the LSTM')
    parser.add_argument('--encoder_dropout', type=float, default=0.0,
                        help='dropout for the encoder FC layer')

    # Training parameters
    parser.add_argument('--force_epoch', type=int, default=0,
                        help='Force start epoch (for broken model files...)')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_batches', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--validate', type=str,
                        help='Dataset to validate against after each epoch')
    parser.add_argument('--validation_step', type=int, default=1,
                        help='After how many epochs to perform validation, default=1')
    parser.add_argument('--optimizer', type=str, default="rmsprop")
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--no_tokenize', action='store_true')
    parser.add_argument('--show_tokens', action='store_true')
    # For teacher forcing schedule see - https://arxiv.org/pdf/1506.03099.pdf
    parser.add_argument('--teacher_forcing', type=str, default='always',
                        help='Type of teacher forcing to use for training the Decoder RNN: \n'
                             'always: always use groundruth as LSTM input when training'
                             'sampled: follow a sampling schedule detemined by the value '
                             'of teacher_forcing_parameter\n'
                             'additive: use the sampling schedule formula to determine weight '
                             'ratio between the teacher and model inputs\n'
                             'additive_sampled: combines two of the above modes')
    parser.add_argument('--teacher_forcing_k', type=float, default=6500,
                        help='value of the sampling schedule parameter k. '
                        'Good values can be found in a range between 400 - 20000'
                        'small values = start using model output quickly, large values -'
                        ' wait for a while before start using model output')
    parser.add_argument('--teacher_forcing_beta', type=float, default=1,
                        help='sample scheduling parameter that determins the slope of '
                        'the middle segment of the sigmoid')

    args = parser.parse_args()

    begin = datetime.now()
    print('Started training at {}.'.format(begin))

    if args.profiler:
        import cProfile
        cProfile.run('main(args=args)', filename='train.prof')
    else:
        main(args=args)

    end = datetime.now()
    print('Training ended at {}. Total training time: {}.'.format(end, end - begin))
