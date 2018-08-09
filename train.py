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

from build_vocab import Vocabulary  # (Needed to handle Vocabulary pickle)
from data_loader import get_loader, DatasetParams
from model import ModelParams, EncoderCNN, DecoderRNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def feats_to_str(feats):
    return '+'.join(feats.internal + [os.path.splitext(os.path.basename(f))[0]
                                      for f in feats.external])


# This is to print the float without exponential-notation, and without trailing zeros.
# Normal formatting, e.g.: '{:f}'.format(0.01) produces "0.010000"
def f2s(f):
    return '{:0.16f}'.format(f).rstrip('0')


def get_model_name(args, params):
    """Create model name"""
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


def save_model(args, params, encoder, decoder, optimizer, epoch):
    model_name = get_model_name(args, params)

    state = {
        'epoch': epoch + 1,
        'encoder': encoder.state_dict(),
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
    }

    file_name = 'ep{}.model'.format(epoch + 1)

    model_path = os.path.join(args.model_path, model_name, file_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(state, model_path)
    print('Saved model as {}'.format(model_path))
    if args.verbose:
        print(params)


def save_stats(args, params, all_stats):
    model_name = get_model_name(args, params)
    model_dir = os.path.join(args.model_path, model_name)
    os.makedirs(model_dir, exist_ok=True)

    stats_filename = os.path.join(model_dir, 'train_stats.json')
    with open(stats_filename, 'w') as outfile:
        json.dump(all_stats, outfile, indent=2, sort_keys=True)


def main(args):

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
    dataset_params, vocab = DatasetParams.fromargs(args).get_all()
    for ds in dataset_params:
        print('[Dataset]', ds.name)
        for name, value in ds._asdict().items():
            if name != 'name' and value is not None:
                print('    {}: {}'.format(name, value))

    params = ModelParams.fromargs(args)
    print(params)
    start_epoch = 0

    # Intelligently resume from the newest trained epoch matching
    # supplied configuration:
    if args.resume:
        print('Attempting to resume from latest epoch matching supplied '
              'parameters...')
        # Get a matching filename without the epoch part
        model_name = get_model_name(args, params)

        # Files matching model:
        full_path_prefix = os.path.join(args.model_path, model_name, model_name)
        matching_files = glob.glob(full_path_prefix + '*.model')

        print("Looking for: {}".format(full_path_prefix + '*.model'))

        # get a file name with a largest matching epoch:
        file_regex = full_path_prefix + '-ep([0-9]*).model'
        r = re.compile(file_regex)
        last_epoch = -1

        for file in matching_files:
            m = r.match(file)
            if m:
                matched_epoch = int(m.group(1))
                if matched_epoch > last_epoch:
                    last_epoch = matched_epoch

        if last_epoch is not -1:
            args.load_model = '{}-ep{}.model'.format(full_path_prefix, last_epoch)
            print('Found matching model: {}'.format(args.load_model))
        else:
            print("Warning: Failed to intelligently resume...")

    if args.load_model:
        state = torch.load(args.load_model)
        external_features = params.features.external
        params = ModelParams(state)
        if params.features.external != external_features:
            print('WARNING: external features changed: ',
                  params.features.external, external_features)
        start_epoch = state['epoch']
        print('Loading model {} at epoch {}.'.format(args.load_model,
                                                     start_epoch))
        print(params)

    if args.force_epoch:
        start_epoch = args.force_epoch - 1

    # Build data loader
    print("Loading dataset: {}".format(args.dataset))
    ext_feature_sets = [params.features.external, params.persist_features.external]
    data_loader, ef_dims = get_loader(dataset_params, vocab, transform, args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      ext_feature_sets=ext_feature_sets,
                                      skip_images=not params.has_internal_features())

    if args.validation > 0:
        valid_loader, _ = get_loader(dataset_params, vocab, transform, args.batch_size,
                                     subset='validate',
                                     shuffle=True, num_workers=args.num_workers,
                                     ext_feature_sets=ext_feature_sets,
                                     skip_images=not params.has_internal_features())

    # Build the models
    print('Using device: {}'.format(device.type))
    print('Initializing model...')
    encoder = EncoderCNN(params, ef_dims[0]).to(device)
    decoder = DecoderRNN(params, len(vocab), ef_dims[1]).to(device)
    if state:
        encoder.load_state_dict(state['encoder'])
        decoder.load_state_dict(state['decoder'])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    opt_params = (list(decoder.parameters()) +
                  list(encoder.linear.parameters()) +
                  list(encoder.bn.parameters()))
    default_lr = 0.001
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=default_lr, weight_decay=args.weight_decay)
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

    # Train the models
    total_step = len(data_loader)
    all_stats = {}

    print('Start Training... ')
    print('Optimizer:', optimizer)
    for epoch in range(start_epoch, args.num_epochs):
        stats = {}
        begin = datetime.now()
        total_loss = 0
        num_batches = 0
        for i, (images, captions, lengths, image_ids, features) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths,
                                           batch_first=True)[0]
            init_features = features[0].to(device) if len(features) > 0 else None
            persist_features = features[1].to(device) if len(features) > 1 else None

            # Forward, backward and optimize
            encoded = encoder(images, init_features)
            outputs = decoder(encoded, captions, lengths, images, persist_features)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()

            grad_norms = [x.grad.data.norm(2) for x in opt_params]
            batch_max_grad = np.max(grad_norms)
            if batch_max_grad > 10.0:
                print('WARNING: gradient norms larger than 10.0')

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

        end = datetime.now()

        stats['training_loss'] = total_loss/num_batches
        print('Epoch {} duration: {}, average loss: {:.4f}.'.format(epoch + 1, end - begin,
                                                                    stats['training_loss']))
        save_model(args, params, encoder, decoder, optimizer, epoch)

        if args.validation > 0 and (epoch + 1) % args.validation == 0:
            begin = datetime.now()
            encoder = encoder.eval()
            decoder = decoder.eval()

            total_loss = 0
            num_batches = 0
            for i, (images, captions, lengths, image_ids, features) in enumerate(valid_loader):
                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths,
                                               batch_first=True)[0]
                init_features = features[0].to(device) if len(features) > 0 else None
                persist_features = features[1].to(device) if len(features) > 1 else None

                # Forward, backward and optimize
                with torch.no_grad():
                    encoded = encoder(images, init_features)
                    outputs = decoder(encoded, captions, lengths, images, persist_features)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

            encoder = encoder.train()
            decoder = decoder.train()

            end = datetime.now()
            stats['validation_loss'] = total_loss/num_batches
            print('Epoch {} validation duration: {}, validation average loss: {:.4f}.'.format(
                epoch + 1, end - begin, stats['validation_loss']))

        all_stats[epoch+1] = stats
        save_stats(args, params, all_stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--subset', type=str, default=None,
                        help='file defining the subset of training images')
    parser.add_argument('--load_model', type=str,
                        help='existing model, for continuing training')
    parser.add_argument('--model_basename', type=str, default='model',
                        help='base name for model snapshot filenames')
    parser.add_argument('--model_path', type=str, default='models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='directory for resized images'
                        'if "image_dir" points to zip archive - extract '
                        'to /tmp/ , use the extracted images to train')
    parser.add_argument('--tmp_dir_prefix', type=str,
                        default='image_captioning',
                        help='where in /tmp folder to store project data')
    parser.add_argument('--caption_path', type=str, default=None,
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--resume', action="store_true",
                        help="Resume from largest epoch checkpoint matching \
                        current parameters")
    parser.add_argument('--verbose', action="store_true", help="Increase verbosity")

    # Model parameters
    parser.add_argument('--features', type=str, default='resnet152',
                        help='features to use as the initial input for the '
                        'caption generator, given as comma separated list, '
                        'multiple features are concatenated, '
                        'features ending with .npy are assumed to be '
                        'precalculated features read from the named npy file, '
                        'example: "resnet152,foo.npy"')
    parser.add_argument('--persist_features', type=str,
                        help='features accessible in all caption generation '
                        'steps, given as comma separated list')
    parser.add_argument('--features_path', type=str,
                        help='directory of external feature files, if not '
                        'specified should be given with absolute paths, or '
                        'expected to be found in the working directory')
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--validation', type=int, default=0,
                        help='Validate at every VALIDATION epochs, 0 means never validate.')
    parser.add_argument('--optimizer', type=str, default="rmsprop")
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    args = parser.parse_args()

    begin = datetime.now()
    print('Started training at {}.'.format(begin))

    # import cProfile
    # cProfile.run('main(args=args)', filename='train.prof')
    main(args=args)

    end = datetime.now()
    print('Training ended at {}. Total training time: {}.'.
          format(end, end - begin))
