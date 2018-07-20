#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import pickle
import sys
import zipfile
from build_vocab import Vocabulary  # (Needed to handle Vocabulary pickle)

from data_loader import get_loader, ExternalFeature
from datetime import datetime
from model import ModelParams, EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_file_name(args, params, epoch):
    """Create filename based on parameters supplied"""
    bn = args.model_basename
    file_name = '{}-es{}-hs{}-nl{}-bs{}-lr{}-da{}-ep{}.ckpt'.format(bn,
                                                                    params.embed_size,
                                                                    params.hidden_size,
                                                                    params.num_layers,
                                                                    params.batch_size,
                                                                    params.learning_rate,
                                                                    params.dropout,
                                                                    epoch + 1)
    return file_name


def save_models(args, params, encoder, decoder, optimizer, epoch):
    file_name = get_file_name(args, params, epoch)

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
        'dropout': params.dropout
    }

    torch.save(state, os.path.join(args.model_path, file_name))


def main(args):
    if not os.path.exists(args.image_dir):
        print("Image directory or ZIP file not found at {}. Exiting...".format(args.image_dir))
        sys.exit(1)

    if not os.path.exists(args.vocab_path):
        print("Vocabulary file not found at {}. Exiting...".format(args.vocab_path))
        sys.exit(1)

    if not os.path.exists(args.caption_path):
        print("Caption file not found at {}. Exiting...".format(args.caption_path))
        sys.exit(1)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Unzip training images to /tmp/data if image_dir argument points to zip file:
    if zipfile.is_zipfile(args.image_dir):
        # Check if $TMPDIR envirnoment variable is set and use that
        env_tmp = os.environ.get('TMPDIR')
        # Also check if the environment variable points to '/tmp/some/dir' to avoid
        # nasty surprises
        if env_tmp and os.path.commonprefix([os.path.abspath(env_tmp), '/tmp']) == '/tmp':
            tmp_root = os.path.abspath(env_tmp)
        else:
            tmp_root = '/tmp'

        extract_path = os.path.join(tmp_root, args.tmp_dir_prefix)

        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        with zipfile.ZipFile(args.image_dir, 'r') as zipped_images:
            print("Extracting training data from {} to {}".format(
                args.image_dir, extract_path))
            zipped_images.extractall(extract_path)
            unzipped_dir = os.path.basename(args.image_dir).split('.')[0]
            args.image_dir = os.path.join(extract_path, unzipped_dir)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        print("Extracting vocabulary from {}".format(args.vocab_path))
        vocab = pickle.load(f)

    # Build data loader
    print("Loading dataset: {}".format(args.dataset))
    data_loader = get_loader(args.dataset, args.image_dir, args.caption_path,
                             vocab, transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    state = None
    params = ModelParams.fromargs(args)
    start_epoch = 0

    # Intelligently resume from the newest trained epoch matching supplied configuration:
    if args.resume:
        print("Attempting to resume from latest epoch matching supplied parameters...")
        # Get a matching filename without the epoch part
        model_no_epoch = get_file_name(args, params, 0).split('ep')[0]

        # Files matching model:
        full_path_prefix = os.path.join(args.model_path, model_no_epoch)
        matching_files = glob.glob(full_path_prefix + 'ep*.ckpt')

        print("Looking for: {}".format(full_path_prefix + 'ep*.ckpt'))

        # get a file name with a largest matching epoch:
        file_regex = full_path_prefix + 'ep([0-9]*).ckpt'
        r = re.compile(file_regex)
        last_epoch = -1

        for file in matching_files:
            m = r.match(file)
            if m:
                matched_epoch = int(m.group(1))
                if matched_epoch > last_epoch:
                    last_epoch = matched_epoch

        if last_epoch is not -1:
            args.load_model = '{}ep{}.ckpt'.format(full_path_prefix, last_epoch)
            print('Found matching model: {}'.format(args.load_model))
        else:
            print("Warning: Failed to intelligently resume...")

    if args.load_model:
        state = torch.load(args.load_model)
        params = ModelParams(state)
        start_epoch = state['epoch']
        print('Loading model {} at epoch {}.'.format(args.load_model,
                                                     start_epoch))

    if args.force_epoch:
        start_epoch = args.force_epoch - 1

    # Construct external feature loaders
    ef_loaders = []
    params.external_features_total_dim = 0
    for fn in params.external_features:
        ef = ExternalFeature(fn)
        ef_loaders.append(ef)
        params.external_features_total_dim += ef.vdim()

    # Build the models
    print('Using device: {}'.format(device.type))
    print('Initializing model...')
    encoder = EncoderCNN(params).to(device)
    decoder = DecoderRNN(params, len(vocab)).to(device)
    if state:
        encoder.load_state_dict(state['encoder'])
        decoder.load_state_dict(state['decoder'])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    opt_params = (list(decoder.parameters()) +
                  list(encoder.linear.parameters()) +
                  list(encoder.bn.parameters()))
    optimizer = torch.optim.Adam(opt_params, lr=args.learning_rate)
    if state:
        optimizer.load_state_dict(state['optimizer'])

    # Train the models
    total_step = len(data_loader)
    print('Start Training...')
    for epoch in range(start_epoch, args.num_epochs):
        for i, (images, captions, lengths, image_ids) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Get the features batches from each of the external features
            ef_batches = [ef.get_batch(image_ids).to(device)
                          for ef in ef_loaders]
            
            # Forward, backward and optimize
            features = encoder(images, ef_batches)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if (i + 1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch + 1, args.num_epochs, i + 1, total_step,
                              loss.item(), np.exp(loss.item())))
                sys.stdout.flush()

        save_models(args, params, encoder, decoder, optimizer, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='which dataset to use')
    parser.add_argument('--load_model', type=str,
                        help='existing model, for continuing training')
    parser.add_argument('--model_basename', type=str, default='model',
                        help='base name for model snapshot filenames')
    parser.add_argument('--model_path', type=str, default='models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='datasets/processed/COCO/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default='datasets/processed/COCO/train2014_resized',
                        help='directory for resized images'
                        'if "image_dir" points to zip archive - extract '
                        'to /tmp/ , use the extracted images to train')
    parser.add_argument('--tmp_dir_prefix', type=str, default='image_captioning',
                        help='where in /tmp folder to store project data')
    parser.add_argument('--caption_path', type=str,
                        default='datasets/data/COCO/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--resume', action="store_true",
                        help="Resume from largest epoch checkpoint matching \
                        current parameters")
    
    # Model parameters
    parser.add_argument('--features', type=str, default='resnet152',
                        help='features to use as comma separated list, '
                        'features ending with .npy are assumed to be '
                        'precalculated features read from the named npy file, '
                        'e.g., resnet152,foo.npy')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout for the LSTM')

    # Training parameters
    parser.add_argument('--force_epoch', type=int, default=0,
                        help='Force start epoch (for broken model files...)')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    begin = datetime.now()
    print('Started training at {}, with parameters:'.format(begin))
    for k, v in vars(args).items():
        print('[args] {}={}'.format(k, v))

    main(args=args)

    end = datetime.now()
    print('Training ended at {}. Total training time: {}.'.format(end, end - begin))
