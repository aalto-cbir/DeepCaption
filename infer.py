#!/usr/bin/env python3

import argparse
import glob
import json
import os
import pickle
import re
import sys

from datetime import datetime
from PIL import Image

import torch
from torchvision import transforms

from build_vocab import Vocabulary  # (Needed to handle Vocabulary pickle)
from data_loader import get_loader, ExternalFeature
from model import ModelParams, EncoderCNN, DecoderRNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def basename(fname):
    fname.split(':')
    return os.path.splitext(os.path.basename(fname))[0]


def fix_caption(caption):
    m = re.match(r'^<start> (.*?)( <end>)?$', caption)
    if m is None:
        print('ERROR: unexpected caption format: "{}"'.format(caption))
        return caption.capitalize()

    ret = m.group(1)
    ret = re.sub(r'\s([.,])(\s|$)', r'\1\2', ret)
    return ret.capitalize()


def path_from_id(image_dir, image_id):
    """Return image path based on image directory, image id and
    glob matching for extension"""
    return glob.glob(os.path.join(image_dir, image_id) + '.*')[0]


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if image.mode != 'RGB':
        print('WARNING: converting {} from {} to RGB'.
              format(image_path, image.mode))
        image = image.convert('RGB')

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def remove_duplicate_sentences(caption):
    """Removes consecutively repeating sentences from the caption"""
    sentences = caption.split('.')

    no_dupes = [sentences[0]]

    for i, _ in enumerate(sentences):
        if i:
            if sentences[i - 1] == sentences[i]:
                no_dupes.append(sentences[i])

    return '.'.join(no_dupes)


def remove_incomplete_sentences(caption):
    """Removes sentences that don't end with a period (truncated or incomplete)"""
    sentences = caption.split('.')
    if sentences[-1] != '':
        sentences[-1] = ''
        return '.'.join(sentences)
    else:
        return caption


def main(args):
    # Create model directory
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    print('Reading vocabulary from {}.'.format(args.vocab_path))
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    print('Bulding models.')

    state = torch.load(args.model)
    params = ModelParams(state)
    if args.ext_features:
        params.update_ext_features(args.ext_features)
    if args.ext_persist_features:
        params.update_ext_persist_features(args.ext_persist_features)
    print(params)

    # Construct external feature loaders
    (ef_loaders, ef_dim) = ExternalFeature.loaders(params.features.external,
                                                   args.features_path)
    (pef_loaders, pef_dim) = ExternalFeature.loaders(
        params.persist_features.external, args.features_path)

    encoder = EncoderCNN(params, ef_dim).eval()
    decoder = DecoderRNN(params, len(vocab), pef_dim).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    output_data = []

    # Build data loader
    print("Loading dataset: {}".format(args.dataset))

    if args.dataset == 'generic':
        root = []
        if args.image_files:
            root += args.image_files
        if args.image_dir:
            if args.subset:
                root = [path_from_id(args.image_dir, line.rstrip())
                        for line in open(args.subset)]
            else:
                root += glob.glob(args.image_dir + '/*.jpg')
                root += glob.glob(args.image_dir + '/*.jpeg')
                root += glob.glob(args.image_dir + '/*.png')
    else:
        root = args.image_dir

    data_loader = get_loader(args.dataset, root, None,
                             vocab, transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             subset=args.subset, feature_loaders=(ef_loaders, pef_loaders),
                             skip_images=not params.has_internal_features())

    for i, (images, captions, lengths, image_ids, features) in enumerate(data_loader):
        images = images.to(device)

        init_features = features[0].to(device) if len(features) > 0 else None
        persist_features = features[1].to(device) if len(features) > 1 else None

        # Generate a caption from the image
        encoded = encoder(images, init_features)
        sampled_ids_batch = decoder.sample(encoded, images, persist_features,
                                           max_seq_length=args.max_seq_length)

        for i in range(sampled_ids_batch.shape[0]):
            sampled_ids = sampled_ids_batch[i].cpu().numpy()

            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            caption = fix_caption(' '.join(sampled_caption))

            if args.no_repeat_sentences:
                caption = remove_duplicate_sentences(caption)

            if args.only_complete_sentences:
                caption = remove_incomplete_sentences(caption)

            if args.verbose:
                print('=>', caption)

            output_data.append({'caption': caption, 'image_id': image_ids[i]})

    output_file = None
    if not args.output_file and not args.print_results:
        output_file = basename(args.model) + '.json'
    else:
        output_file = args.output_file

    if output_file:
        json.dump(output_data, open(os.path.join(args.results_path, output_file), 'w'))

    if args.print_results:
        for d in output_data:
            print('{}: {}'.format(d['image_id'], d['caption']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='generic',
                        help='which dataset to use')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('image_files', type=str, nargs='*')
    parser.add_argument('--image_dir', type=str,
                        help='input image dir for generating captions')
    parser.add_argument('--subset', type=str,
                        help='path to (optional) new-line separated file '
                        'listing ids of images to include')
    parser.add_argument('--model', type=str, required=True,
                        help='path to existing model')
    parser.add_argument('--vocab_path', type=str,
                        default='datasets/processed/COCO/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--ext_features', type=str,
                        help='paths for the external features, overrides the '
                        'paths in the model ckpt file (which are the ones '
                        'used for training), comma separated')
    parser.add_argument('--ext_persist_features', type=str,
                        help='paths for external persist features')
    parser.add_argument('--features_path', type=str, default='',
                        help='directory of external feature files, if not '
                        'specified should be given with absolute paths, or '
                        'expected to be found in the working directory')
    parser.add_argument('--output_file', type=str,
                        help='path for output JSON, default: model_name.json')
    parser.add_argument('--verbose', help='verbose output',
                        action='store_true')
    parser.add_argument('--results_path', type=str, default='results/',
                        help='path for saving results')
    parser.add_argument('--print_results', action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=20,
                        help='maximum allowed length of the decoded sequence')
    parser.add_argument('--no_repeat_sentences', action='store_true',
                        help='allow repeating sentences inside a paragraph')
    parser.add_argument('--only_complete_sentences', action='store_true')

    args = parser.parse_args()
    if not args.image_files and not args.image_dir:
        args.image_dir = 'datasets/data/COCO/images/val2014'

    begin = datetime.now()
    print('Started inference at {}, with parameters:'.format(begin))
    for k, v in vars(args).items():
        print('[args] {}={}'.format(k, v))
    sys.stdout.flush()

    main(args)

    end = datetime.now()
    print('Inference ended at {}. Total time: {}.'.format(end, end - begin))
