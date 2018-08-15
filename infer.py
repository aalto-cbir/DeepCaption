#!/usr/bin/env python3

import argparse
import glob
import json
import os
import re
import sys

from datetime import datetime
from PIL import Image

import torch
from torchvision import transforms

from vocabulary import Vocabulary, get_vocab  # (Needed to handle Vocabulary pickle)
from data_loader import get_loader, ExternalFeature, DatasetConfig, DatasetParams
from model import ModelParams, EncoderCNN, DecoderRNN

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x

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


    # Get dataset parameters and vocabulary wrapper:
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, vocab_path = dataset_configs.get_params(args.dataset,
                                                            args.image_dir,
                                                            args.image_files)
    vocab = get_vocab(vocab_path)

    # Build models
    print('Bulding models.')

    state = torch.load(args.model)
    params = ModelParams(state)
    if args.ext_features:
        params.update_ext_features(args.ext_features)
    if args.ext_persist_features:
        params.update_ext_persist_features(args.ext_persist_features)
    print(params)

    # Build data loader
    print("Loading dataset: {}".format(args.dataset))

    ext_feature_sets = [params.features.external, params.persist_features.external]
    data_loader, ef_dims = get_loader(dataset_params, vocab, transform, args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      ext_feature_sets=ext_feature_sets,
                                      subset=args.subset,
                                      skip_images=not params.has_internal_features())

    encoder = EncoderCNN(params, ef_dims[0]).eval()
    decoder = DecoderRNN(params, len(vocab), ef_dims[1]).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    output_data = []

    print('Starting inference...')
    show_progress = sys.stderr.isatty() and not args.verbose
    for i, (images, captions, lengths, image_ids,
            features) in enumerate(tqdm(data_loader, disable=not show_progress)):
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
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
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
    parser.add_argument('--features_path', type=str,
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
    # if not args.image_files and not args.image_dir:
    #     args.image_dir = 'datasets/data/COCO/images/val2014'

    begin = datetime.now()
    print('Started inference at {}.'.format(begin))

    main(args)

    end = datetime.now()
    print('Inference ended at {}. Total time: {}.'.format(end, end - begin))
