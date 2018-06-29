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
from model import ModelParams, EncoderCNN, DecoderRNN
from torchvision import transforms
from tqdm import tqdm

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
    ret = re.sub(r'\s([.,])(\s|$)',r'\1\2', ret)
    return ret.capitalize()

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if image.mode != 'RGB':
        print('WARNING: converting {} to RGB'.format(image_path))
        image = image.convert('RGB')
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

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

    # We set encoder to eval mode (BN uses moving mean/variance)
    encoder = EncoderCNN(params).eval() 
    decoder = DecoderRNN(params, len(vocab)).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    output_data = []
    file_list = []

    if args.image_files is not None:
        file_list += args.image_files

    if args.image_dir is not None:
        file_list += glob.glob(args.image_dir + '/*.jpg')
        file_list += glob.glob(args.image_dir + '/*.png')
    
    N = len(file_list)
    print('Processing {} image files.'.format(N))
    show_progress = sys.stderr.isatty() and not args.verbose
    for i, image_file in tqdm(enumerate(file_list), disable=not show_progress):
        bn = basename(image_file)
        m = re.search(r'0*(\d+)$', bn)
        if m is not None:
            image_id = int(m.group(1))
            assert image_id > 0
        else:
            image_id = bn

        if args.verbose:
            print('[{:.2%}] Reading [{}] as {} ...'.format(i / N, image_id, image_file))
        # Prepare an image
        image = load_image(image_file, transform)
        image_tensor = image.to(device)

        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = fix_caption(' '.join(sampled_caption))

        if args.verbose:
            print('=>', sentence)

        output_data.append({'caption': sentence, 'image_id': image_id})

    output_file = None
    if not args.output_file and not args.print_results:
        output_file = basename(args.model) + '.json'
    else:
        output_file = args.output_file

    if output_file:
        json.dump(output_data, open(output_file, 'w'))

    if args.print_results:
        for d in output_data:
            print('{}: {}'.format(d['image_id'], d['caption']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_files', type=str, nargs='*')
    parser.add_argument('--image_dir', type=str,
                        help='input image dir for generating captions')
    parser.add_argument('--model', type=str, required=True,
                        help='path to existing model')
    parser.add_argument('--vocab_path', type=str, default='datasets/processed/COCO/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--output_file', type=str, help='path for output JSON')
    parser.add_argument('--verbose', help='verbose output', action='store_true')
    parser.add_argument('--results_path', type=str, default='results/',
                        help='path for saving results')
    parser.add_argument('--print_results', action='store_true')

    args = parser.parse_args()
    if not args.image_files and not args.image_dir:
        args.image_dir = 'datasets/data/COCO/val2014'

    begin = datetime.now()
    print('Started evaluation at {}, with parameters:'.format(begin))
    for k, v in vars(args).items(): print('[args] {}={}'.format(k, v))
    sys.stdout.flush()

    main(args)

    end = datetime.now()
    print('Evaluation ended at {}. Total time: {}.'.format(end, end-begin))
