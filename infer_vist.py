#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import sys

from datetime import datetime

import torch

from model.vist import ModelParams, EncoderCNN, EncoderRNN, DecoderRNN
from torchvision import transforms
from data_loader import get_loader, collate_fn_vist

from utils import basename, fix_caption, torchify_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # create model directory
    os.makedirs(args.results_path, exist_ok=True)

    # image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # load vocabulary wrapper
    print('Reading vocabulary from {}.'.format(args.vocab_path))
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # build data loader
    data_loader = get_loader(args.dataset, args.image_dir, args.caption_path,
                             vocab, transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, _collate_fn=collate_fn_vist)

    print('data load complete, sequences to infer: ', len(data_loader))

    state = torch.load(args.model)
    params = ModelParams(state)

    # set encoder to eval mode (BN uses moving mean/variance)
    encoder_cnn = EncoderCNN(params).eval()
    encoder_rnn = EncoderRNN(params).eval()
    decoder = DecoderRNN(params, len(vocab)).eval()

    encoder_cnn = encoder_cnn.to(device)
    encoder_rnn = encoder_rnn.to(device)
    decoder = decoder.to(device)

    # load the trained model parameters
    encoder_cnn.load_state_dict(state['encoder_cnn'])
    encoder_rnn.load_state_dict(state['encoder_rnn'])
    decoder.load_state_dict(state['decoder'])

    # use models for inference
    output_data = []
    total_step = len(data_loader)
    print('start Testing...')
    print('processing {} sequences.'.format(total_step))
    for i, (images, captions, lengths, story_id) in enumerate(data_loader):
        # forward pass
        sequence_data = torchify_sequence(images[0]).to(device)
        sequence_features = encoder_cnn(sequence_data)
        input_sequence_features = sequence_features.unsqueeze(0)
        input_sequence_features = input_sequence_features.view(1, 1, -1)
        context_vector = encoder_rnn(input_sequence_features)
        sampled_ids = decoder.sample(context_vector)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        story = fix_caption(' '.join(sampled_caption))

        if args.verbose:
            print('=>', story)

        output_data.append({'story': story, 'story_id': story_id})

    output_file = args.output_file
    if not args.output_file and not args.print_results:
        output_file = basename(args.model, split=':') + '.json'

    if output_file:
        json.dump(output_data, open(os.path.join(args.results_path, output_file), 'w'))

    if args.print_results:
        for d in output_data:
            print('{}: {}'.format(d['image_id'], d['caption']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vist-seq',
                        help='which dataset to use')
    parser.add_argument('--caption_path', type=str,
                        default='./resources/sis/val.story-in-sequence.json',
                        help='path for val annotation json file')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('image_files', type=str, nargs='*')
    parser.add_argument('--image_dir', type=str,
                        help='input image dir for generating captions')
    parser.add_argument('--model', type=str, required=False,
                        help='path to existing model',
                        default='./models/model-ep10.ckpt')
    parser.add_argument('--vocab_path', type=str,
                        default='vocab_train.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--output_file', type=str,
                        help='path for output JSON, default: model_name.json',
                        default='results.json')
    parser.add_argument('--verbose', help='verbose output', action='store_true')
    parser.add_argument('--results_path', type=str, default='results/',
                        help='path for saving results')
    parser.add_argument('--print_results', action='store_true')

    args = parser.parse_args()
    if not args.image_files and not args.image_dir:
        args.image_dir = 'resources/images/validate/'

    begin = datetime.now()
    print('Started inference at {}, with parameters:'.format(begin))
    for k, v in vars(args).items():
        print('[args] {}={}'.format(k, v))
    sys.stdout.flush()

    main(args=args)

    end = datetime.now()
    print('Inference ended at {}. Total time: {}.'.format(end, end - begin))
