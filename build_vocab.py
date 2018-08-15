#!/usr/bin/env python3

import nltk
import pickle
import argparse
import sys
import os
from collections import Counter

import data_loader as dl


try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def main(args):
    dataset_params = dl.DatasetParams.fromargs(args).configs

    # Take vocab path from dataset config, can be overridden by
    # command line argument
    if args.vocab_path is None:
        if len(dataset_params) == 1:
            vocab_path = dataset_params[0].vocab_path
        else:
            print('ERROR: for combined datasets you need to define the vocabulary on the '
                  'command line, e.g. --vocab_path=/some/dir/vocab.pkl')
            sys.exit(1)
    else:
        vocab_path = args.vocab_path

    # Check that we are not overwriting anything
    if os.path.exists(vocab_path):
        print('ERROR: {} exists, please remove it first if you really want to replace it.'.
              format(vocab_path))
        sys.exit(1)

    # Get data loader
    data_loader, _ = dl.get_loader(dataset_params, None, None, 128, shuffle=False,
                                   num_workers=args.num_workers, skip_images=True)

    if len(data_loader) == 0:
        print('ERROR: No captions found, please specify a dataset that has captions defined.')
        if args.dataset == 'coco':
            print('HINT: instead of "coco" use "coco:train2014"')
        sys.exit(1)

    # Start counting words...
    counter = Counter()
    show_progress = sys.stderr.isatty()
    for _, captions, _, _, _ in tqdm(data_loader, disable=not show_progress):
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded
    words = [word for word, cnt in counter.items() if cnt >= args.threshold]

    # Create a vocab wrapper and add some special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary
    for word in words:
        vocab.add_word(word)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco:train2014',
                        help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str, default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--vocab_path', type=str, help='path for saving vocabulary')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    main(args)
