#!/usr/bin/env python3

import nltk
import pickle
import argparse
import sys
import os
from collections import Counter

import data_loader as dl
from vocabulary import Vocabulary


try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x


def main(args):
    dataset_configs = dl.DatasetParams(args.dataset_config_file)

    # Make sure that the user specifies vocab output path
    # as an argument:
    if args.vocab_output_path is None:
        print('ERROR: When creating a vocabulary file you should always specify it on a '
              'command line, e.g. --vocab_output_path=/some/dir/vocab.pkl')
        sys.exit(1)
    else:
        vocab_output_path = args.vocab_output_path

    # Check that we are not overwriting anything
    if os.path.exists(vocab_output_path):
        print('ERROR: {} exists, please remove it first if you really want to replace it.'.
              format(vocab_output_path))
        sys.exit(1)

    dataset_params, _ = dataset_configs.get_params(args.dataset)
    for i in dataset_params:
        i.config_dict['no_tokenize'] = args.no_tokenize
        i.config_dict['show_tokens'] = args.show_tokens

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
            if args.no_tokenize:
                words = caption.split()
            else:
                words = nltk.tokenize.word_tokenize(caption.lower())
            if args.show_tokens:
                joined = ' '.join(words)
                diff_same = "DIFF" if caption!=joined else "SAME"
                print(diff_same, caption, '=>', joined)                    
            counter.update(words)

    if args.show_stats:
        for k in counter.keys():
            print(k, counter[k])

    # If the word frequency is less than 'threshold', then the word is discarded
    words = [word for word, cnt in counter.items() if cnt >= args.threshold]

    if True:
        print(words)
    
    # Create a vocab wrapper and add some special tokens
    vocab = Vocabulary()
    vocab.add_special_tokens()

    # Add the words to the vocabulary
    for word in words:
        vocab.add_word(word)

    # Save it
    vocab.save(vocab_output_path)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary to '{}'".format(vocab_output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco:train2014',
                        help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str, default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--vocab_output_path', type=str, help='path for saving vocabulary')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--show_stats', action="store_true")
    parser.add_argument('--show_tokens', action="store_true")
    parser.add_argument('--no_tokenize', action="store_true")
    args = parser.parse_args()
    main(args)
