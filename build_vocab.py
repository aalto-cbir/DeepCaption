#!/usr/bin/env python3
import argparse
from datetime import datetime
import sys

import data_loader as dl
from vocabulary import build_vocab, get_vocab
from data_loader import DatasetParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str, default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--vocab_output_path', type=str, help='path for saving vocabulary')
    parser.add_argument('--vocab_threshold', type=int, default=4,
                        help='minimum word count threshold')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--show_vocab_stats', action="store_true",
                        help='show key counts')
    parser.add_argument('--show_tokens', action="store_true")
    parser.add_argument('--no_tokenize', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--dont_create_vocab', action="store_true",
                        help='Dont create the vocabulary file.')
    parser.add_argument('--create_vocab_stats_file', action="store_true",
                        help='Create {vocab_name}-vocab_stats.txt file with key counts.')
    parser.add_argument('--create_leftovers_file', action="store_true",
                        help='Create {vocab_name}-leftovers.txt file with the words and counts that '
                             'were under the threshold.')
    parser.add_argument('--leftovers_from_vocabulary', type=str, default=None,
                        help='Create leftovers file relative to specified vocabulary. No other file is created.')

    return parser.parse_args()


def check_dataset(args):
    if args.dataset is None:
        print('ERROR: No dataset selected!')
        print('Please supply a training dataset with the argument --dataset DATASET')
        print('The following datasets are configured in {}:'.format(args.dataset_config_file))

        dataset_configs = DatasetParams(args.dataset_config_file)
        for ds, _ in dataset_configs.config.items():
            if ds not in ('DEFAULT', 'generic'):
                print(' ', ds)

        sys.exit(1)

    return args.dataset


def main():
    args = parse_args()

    if args.leftovers_from_vocabulary is not None:
        args.vocab = args.leftovers_from_vocabulary
        vocab = get_vocab(args)
        if hasattr(vocab, 'metadata'):
            args.dataset = vocab.metadata['dataset']
            args.vocab_threshold = vocab.metadata['vocab_threshold']
        else:
            print('No metadata found in the vocabulary, so no dataset could be retrieved. This should be an old vocabulary.')
            print('Falling back to --dataset and --vocab_threshold parameters.')
            print()

        args.vocab_output_path = args.leftovers_from_vocabulary.split('.')[0]
        args.dont_create_vocab = True
        args.create_leftovers_file = True
        args.create_vocab_stats_file = False

    check_dataset(args)

    assert args.show_vocab_stats is False, "Deprecated flag, please use --create_vocab_stats_file."

    assert args.vocab_output_path is not None or args.dont_create_vocab, \
        'When creating a vocabulary file you should always specify it on a command line, ' \
        'e.g. --vocab_output_path=/some/dir/vocab.pkl'

    dataset_configs = dl.DatasetParams(args.dataset_config_file)
    dataset_params = dataset_configs.get_params(args.dataset)
    for i in dataset_params:
        i.config_dict['no_tokenize'] = args.no_tokenize
        i.config_dict['show_tokens'] = args.show_tokens

    build_vocab(args.vocab_output_path, dataset_params, args)


if __name__ == '__main__':
    begin = datetime.now()
    print('Started to build vocab at {}.'.format(begin))

    main()

    end = datetime.now()
    print('Finished building vocab at {}. Total time: {}.'.format(end, end - begin))
