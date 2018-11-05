#!/usr/bin/env python3
import argparse
from datetime import datetime
import sys

import data_loader as dl
from vocabulary import build_vocab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco:train2014',
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

    return parser.parse_args()


def main():
    args = parse_args()
    # Make sure that the user specifies vocab output path
    # as an argument:
    if args.vocab_output_path is None:
        print('ERROR: When creating a vocabulary file you should always specify it on a '
              'command line, e.g. --vocab_output_path=/some/dir/vocab.pkl')
        sys.exit(1)
    else:
        vocab_output_path = args.vocab_output_path

    dataset_configs = dl.DatasetParams(args.dataset_config_file)
    dataset_params = dataset_configs.get_params(args.dataset)
    for i in dataset_params:
        i.config_dict['no_tokenize'] = args.no_tokenize
        i.config_dict['show_tokens'] = args.show_tokens

    build_vocab(vocab_output_path, dataset_params, args)


if __name__ == '__main__':
    begin = datetime.now()
    print('Started to build vocab at {}.'.format(begin))

    main()

    end = datetime.now()
    print('Finished building vocab at {}. Total time: {}.'.format(end, end - begin))
