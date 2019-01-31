#!/usr/bin/env python3

import argparse
import torch

from collections import OrderedDict
import numbers

from vocabulary import Vocabulary
from model.encoder_decoder import Features

verbose = False


def dump_dict(d, prefix=''):
    global verbose

    for key, value in d.items():
        print(prefix + str(key), end=': ')
        if value is None or isinstance(value, (Vocabulary, numbers.Number, list)):
            print(value)
        elif isinstance(value, (dict, OrderedDict, Features)):
            if isinstance(value, Features):
                value = value._asdict()
            print()
            dump_dict(value, prefix + '  ')
        elif isinstance(value, torch.Tensor):
            print('Tensor: shape: {}, type: {}, device: {}'.format(
                'x'.join([str(x) for x in value.shape]), value.dtype, value.device))
            if verbose:
                print(prefix + '  ', value)
            # else:
            #     print(prefix + '  ', 'First values: {}'.format(
            #         ', '.join(['{:.5f}'.format(x) for x in value.view(-1)[:5]]) + ', ...' ))
        else:
            print(type(value))


def model_info(filename):
    state = torch.load(filename, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    dump_dict(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    verbose = args.verbose
    model_info(args.model_filename)
