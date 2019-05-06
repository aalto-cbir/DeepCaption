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


def model_info(filename, vocab_filename=None):
    try:
        state = torch.load(filename, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    except AttributeError:
        print('WARNING: Old model found. Trying an import trick to load Features properly. '
              'Please use model_update.py to fix the model.')
        model = __import__('model.encoder_decoder')
        model.Features = model.encoder_decoder.Features
        # one can also create a file __init__.py in model/ with `from model.encoder_decoder import Features`
        # and do `import model`

        state = torch.load(filename, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    dump_dict(state)

    if vocab_filename is not None:
        state['vocab'].save(vocab_filename)

        print()
        print('Model vocabulary saved as', vocab_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dump_vocabulary', type=str, default=None,
                        help='File name where to save the vocabulary used in the model, as pickle.')

    args = parser.parse_args()

    verbose = args.verbose
    model_info(filename=args.model_filename, vocab_filename=args.dump_vocabulary)
