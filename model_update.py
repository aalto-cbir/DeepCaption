#!/usr/bin/env python3

import argparse
import torch


def model_update(filenames):
    model = __import__('model.encoder_decoder')
    model.Features = model.encoder_decoder.Features

    for filename in filenames:
        state = torch.load(filename, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        name, _, extension = filename.rpartition('.')
        fixed_name = name + '_fixed.' + extension
        print('New model saved as', fixed_name)
        torch.save(state, fixed_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename', nargs='+', help='Model or list of models to update saved paths to.')
    args = parser.parse_args()

    model_update(args.model_filename)
