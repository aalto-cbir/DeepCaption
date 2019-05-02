#!/usr/bin/env python3

import argparse
import torch
from data_loader import ExternalFeature, DatasetParams


def get_feature_dims(state, args):
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params = dataset_configs.get_params(args.dataset)[0]

    features_paths = dataset_params.features_path

    ext_feature_sets = [state['features'].external, state['persist_features'].external]
    loaders_and_dims = [ExternalFeature.loaders(fs, features_paths) for fs in ext_feature_sets]

    loaders, dims = zip(*loaders_and_dims)
    return dims


def model_update(args):
    model = __import__('model.encoder_decoder')
    model.Features = model.encoder_decoder.Features

    state = torch.load(args.model_filename, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    if args.create_ext_features_dim:
        state['ext_features_dim'] = get_feature_dims(state, args=args)

    print('New model saved as', args.fixed_model_filename)
    torch.save(state, args.fixed_model_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename', help='Model to update saved paths to.')
    parser.add_argument('fixed_model_filename', help='Fixed model to be saved.')
    parser.add_argument('--create_ext_features_dim', action='store_true')
    parser.add_argument('--dataset', type=str, default='generic',
                        help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')

    model_update(parser.parse_args())
