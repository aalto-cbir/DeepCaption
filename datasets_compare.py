#!/usr/bin/env python3

import argparse
import os
import sys
from itertools import chain

from data_loader import get_loader, DatasetParams

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x


def main(args):
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params_1 = dataset_configs.get_params(args.dataset_1)
    dataset_params_2 = dataset_configs.get_params(args.dataset_2)

    for p in (dataset_params_1, dataset_params_2):
        for d in p:
            # Tell dataset to output id in integer or other simple format:
            d.config_dict['return_simple_image_id'] = True

    data_loader_1, _ = get_loader(dataset_params_1, vocab=None, transform=None,
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  ext_feature_sets=None,
                                  skip_images=True,
                                  iter_over_images=True)

    data_loader_2, _ = get_loader(dataset_params_2, vocab=None, transform=None,
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  ext_feature_sets=None,
                                  skip_images=True,
                                  iter_over_images=True)

    show_progress = sys.stderr.isatty()
    print("Reading image ids from dataset {}".format(args.dataset_1))
    ids_1 = [img_ids for _, _, _, img_ids, _ in tqdm(data_loader_1, disable=not show_progress)]
    set_1 = set(chain(*ids_1))

    print("Reading image ids from dataset {}".format(args.dataset_2))
    ids_2 = [img_ids for _, _, _, img_ids, _ in tqdm(data_loader_2, disable=not show_progress)]
    set_2 = set(chain(*ids_2))

    intersection = set_1.intersection(set_2)
    len_intersect = len(intersection)

    print("There are {} images shared between {} and {}".format(len_intersect,
                                                                args.dataset_1,
                                                                args.dataset_2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_1', type=str,
                        help='First dataset to compare')
    parser.add_argument('--dataset_2', type=str,
                        help='Second dataset to compare')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    main(args=args)