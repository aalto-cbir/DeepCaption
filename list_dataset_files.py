#!/usr/bin/env python3

import argparse
import os
import sys

from data_loader import get_loader, DatasetParams

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x


def main(args):

    if args.output_file:
        file_name = args.output_file
    else:
        if args.environment is not None:
            environment = args.environment
        else:
            environment = os.getenv('HOSTNAME')
            if environment is None:
                environment = 'unknown_host'
        file_name = 'image_file_list-{}-{}.txt'.format(args.dataset, environment)

    file_name = os.path.join(args.output_path, file_name)

    os.makedirs(args.output_path, exist_ok=True)

    # If we want to generate multiple files we need to add "_X_of_Y" string to the file
    # to indicate which file out of the set it is:
    if args.num_files > 1:
        file_name_list = []
        for i in range(args.num_files):
            file_name_i = os.path.splitext(file_name)[0] + '_{}_of_{}.txt'.format(
                i + 1, args.num_files)
            file_name_list.append(file_name_i)
    else:
        file_name_list = None

    # Check that we are not overwriting anything
    if os.path.exists(file_name):
        print('ERROR: {} exists, please remove it first if you really want to replace it.'.
              format(file_name))
        sys.exit(1)

    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params = dataset_configs.get_params(args.dataset)

    for d in dataset_params:
        # Tell dataset to output full image paths instead of image id:
        d.config_dict['return_full_image_path'] = True

    # We ask it to iterate over images instead of all (image, caption) pairs
    data_loader, _ = get_loader(dataset_params, vocab=None, transform=None,
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                ext_feature_sets=None,
                                skip_images=True,
                                iter_over_images=True)

    print("Getting file paths from dataset {}...".format(args.dataset))
    show_progress = sys.stderr.isatty()
    for i, (_, _, _,
            paths, _) in enumerate(tqdm(data_loader, disable=not show_progress)):

        if args.num_files == 1:
            _file_name = file_name
        else:
            n = int(i * data_loader.batch_size * args.num_files / len(data_loader.dataset))
            _file_name = file_name_list[n]

        with open(_file_name, 'a') as f:
            for path in paths:
                f.write(path + '\n')

        # Print log info
        if not show_progress and ((i + 1) % args.log_step == 0):
            print('Batch [{}/{}]'.format(i + 1, len(data_loader)))
            sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='coco:train2014',
                        help='dataset that defines images for which features are needed')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_file', type=str, default='',
                        help='file for saving file_list, if no name specified it '
                             'defaults to "file_list-dataset_name_X_of_Y.txt"')
    parser.add_argument('--environment', type=str,
                        help='Opetionally specify the environment where the paths are valid '
                        'by default value of $HOSTNAME environment variable will be used')
    parser.add_argument('--log_step', type=int, default=10,
                        help='How often do we want to log output')
    parser.add_argument('--num_files', type=int, default=1,
                        help='How many output files should be generated')
    parser.add_argument('--output_path', type=str, default='file_lists',
                        help='Path where to save generated file lists')

    args = parser.parse_args()
    main(args=args)
