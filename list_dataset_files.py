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

    # Check that we are not overwriting anything
    if os.path.exists(file_name):
        print('ERROR: {} exists, please remove it first if you really want to replace it.'.
              format(file_name))
        sys.exit(1)

    # Get dataset parameters and vocabulary wrapper:
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, _ = dataset_configs.get_params(args.dataset, vocab_path=None)

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

        with open(file_name, 'a') as f:
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
                             'defaults to "file_list-dataset_name.txt"')
    parser.add_argument('--environment', type=str,
                        help='Opetionally specify the environment where the paths are valid '
                        'by default value of $HOSTNAME environment variable will be used')
    parser.add_argument('--log_step', type=int, default=10,
                        help='How often do we want to log output')

    args = parser.parse_args()
    main(args=args)
