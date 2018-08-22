# Use datasets.conf
# Use data loader with image-specific stuff
# Use batches
# Use shuffle off
# Support COCO and VG from the get go
# Import feature extractor from the models
import argparse
import os
import sys
import lmdb

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from model import ModelParams, EncoderCNN, FeatureExtractor
from data_loader import get_loader, DatasetParams

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def put_feature(mdb, idx):
    x = _lmdb_to_numpy(mdb.put(str(idx).encode('ascii')))

    return torch.tensor(x).float()


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    # Get dataset parameters and vocabulary wrapper:
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, _ = dataset_configs.get_params(args.dataset, vocab_path=None)

    # We ask it to iterate over images instead of all (image, caption) pairs
    data_loader, _ = get_loader(dataset_params, vocab=None, transform=transform,
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                ext_feature_sets=None,
                                skip_images=False,
                                iter_over_images=True)

    extractor = FeatureExtractor(args.extractor, True).to(device)

    # Open and lmdb handle and prepare it for the right size - total number of elements 
    # in the dataset

    map_size = X.nbytes * 10

    # Replace 'mylmdb' with path to file
    env = lmdb.open('mylmdb', map_size=map_size)
 
    show_progress = sys.stderr.isatty() and not args.verbose
    for i, (images, _, _,
            image_ids, _) in enumerate(tqdm(data_loader, disable=not show_progress)):
        images = images.to(device)
        features = extractor(images).data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='coco:train2014+coco:val2014',
                        help='dataset that defines images for which features are needed')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='features/',
                        help='directory for saving image features')
    parser.add_argument('--output_file', type=str, default='',
                        help='file for saving features, if no name specified it '
                             'defaults to "dataset_name-extractor.lmdb"')
    parser.add_argument('--extractor', type=str, default='resnet152',
                        help='name of the extractor, ex: alexnet, resnet152, densenet201')
    parser.add_argument('--verbose', help='verbose output', action='store_true')

    args = parser.parse_args()
    main(args=args)
