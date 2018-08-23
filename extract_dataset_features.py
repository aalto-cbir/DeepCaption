import argparse
import os
import sys
import lmdb

import torch
from torchvision import transforms

from model import FeatureExtractor
from data_loader import get_loader, DatasetParams

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
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

    # To open an lmdb handle and prepare it for the right size
    # it needs to fit the total number of elements in the dataset
    # so we set a map_size to a largish value here:
    map_size = 1e10

    lmdb_path = None
    file_name = None

    if args.output_file:
        file_name = args.output_file
    else:
        file_name = '{}-{}.lmdb'.format(args.dataset, args.extractor)

    os.makedirs(args.output_dir, exist_ok=True)

    lmdb_path = os.path.join(args.output_dir, file_name)

    print("Preparing to store extracted features to {}...".format(lmdb_path))
    env = lmdb.open(lmdb_path, map_size=map_size)

    print("Starting to extract features from dataset {} using {}...".
          format(args.dataset, args.extractor))
    show_progress = sys.stderr.isatty()
    for i, (images, _, _,
            image_ids, _) in enumerate(tqdm(data_loader, disable=not show_progress)):
        images = images.to(device)
        features = extractor(images).data.cpu().numpy()

        # Write to LMDB object:
        with env.begin(write=True) as txn:
            for j, image_id in enumerate(image_ids):
                txn.put(str(image_id).encode('ascii'), features[j])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='coco:train2014',
                        help='dataset that defines images for which features are needed')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--image_size', type=int, default=224,
                        help='size for cropping images')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='features/',
                        help='directory for saving image features')
    parser.add_argument('--output_file', type=str, default='',
                        help='file for saving features, if no name specified it '
                             'defaults to "dataset_name-extractor.lmdb"')
    parser.add_argument('--extractor', type=str, default='resnet152',
                        help='name of the extractor, ex: alexnet, resnet152, densenet201')

    args = parser.parse_args()
    main(args=args)
