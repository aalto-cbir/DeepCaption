import argparse
import os
import sys
import lmdb
import h5py
import numpy as np

# Must assert that number of images in h5 files are the same as number of files in file list
# File names will be generated manually based on the basenames in order to avoid wrong matches
# Create an array of tuples (file_list, h5_file), make sure that all files exist, make sure
# that the number of entries in both files are the same

# Image list basename:
# "file_lists/image_file_list-coco:train2014:no_resize+coco:val2014:no_resize-taito-gpu.csc.fi"

# Feature file basename:
# "features/densecap_features-coco:train2014:no_resize+coco:val2014:no_resize"


def make_file_names(inputs_list_basename, features_basename, num_files):
    if num_files == 1:
        image_list_files = ['{}.txt'.format(inputs_list_basename)]
        features_files = ['{}.h5'.format(features_basename)]
    elif num_files > 1:
        image_list_files = []
        features_files = []
        for i in range(num_files):
            image_list_file = '{}_{}_of_{}.txt'.format(
                inputs_list_basename, i + 1, num_files)
            features_file = '{}_{}_of_{}.h5'.format(
                features_basename, i + 1, num_files)

            image_list_files.append(image_list_file)
            features_files.append(features_file)
    else:
        print("ERROR: Invalid number of files specified: {}".format(num_files))
        sys.exit(1)

    return image_list_files, features_files


def make_output_file_name(output_file, output_path, features_basename):
    if output_file is None:
        output_file = os.path.basename(features_basename) + '.lmdb'

    if not output_file.lower().endswith('.lmdb'):
        print("ERROR: Output file name should end with 'lmdb'")
        sys.exit(1)

    full_file_path = os.path.join(output_path, output_file)

    return full_file_path


def main(args):
    print("Preparing to convert features from H5 to LMDB format. "
          "Features spread across {} files".format(args.num_files))
    img_lists, feature_files = make_file_names(args.inputs_list_basename,
                                               args.features_basename, args.num_files)

    lmdb_path = make_output_file_name(args.output_file, args.output_path,
                                      args.features_basename)

    # Open LMDB file for writing. Make sure it doesn't exist from before
    if os.path.exists(lmdb_path):
        print("ERROR: File with name {} already exists."
              "Use different file name or rename/remove "
              "the existing file.".format(lmdb_path))
        sys.exit(1)

    n_images = 0

    for file_name in img_lists:
        with open(file_name, 'r') as f:
            n_images += len(f.readlines())

    print("Handling features for {} images".format(n_images))

    # Map size setting for lmdb:
    map_size = 1e12

    for img_list, feature_file in zip(img_lists, feature_files):

        # Read in the file list and get image names:
        with open(img_list, "r") as fl:
            img_files = fl.readlines()
            img_names = [os.path.basename(s).rstrip() for s in img_files]

        # Read in the feature from HDF5 file and average them across regions:
        with h5py.File(feature_file, "r") as ff:
            print("Processing {}...".format(feature_file))

            if ff.get('feats') is None:
                print("ERROR: Feature file does not contain dataset 'feats'")
                sys.exit(1)

            V = np.mean(ff['feats'], axis=1)

        # Compare the lengths:
        assert len(img_names) == len(V), 'Image file list and feature '
        'matrix have different lengths!'

        # Write to LMDB object:
        with lmdb.open(lmdb_path, map_size=map_size) as env:
            with env.begin(write=True) as txn:
                for i, img_name in enumerate(img_names):
                    txn.put(str(img_name).encode('ascii'), V[i])

    print("DONE!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_list_basename', type=str,
                        help="First part of the file name contaning a list of image paths")
    parser.add_argument('--features_basename', type=str,
                        help="First part of the h5 filename containing image features")
    parser.add_argument('--num_files', type=int, default=1,
                        help='Number of image list files and h5 features files (must match)')
    parser.add_argument('--output_file', type=str,
                        help='File name for LMDB database output')
    parser.add_argument('--output_path', type=str, default='features',
                        help='Path where to save LMDB features')

    args = parser.parse_args()
    main(args=args)
