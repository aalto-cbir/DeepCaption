#!/usr/bin/env python3

import argparse
from pandas.io.json import json_normalize  # package for flattening json in pandas df
import json
from pprint import pprint
import os
import sys


def main(args):
    json_file_name = os.path.basename(args.json_file)
    feather_file_name = os.path.splitext(json_file_name)[0] + '.feather'

    if args.output_dir is None:
        output_dir = '.'
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, feather_file_name)

    if os.path.exists(output_path):
        print("File {} already exists. Please (re)move it if you want to continue".format(
            output_path))
        sys.exit(1)

    with open(args.json_file) as f:
        print("Opening {}".format(args.json_file))
        data = json.load(f)

    if args.data_root:
        print("Setting data root to {}".format(args.data_root))
        data = data[args.data_root]

    print("Flattening JSON file and storing as data frame, using {} as record_path".
          format(args.record_path))

    df = json_normalize(data, args.record_path)

    print("First 5 rows of resulting data frame:")
    pprint(df[:5])
    print("Last 5 rows of resulting data frame:")
    pprint(df[-5:])

    print("Saving data to {}".format(output_path))

    df.to_feather(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert JSON file to flattened Pandas dataframe.'
                    'Output is stored in feather file format for fast access')
    parser.add_argument('json_file', type=str, default='',
                        help='Path to json file holding records')
    parser.add_argument('--record_path', type=str, default=None,
                        help="Record that will correspond to 1 row in pandas dataset")
    parser.add_argument('--data_root', type=str, default=None,
                        help='R.g. if we want root to be data[\'images\'] pass \'images\'')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output file to. Output file name '
                        ' is the same as JSON input, but with "*.feather" extension')
    args = parser.parse_args()

    main(args=args)
