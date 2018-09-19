import pandas as pd
import argparse
import json
import os
import sys


def main(args):
    output_data = []

    if args.output_file is None:
        print("No output file specified, exiting.")
        sys.exit(1)

    for file in args.file_list:
        print("Reading in {}".format(file))
        with open(file) as raw_data:
            results = json.load(raw_data)
        filename = os.path.basename(file)
        data_row = {}
        data_row.update({'name': '.'.join(filename.split('.')[:-1])})
        data_row.update(results)

        output_data.append(data_row)

    columns = output_data[0].keys()

    df = pd.DataFrame(output_data, columns=columns)

    if args.verbose:
        print(df)

    output_dir = args.output_dir

    output_file = os.path.join(output_dir, args.output_file)

    print("Writing combined results to {}".format(output_file))
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine evaluation results into a single CSV file")
    parser.add_argument('file_list', type=str, nargs='*',
                        help='list of files to be evaluated')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory where to output the CSV file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Name of output CSV file')
    parser.add_argument('--verbose', action='store_true',
                        help='Output parsed results to stdout')
    args = parser.parse_args()

    main(args=args)
