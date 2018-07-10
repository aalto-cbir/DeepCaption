import re
import pandas as pd
import argparse
import json
import glob
import os

# Example filename format:
# model-es256-hs512-nl2-bs128-lr0.001-da0.0-ep5.json
params_mapping = {'es': 'embed_size',
                  'hs': 'hidden_size',
                  'nl': 'num_layers',
                  'ep': 'epoch',
                  'bs': 'batch_size',
                  'lr': 'learning_rate',
                  'da': 'dropout'}       


def params_from_file_name(filename):
    """Exctract parameters from filename"""
    params = {}

    for key in params_mapping:
        # Extract parameter values:
        m = re.search("-" + key + "([0-9]*\.*[0-9]+)", filename)
        value = m.group(1)
        params[params_mapping[key]] = value

    return params


def main(args):
    file_list = glob.glob(args.evaluations_dir + '/*.eval')

    output_data = []

    for file in file_list:
        print("Reading in {}".format(file))
        with open(file) as raw_data:
            results = json.load(raw_data)
        filename = os.path.basename(file)
        params = params_from_file_name(filename)
        data_row = {}
        data_row.update({'name': filename.split('.')[0]})
        data_row.update(params)
        data_row.update(results)

        output_data.append(data_row)

    columns = output_data[0].keys()

    df = pd.DataFrame(output_data, columns=columns)

    if args.verbose:
        print(df)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.evaluations_dir

    output_file = os.path.join(output_dir, args.output_file)

    print("Writing combined results to {}".format(output_file))
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine evaluation results into a single CSV file")
    parser.add_argument('--evaluations_dir', type=str, default='results/',
                        help='Directory containing evaluation files')
    parser.add_argument('--output_dir', type=str,
                        help='Directory where to output the CSV file,\
                        same as evaluations_dir by default')
    parser.add_argument('--output_file', type=str, default='eval_results.csv',
                        help='Name of output CSV file')
    parser.add_argument('--verbose', action='store_true',
                        help='Output parsed results to stdout')
    args = parser.parse_args()

    main(args=args)
