import json
import argparse

# Helper utility for generating a new-line separated class file from json array
# Used for creating class files corresponding to Visual Genome image2paragraph datasets


def main(args):
    with open(args.json_file) as jf:
        data = json.load(jf)

    with open(args.class_file, 'w') as cf:
        cf.writelines([str(id) + '\n' for id in data])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                        help='path to file containing json array of image ids')
    parser.add_argument('--class_file', type=str,
                        help='output path to "class" file containing image ids \
                        separated by new lines')
    args = parser.parse_args()
    main(args=args)
