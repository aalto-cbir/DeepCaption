#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt


def plot_stats(stats):
    epochs = sorted([int(x) for x in stats.keys()])

    measures = list(stats[str(epochs[0])].keys())

    for measure in measures:
        vals = [stats[str(e)][measure] for e in epochs]
        plt.plot(epochs, vals, label=measure)

    plt.legend()
    plt.show()


def main(args):
    with open(args.stats_file, 'r') as fp:
        plot_stats(json.load(fp))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('stats_file', type=str, help='JSON file with training stats')
    args = parser.parse_args()

    main(args)
