#!/usr/bin/env python3

import json

import matplotlib.pyplot as plt


def plot_stats(label, stats, color):
    epochs = sorted([int(x) for x in stats.keys()])

    measures = list(stats[str(epochs[0])].keys())

    for measure in measures:
        vals = [stats[str(e)][measure] for e in epochs]
        if measure == 'validation_loss':
            linestyle = '--'
        else:
            linestyle = '-'
        plt.plot(epochs, vals, label='{}: {}'.format(label, measure), color=color,
                 linestyle=linestyle, linewidth=3)


def main(args):
    colors = ['blue', 'red', 'cyan', 'yellow', 'gray', 'black']
    labels = args.labels.split(',') if args.labels else []

    for i, filename in enumerate(args.files):
        with open(filename, 'r') as fp:
            if i < len(labels):
                label = labels[i]
            else:
                label = str(i+1)
                print('{}: {}'.format(label, filename))
            plot_stats(label, json.load(fp), colors[i % len(colors)])

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+',
                        help='JSON file(s) with training stats')
    parser.add_argument('--labels', type=str, help='Labels for plot')
    args = parser.parse_args()

    main(args)
