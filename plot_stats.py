#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt


def plot_stats(label, stats, color, ax1, ax2, args):
    epochs = sorted([int(x) for x in stats.keys()])

    measures = list(stats[str(epochs[0])].keys())

    for measure in measures:
        vals = [stats[str(e)][measure] for e in epochs]
        ax = ax1
        if measure == 'validation_loss':
            linestyle = '--'
        elif measure == 'validation_cider':
            linestyle = ':'
            ax = ax2
            ax2.set_ylabel('CIDEr')
        else:
            linestyle = '-'

        mparts = measure.split('_', maxsplit=1)
        measure_name = mparts[1] if len(mparts) > 1 else measure
        if measure_name == 'cider' and args.smooth_cider is not None:
            w = args.smooth_cider
            ext = w//2
            k = np.ones(w)/w
            # duplicate first and last element for convolution to be "valid"
            vals_ext = np.concatenate((np.tile(vals[0], ext), vals, np.tile(vals[-1], ext)))
            valsc = np.convolve(k, vals_ext, mode='valid')
            # print(vals_ext)
            # print(valsc)
            assert len(valsc) == len(vals)
            vals = valsc
            print('Smoothed {} with moving average width {}.'.format(measure, w))

        ax.plot(epochs, vals, label='{}: {}'.format(label, measure), color=color,
                linestyle=linestyle, linewidth=3)


def main(args):
    colors = ['blue', 'red', 'cyan', 'yellow', 'gray', 'black']
    labels = args.labels.split(',') if args.labels else []

    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')

    first = True

    while True:
        for i, filename in enumerate(args.files):
            with open(filename, 'r') as fp:
                if i < len(labels):
                    label = labels[i]
                else:
                    label = str(i+1)
                    print('{}: {}'.format(label, filename))
                plot_stats(label, json.load(fp), colors[i % len(colors)], ax1, ax2, args)

        if first:
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
            first = False

        ax1.legend(loc='center left', bbox_to_anchor=(1.2, 0.8))
        ax2.legend(loc='center left', bbox_to_anchor=(1.2, 0.6))

        if not args.watch:
            plt.show()
        else:
            plt.draw()
            plt.pause(3)
            # print('x')
            ax1.clear()
            ax2.clear()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+',
                        help='JSON file(s) with training stats')
    parser.add_argument('--labels', type=str, help='Labels for plot')
    parser.add_argument('--smooth_cider', type=int, help='Moving average smoothing')
    parser.add_argument('--watch', action='store_true', help='Watches input files')
    args = parser.parse_args()

    main(args)
