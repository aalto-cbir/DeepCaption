#!/usr/bin/env python3

import argparse


def read_results_file(file):
    with open(file) as f:
        lines = f.read()

    file_results = {}
    for line in lines.splitlines():
        metric = line.split(':')
        file_results[metric[0].strip()] = float(metric[1].strip())

    return file_results


def update_better_scores(scores, file_results, model, epoch, group):
    metrics = scores.setdefault(model, {}).setdefault(group, {})

    for metric, score in file_results.items():
        if metric in metrics:
            if metrics[metric]['score'] < score:
                metrics[metric]['score'] = score
                metrics[metric]['epoch'] = epoch
        else:
            metrics[metric] = {'score': score, 'epoch': epoch}


def print_scores(scores):
    for model, groups in sorted(scores.items()):
        print(model)
        for group, metrics in sorted(groups.items()):
            print(' ', group)
            for metric, stats in sorted(metrics.items()):
                print('   ', metric)
                for k, v in sorted(stats.items()):
                    print('     ', k, ':', v)


def metric_strs(g, metrics_strs, separator):
    stringo = ''

    for m in metrics_strs:
        stringo += (str(g[m]['score']) if m != '<space>' else '') + separator

    for m in metrics_strs:
        if m != '<space>':
            stringo += g[m]['epoch'].split('ep')[-1] + ','

    return stringo[:-1]


def print_scores_gdoc(scores, separator):
    header = 'model METEOR	CIDER	BLEU	Best epoch each	METEOR	CIDER	BLEU	Best epoch each	METEOR	METEOR*	CIDER	CIDER*'

    print(header)
    for model, groups in sorted(scores.items()):
        model_str = model \
                    + metric_strs(groups['2016'], ['METEOR', 'CIDER', 'BLEU-4'], separator) + separator \
                    + metric_strs(groups['2017-G2'], ['METEOR', 'CIDER', 'BLEU-4'], separator) + separator \
                    + metric_strs(groups['2018'], ['METEOR', '<space>', 'CIDER', '<space>'], separator)
        print(model_str)


def main(args):
    scores = {}

    for file in args.result_files:
        if not file.endswith('.result'):
            continue

        model_name, _, b = file.rpartition('/')[2].rpartition('.result')[0].rpartition('_')
        epoch, _, group = b.partition('-')

        file_results = read_results_file(file)

        update_better_scores(scores, file_results, model_name, epoch, group)

    if args.gdocs:
        print_scores_gdoc(scores, args.gdocs_separator)
    else:
        print_scores(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_files', nargs='+',
                        help='JSON result files')
    parser.add_argument('--gdocs', action='store_true',
                        help='Output in Google Docs experiments format')
    parser.add_argument('--gdocs_separator', type=str, default=';',
                        help='row separator for Google Docs experiments format')
    args = parser.parse_args()

    main(args)
