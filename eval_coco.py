#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Note: This script requires Python 2 to run!"""
import argparse
import json
import os
import re

from pycocotools.coco import COCO
from datasets.pycocoevalcap.eval import COCOEvalCap


def basename(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def main(args):
    coco = COCO(args.ground_truth)
    cocoRes = coco.loadRes(args.result_file)
    cocoEval = COCOEvalCap(coco, cocoRes)

    if args.subset:
        cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    cocoEval.evaluate()

    # print output evaluation scores
    print "="*20
    for metric, score in cocoEval.eval.items():
        print '%s: %.3f' % (metric, score)
    print "="*20

    # save evaluation results in JSON format
    output_file = None
    if not args.eval_results:
        output_file = basename(args.result_file) + '.eval'
    else:
        output_file = args.eval_results

    json.dump(cocoEval.eval, open(os.path.join(args.eval_path, output_file), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run COCO evaluation on the provided JSON file containing\
        captions")
    parser.add_argument('result_file', type=str,
                        help='Captions to evaluate in JSON format')
    parser.add_argument('--ground_truth', type=str,
                        default='datasets/data/COCO/annotations/captions_val2014.json',
                        help='Ground truth captions to evaluate against')
    parser.add_argument('--eval_results', type=str,
                        help='JSON output file for writing evaluation results')
    parser.add_argument('--eval_path', type=str, default='results/',
                        help='path for saving evaluation results')
    parser.add_argument('--subset', action='store_true',
                        help='Set this if we are evaluating on a subset only')
    args = parser.parse_args()

    main(args=args)
