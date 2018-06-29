#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

base_dir = os.path.expanduser('~/src/coco-caption')
default_groundtruth = base_dir + '/annotations/captions_val2014.json'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', type=str,
                        help='Captions to evaluate in JSON format')
    parser.add_argument('--ground_truth', type=str, default=default_groundtruth,
                        help='Ground truth captions to evaluate against')
    parser.add_argument('--eval_results', type=str,
                        help='JSON output file for writing evaluation results')
    parser.add_argument('--subset', action='store_true',
                        help='Set this if we are evaluating on a subset only')
    args = parser.parse_args()

    coco = COCO(args.ground_truth)
    cocoRes = coco.loadRes(args.result_file)
    cocoEval = COCOEvalCap(coco, cocoRes)

    if args.subset:
        cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    cocoEval.evaluate()

    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        print '%s: %.3f'%(metric, score)

    # save evaluation results in JSON format
    if args.eval_results:
        json.dump(cocoEval.eval, open(args.eval_results, 'w'))

