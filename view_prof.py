#!/usr/bin/env python3

import argparse
from pstats import Stats

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, nargs='?', default='train.prof')
args = parser.parse_args()


s = Stats(args.filename)

s.sort_stats('time')
s.print_stats(0.05)

s.print_callers(0.01)
