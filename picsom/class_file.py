#!/usr/bin/env python2

from __future__ import print_function

import sys
import os

class picsom_class:
    def __init__(self, fn):
        self._path    = None
        self._ismeta  = False
        self._clslist = []
        self._objects = set()
        ok = True
        if os.path.isfile(fn):
            try:
                with open(fn, 'r') as f:
                    self._path = fn
                    entries = []
                    first = True
                    for l in f:
                        l = l.rstrip()
                        if first:
                            first = False
                            if l[0:15]=='# METACLASSFILE':
                                self._ismeta = True
                                continue
                        if l[0]=='#':
                            continue
                        entries.append(l)
                    if self._ismeta:
                        self._clslist = entries
                    else:
                        self._objects = set(entries)
            except IOError as e:
                print('Could not open file <{}>'.format(fn))

    def path(self):
        return self._path

    def ismeta(self):
        return self._ismeta

    def classnames(self):
        return self._clslist

    def objects(self):
        return self._objects

if __name__ == "__main__":
    if len(sys.argv)!=2 :
        print('{} expected one argument: classfile'.format(sys.argv[0]))
        exit(1)
    cls = picsom_class(sys.argv[1])
    print('path   =', cls.path())
    print('ismeta =', cls.ismeta())
    if cls.ismeta():
        print('n_classes =', len(cls.classnames()), ':', cls.classnames())
    else:
        print('n_objects =', len(cls.objects()), ':', cls.objects())
