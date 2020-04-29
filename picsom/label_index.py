#!/usr/bin/env python2

from __future__ import print_function

import sys
import os
import re

class picsom_label_index:
    def __init__(self, path):
        #print("<"+path+">")
        self._lab2idx     = dict()
        self._idx2lab     = []
        self._path        = ''
        self._labellength = 0
        self._extra_f     = dict()
        self._extra_b     = dict()

        lab2idx = dict()
        idx2lab = []
        length = 0
        
        ok = True
        if path!='':
            try:
                with open(path) as f:
                    #print("opened")
                    lno = 0
                    for l in f:
                        lno += 1
                        #print(l)
                        a = l.split()
                        #print(a)
                        if a[0][0]!='#':
                            print('Expected # in "{}" of line {} in {}'.\
                                  format(a[0], lno, path))
                            ok = False
                            break
                        idx = int(a[0][1:])
                        if a[1][0]!='<' or a[1][-1]!='>':
                            print('Expected <...> in "{}" of line {} in {}'.\
                                  format(a[1], lno, path))
                            ok = False
                            break
                        lab = a[1][1:-1]
                        if length==0:
                            length = len(lab)
                        #print(idx, lab)
                        idx2lab.append(lab)
                        lab2idx[lab] = idx

                        for e in a[2:]:
                            #print(e)
                            m = re.match('(.*)=\[(.*)\]', e)
                            k, v = m.group(1), m.group(2)
                            #print(k, v)
                            if not k in self._extra_f:
                                self._extra_f[k] = dict()
                                self._extra_b[k] = dict()
                            self._extra_f[k][idx] = v
                            self._extra_b[k][v]   = idx

            except IOError as e:
                print('Could not open file <{}>'.format(f))
                ok = False

        if ok:
            self._path        = path
            self._labellength = length
            self._lab2idx     = lab2idx
            self._idx2lab     = idx2lab
            
    def path(self):
        return self._path

    def labellength(self):
        return self._labellength

    def nobjects(self):
        return len(self._idx2lab)

    def label_by_index(self, idx):
        return self._idx2lab[idx]

    def index_by_label(self, lab):
        return self._lab2idx[lab]

    def extra_by_index(self, k, idx):
        return self._extra_f[k][idx]

    def index_by_extra(self, k, v):
        return self._extra_b[k][v]

    def extras(self):
        return self._extra_f.keys()

if __name__ == "__main__":
    if len(sys.argv)!=2 :
        print('%s expected one argument: labels.txt'.format(sys.argv[0]))
        exit(1)
    li = picsom_label_index(sys.argv[1])
    print('path={} nobjects={:d} labellength={:d} extras={}'.format(
        li.path(), li.nobjects(), li.labellength(), ','.join(li.extras())))

    m = 10
    if m>li.nobjects():
        m = li.nobjects()

    for idx in range(m):
        lab = li.label_by_index(idx)
        iii = li.index_by_label(lab)
        es = []
        for e in li.extras():
            ee = li.extra_by_index(e, idx)
            es += [ e+'='+ee+'=>#'+str(li.index_by_extra(e, ee)) ]
        print('label of #{:d} is {} => #{:d} extras: {}'.
              format(idx, lab, iii, ' '.join(es)))

