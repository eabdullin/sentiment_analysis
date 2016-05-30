# -*- coding: utf-8 -*-
import pandas as pn
import numpy as np
import codecs
import re
import urllib2

def valid_and_separate(line):
    vals = line.split(';')
    if len(vals) != 2:
        return None
    for i in range(2):
        v = re.sub(r'[\.\,\%«"\)\(]', '', vals[i]).strip()
        if not v:
            return None
        vals[i] = v
    if vals[0] == vals[1]:
        return None
    return vals


def readvocub(filename):
    tr = {}
    with codecs.open(filename, 'r', encoding='utf8') as fin:
        for line in fin:
            vals = valid_and_separate(line)
            if vals is None:
                continue
            tr[vals[0]] = [vals[1]]
    fin.close()
    return tr

def request(words):
    data = {}
    data = words
    req = urllib2.Request()
