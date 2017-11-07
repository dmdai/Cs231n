#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__coauthor__ = 'Daylock'


import pickle, os, random
import numpy as np


def get_CIFAR_batch(filename):

    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='iso-8859-1')
        # datadict = pickle.load(f)
        X = datadict['data']
        y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float64)
        y = np.array(y)

        return X, y


def get_CIFAR10(ROOT):

    f = os.path.join(ROOT, 'data_batch_%d' % random.randint(1, 5))
    Xtr, ytr = get_CIFAR_batch(f)
    Xte, yte = get_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    return Xtr, ytr, Xte, yte
