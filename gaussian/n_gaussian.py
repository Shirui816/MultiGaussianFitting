#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: shirui <shirui816@gmail.com>

# This is a general N gaussian-mix function generator

import inspect as ins
import numpy as np
import warnings


def _n_gaussian(n):
    _signature = []  # no need for x: def func(x, *args)..
    for i in range(n):
        _signature.append(ins.Parameter('a%s' % (i + 1),
                                        kind=ins.Parameter.POSITIONAL_OR_KEYWORD))
        # perhaps x is positional only, others are positional or keyword
        _signature.append(ins.Parameter('s%s' % (i + 1),
                                        kind=ins.Parameter.POSITIONAL_OR_KEYWORD))
        _signature.append(ins.Parameter('m%s' % (i + 1),
                                        kind=ins.Parameter.POSITIONAL_OR_KEYWORD))
        # _signature.append(ins.Parameter('c', kind=ins.Parameter.POSITIONAL_OR_KEYWORD))
    _ns = ins.Signature(parameters=_signature)

    def func(x, *args):  # a * 1/(2\pi)^0.5/std * exp(-0.5*(x-mu)^2/std^2), a,s,m
        res = 0
        for j in range(n):
            c = j*3
            res += args[c] * 1 / args[c + 1] * np.exp(-0.5 * (x - args[c + 2]) ** 2 / args[c + 1] ** 2)
        res *= 1 / (2 * np.pi) ** 0.5
        # res += args[-1]
        return res

    func.__signature__ = _ns
    func.__doc__ = "The arg sequal is a, std, mu"
    return func


def n_gaussian(n, known=()):
    if not known:
        return _n_gaussian(n)
    if len(known) > n:
        raise(Exception("Error! Known components more than specified components!"))
    if len(known) == n:
        warnings.warn("Warning! Number of known components is equal to number of specified components!")
    _signature = []

    # In [4]: a = [[None, None, None]] * 3
    # In [5]: a
    # Out[5]: [[None, None, None], [None, None, None], [None, None, None]]
    # In [6]: a[2][1] = 2
    # In [7]: a
    # Out[7]: [[None, 2, None], [None, 2, None], [None, 2, None]]
    # Shit! Lazy evaluation
    _consts = np.empty((n, 3), dtype=np.object)  # Initialized with None
    for i in range(n):
        if i < len(known):
            for j, p in enumerate(known[i]):
                if p is None:  # solve p == 0 problem
                    _signature.append(ins.Parameter('%s%s' % ('asm'[j], i+1),
                                                    kind=ins.Parameter.POSITIONAL_OR_KEYWORD))
                else:
                    _consts[i, j] = p
        else:
            for j in range(3):
                _signature.append(ins.Parameter('%s%s' % ('asm'[j], i + 1),
                                                kind=ins.Parameter.POSITIONAL_OR_KEYWORD))
    _ns = ins.Signature(parameters=_signature)

    def func(x, *args):
        res = m = 0
        dummy = [0, 0, 0]
        for k in _consts:  # a, s, m as a series of gaussian parameters
            for l in range(3):
                dummy[l] = k[l] if not k[l] is None else args[m]
                m += 0 if not k[l] is None else 1
            res += dummy[0] * 1/dummy[1] * np.exp(-0.5 * (x-dummy[2])**2/dummy[1]**2)
        res *= 1/(2*np.pi)**0.5
        return res

    func.__signature__ = _ns
    return func
