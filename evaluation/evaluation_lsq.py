#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import numpy as np


class Evaluation(object):
    def __init__(self, model):
        self.model = model

    def _log_prob(self, samples, i):
        return np.log(self.model.funcs[i](samples, *self.model.popt[i]) *
                      self.model.normal_factor[i])  # This func is already a sum of functions

    def aic(self, x, i):
        return 2 * self.model.n_parameters(i) - 2 * self.score(x, i) * x.shape[0]

    def bic(self, x, i):
        return 2 * self.model.n_parameters(i) * np.log(x.shape[0]) - 2 * self.score(x, i) * x.shape[0]

    def score(self, x, i):
        return self._log_prob(x, i).mean()
