#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import numpy as np
from fitting import GaussianFittingN
from gaussian import n_gaussian
from evaluation import aic_bic_score, estimating
from scipy.integrate import simps


def gaussianMix(x, weights, std_vs, means):
    return np.sum(np.array([n_gaussian(1)(x, w, std_v, mu, 0)
                            for w, std_v, mu in zip(weights, std_vs, means)]), axis=0)


x = np.linspace(-10, 10, 1000)
y = gaussianMix(x, np.array([0.5, 0.5]), np.array([1, 2]), np.array([0, 1]))
y /= simps(y, x)  # Generate data

n_component = 6
a = GaussianFittingN(bounds=((0,1),(0,10), (0,10)))
a.fit(x, y, n_component)  # max to 5 gaussians with a known component. popt_res gives the signatures.

models = estimating(a)  # Currently known is not supported
aic, bic, likelyhood = aic_bic_score(models, 10, x, y, 20000)

from pylab import *
plot(np.arange(n_component) + 1, aic/20000, label='aic', ls='--')
plot(np.arange(n_component) + 1, bic/20000, label='bic', ls='--')
legend()
twinx()
plot(np.arange(n_component) + 1, likelyhood, label='likelyhood')
legend()
show()