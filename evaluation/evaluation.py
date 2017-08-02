#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky as cpc
from sklearn.mixture.gaussian_mixture import GaussianMixture
import numpy as np
import fitting


def estimating(a, covariance_type='diag', **kwargs):  # currently only work with no known variables
    assert(isinstance(a, fitting.GaussianFittingN))
    models = {}
    for i in a.popt:
        res = a.popt[i]
        res = res.reshape((i, 3))
        weights_ = res[:, 0]
        means_ = res[:, 2].reshape((i, 1))
        covariances_ = res[:, 1].reshape((i, 1)) ** 2
        models[i] = GaussianMixture(i, covariance_type=covariance_type, **kwargs)  # kwargs for models
        param = (weights_, means_, covariances_,
                 cpc(covariances_, covariance_type=covariance_type))
        models[i]._set_parameters(param)
    return models


def generate_sample_from_pdf(n, x, pdf):
    probability = pdf / sum(pdf)
    return np.random.choice(x, size=n, p=probability).reshape((n, 1))


def aic_bic_score(models, mc_tries, x, pdf, sample_size):
    aic, bic, score = np.zeros((len(models),)), np.zeros((len(models),)), np.zeros((len(models),))
    for i in range(mc_tries):
        sample = generate_sample_from_pdf(sample_size, x, pdf)
        aic += np.array([models[i].aic(sample) for i in models])
        bic += np.array([models[i].bic(sample) for i in models])
        score += np.array([models[i].score(sample) for i in models])
    aic /= mc_tries
    bic /= mc_tries
    score /= mc_tries
    return aic, bic, score
