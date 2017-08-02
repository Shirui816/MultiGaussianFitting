#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: shirui <shirui816@gmail.com>


from .fitting import FittingLsq
import numpy as np
from scipy.integrate import simps
from gaussian import n_gaussian


class GaussianFittingN(FittingLsq):
    def __init__(self, bounds=(), p0='Automatic', known=()):
        super(GaussianFittingN, self).__init__(bounds=bounds, p0=p0)
        self.mean = None
        self.var = None
        self.known = known

    def _gaussian_params(self, x, y):
        mean = simps(x * y, x)
        var = np.sqrt(simps((x-mean) ** 2 * y, x))
        self.mean = mean
        self.var = var

    def _get_params(self, x, y):
        self._gaussian_params(x, y)

    def _set_p0(self, n, ks):
        p0 = np.zeros((len(ks),))
        params = [1/n, self.var, self.mean]
        j = 0
        if isinstance(self.p0, str):
            if self.p0.upper() == 'AUTOMATIC':
                for k in ks:
                    for _id, _label in enumerate(['a', 's', 'm']):
                        if _label in k:
                            p0[j] = params[_id]
                            j += 1
        else:
            for k in ks:
                for _id, _label in enumerate(['a', 's', 'm']):
                    if _label in k:
                        p0[j] = self.p0[_id] if not self.p0[_id] is None else params[_id]
                        j += 1
        return p0

    def _set_bound(self, n, ks):
        lb = [-np.inf for _ in ks]
        ub = [+np.inf for _ in ks]
        j = 0
        if not self.bounds:
            return [lb, ub]
        for k in ks:
            for _id, _label in enumerate(['a', 's', 'm']):
                if _label in k:
                    lb[j] = self.bounds[_id][0] if not self.bounds[_id][0] is None else -np.inf
                    ub[j] = self.bounds[_id][1] if not self.bounds[_id][1] is None else +np.inf
                    j += 1
        return [lb, ub]

    def _set_func(self, n):
        return n_gaussian(n, known=self.known)
