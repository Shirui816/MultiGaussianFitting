#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: shirui <shirui816@gmail.com>

# Generally

from abc import ABCMeta, abstractmethod
from scipy.optimize import curve_fit
from typing import Callable, Any
import inspect
from scipy.integrate import simps


class FittingLsq(metaclass=ABCMeta):
    def __init__(self, bounds, p0):
        self.p0 = p0
        self.bounds = bounds
        self.popt_res = {}
        self.popt = {}
        self.pcov = {}
        self.funcs = {}
        self._check_parameter()
        self.normal_factor = {}

    def _check_parameter(self):
        if not isinstance(self.bounds, tuple):
            raise(Exception("Value error, bounds must be tuple!"))
        if not (isinstance(self.p0, tuple) or isinstance(self.p0, str)):
            raise(Exception("Value error, p0 must be tuple or `Automatic'!"))

    @abstractmethod
    def _set_p0(self, n, ks):
        pass

    @abstractmethod
    def _set_bound(self, n, ks):
        pass

    @abstractmethod
    def _set_func(self, n) -> Callable[[Any], Any]:
        pass

    @abstractmethod
    def _get_params(self, x, y):
        pass

    def n_parameters(self, i):
        return len(self.popt_res[i])

    def fit(self, x, y, n):
        self._get_params(x, y)
        for i in range(1, n + 1):
            func = self._set_func(i)
            self.funcs[i] = func
            self.popt_res[i] = list(inspect.signature(func).parameters.keys())
            p0 = self._set_p0(i, self.popt_res[i])
            bounds = self._set_bound(i, self.popt_res[i])
            self.popt[i], self.pcov[i] = curve_fit(func, x, y, p0=p0, bounds=bounds)
            self.normal_factor[i] = 1/simps(func(x, *self.popt[i]), x)
        return self
