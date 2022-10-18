#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/17
# @Author  : github.com/guofei9987; Wang Yujia
# @File    : SA_PT.py.py
# @Description : Modify code for inference purpose in PT model

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/17
# @Author  : github.com/guofei9987

import numpy as np
from sko.base import SkoBase
import numpy as np
from PT_demo_multiP_lossfunc import loss_func

import pandas as pd
import sympy
import time
import datetime
from sko.tools import set_run_mode
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
#from sko.SA import SABoltzmann
import multiprocessing
from multiprocessing.dummy import Pool


class SimulatedAnnealingBase(SkoBase):
    """
    DO SA(Simulated Annealing)

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    x0 : array, shape is n_dim
        initial solution
    T_max :float
        initial temperature
    T_min : float
        end temperature
    L : int
        num of iteration under every temperature（Long of Chain）

    Attributes
    ----------------------


    Examples
    -------------
    See https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py
    """

    def __init__(self, x0, settings, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        assert T_max > T_min > 0, 'T_max > T_min > 0'

        # self.func = func
        # wyj: add settings for loss func
        # settings is a list
        self.settings = settings
        self.T_max = T_max  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = int(L)  # num of iteration under every temperature（also called Long of Chain）
        # stop if best_y stay unchanged over max_stay_counter times (also called cooldown time)
        self.max_stay_counter = max_stay_counter

        self.n_dim = len(x0)

        self.best_x = np.array(x0)  # initial solution
        self.best_y = loss_func(self.best_x,self.settings)
        self.T = self.T_max
        self.iter_cycle = 0
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]
        # history reasons, will be deprecated
        self.best_x_history, self.best_y_history = self.generation_best_X, self.generation_best_Y

    def get_new_x(self, x):
        u = np.random.uniform(-1, 1, size=self.n_dim)
        x_new = x + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)
        return x_new

    def cool_down(self):
        self.T = self.T * 0.7

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        # y_current记录了当前iteration中最好的结果，y_current <= self.best_y
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0
        while True:
            print("-------------- {}_th iteration --------------\n".format(self.iter_cycle))
            for i in range(self.L):
                print("---------- {}/L ----------\n".format(i))
                x_new = self.get_new_x(x_current)
                # print("x_new after clipping: ", x_new)
                y_new = loss_func(x_new,self.settings)

                # Metropolis
                df = y_new - y_current
                print("y_new - y_current is {}: ".format(df))
                # print("---------- {}/L ----------\n".format(i))
                if ((df < 0.0) | (np.exp(-df / self.T) > np.random.rand())):
                    x_current, y_current = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if self.T < self.T_min:
                stop_code = 'Cooled to final temperature'
                break
            if stay_counter > self.max_stay_counter:
                stop_code = 'Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter)
                break
        return self.best_x, self.best_y

    fit = run


class SimulatedAnnealingValue(SimulatedAnnealingBase):
    """
    SA on real value function
    """

    def __init__(self, x0, settings,T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__( x0, settings,T_max, T_min, L, max_stay_counter, **kwargs)
        lb, ub = kwargs.get('lb', None), kwargs.get('ub', None)

        if lb is not None and ub is not None:
            self.has_bounds = True
            self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
            assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
            assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
            self.hop = kwargs.get('hop', self.ub - self.lb)
        elif lb is None and ub is None:
            self.has_bounds = False
            self.hop = kwargs.get('hop', 10)
        else:
            raise ValueError('input parameter error: lb, ub both exist, or both not exist')
        self.hop = self.hop * np.ones(self.n_dim)


class SABoltzmann(SimulatedAnnealingValue):
    """
    std = minimum(sqrt(T) * ones(d), (upper - lower) / (3*learn_rate))
    y ~ Normal(0, std, size = d)
    x_new = x_old + learn_rate * y

    T_new = T0 / log(1 + k)
    """

    def __init__(self, x0, settings, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__(x0, settings,T_max, T_min, L, max_stay_counter, **kwargs)
        self.learn_rate = kwargs.get('learn_rate', 0.5)

    def get_new_x(self, x):
        a, b = np.sqrt(self.T), self.hop / 3.0 / self.learn_rate
        # wyj: in our case, std = b = self.hop / 3.0 / self.learn_rate mostly
        std = np.where(a < b, a, b)
        xc = np.random.normal(0, 1.0, size=self.n_dim)
        # wyj: in our case, std * self.learn_rate = self.hop / 3.0 / self.learn_rate * self.learn_rate = self.hop / 3.0 = 10/3
        # wyj: self.learn_rate is USELESS!!  And x_new is far bigger than x_old
        x_new = x + xc * std * self.learn_rate
        print("x_new before clipping: ",x_new)
        if self.has_bounds:
            return np.clip(x_new, self.lb, self.ub)
        return x_new

    def cool_down(self):
        # wyj: T goes down slower than SACauchy due to np.log() here
        self.T = self.T_max / np.log(self.iter_cycle + 1.0)


