#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/18 10:51
# @Author  : Wang Yujia
# @File    : PT_demo_multiP_lossfunc.py
# @Description :

import sympy
import pandas as pd
import numpy as np
import datetime


def C(t, b):
    return 0.2 * t * b


def OMEGA(p, delta):
    return p ** delta * ((p ** delta + (1 - p) ** delta) ** (-1 / delta))


# valuation function
def f(x, alpha):
    return (1 - sympy.E ** (-alpha * x)) / alpha
    # when x < 0, in fact, it shoule be : (-labda)*(1-sympy.E**(alpha*x))/alpha


def f_Equi(t, v, d, b, alpha, labda, delta):
    u = sympy.Symbol('u')

    tmp = v - d * t - C(t - 1, b) - b

    func_1 = (labda * f(x=C(t - 1, b), alpha=alpha) - labda * OMEGA(u, delta) * f(x=(C(t - 1, b) + b),
                                                                                  alpha=alpha) + OMEGA(1 - u,
                                                                                                       delta) * f(
        tmp, alpha))
    func_2 = (-f(x=C(t - 1, b), alpha=alpha) + OMEGA(u, delta) * f(x=(C(t - 1, b) + b), alpha=alpha) + (
            1 - OMEGA(u, delta)) * f(-tmp, alpha))

    if (tmp >= 0):
        return sympy.nsolve(func_1, (0, 1), solver='bisect', verify=False)
    else:
        return sympy.nsolve(func_2, (0, 1), solver='bisect', verify=False)


def loss_func(params, settings):
    start_time = datetime.datetime.now()

    alpha = params[0]
    delta = params[1]
    labda = params[2]
    print("aaaaa,settings：",settings)
    v, b, d = settings[0], settings[1], settings[2]
    max_T, cnt_row = settings[3],settings[4]
    N_i = settings[5]
    cnt_n_2 = settings[6]


    # solve for U from Equi. condt.
    U_i = [0] * (max_T + 1)
    U_i[0] = 1

    for t in range(1, max_T + 1):
        U_i[t] = f_Equi(t, v, d, b, alpha, labda, delta)

    # calculate NLL under this auction setting & PT params
    nll = 0
    if (U_i[0] == 1):
        U_i.pop(0)  # because U_i[0]=1
    U_tmp_df = pd.DataFrame(U_i, index=np.arange(0, U_i.__len__()), columns=['U'], dtype=float)
    # cnt_row = data_i.shape[0]
    for idx in range(0, cnt_row):
        # sum up the log prob among all durations of this auction
        nll += (np.sum(U_tmp_df[0:(N_i[idx] - 1)][:].apply(np.log, axis=1)) + np.log(
            1 - U_tmp_df.iat[(N_i[idx] - 1), 0])) * cnt_n_2[idx]
    #
    print('loss_func costs {time_costs}s \n'.format(
        time_costs=(datetime.datetime.now() - start_time).total_seconds()))
    return float(-nll)