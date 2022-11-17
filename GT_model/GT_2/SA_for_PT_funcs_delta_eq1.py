#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:47
# @Author  : Wang Yujia
# @File    : SA_for_PT_funcs_delta_eq1.py
# @Description : SA_for_PT_model_delta_eq1.ipynb需要的functions。注意是delta=1的情况下的，有所简化

import numpy as np

# prob. weighting func。根据Eq(5)
def OMEGA(p):
    return p

# C_{t-1}
def C(t,b):
    return 0.2*t*b

# the valuation function

def f(x, alpha):
    return (1-np.exp(-alpha*x))

def f_Equi(t,v,d,b,alpha,labda):

    tmp = v-d*t-C(t-1,b) - b

    if (tmp>=0):

        root = (labda*f(C(t-1,b),alpha) + f(tmp,alpha)) / (labda*f(C(t-1,b)+b,alpha) + f(tmp,alpha))

    else:

        root = (f(C(t-1,b),alpha) - f(-tmp,alpha)) / (f(C(t-1,b)+b,alpha) + f(-tmp,alpha))

    return root