#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/18 10:31
# @Author  : Wang Yujia
# @File    : PT_demo_multiP.py
# @Description : 多线程实验。因为ipynb以及self内部的func对多线程支持不好， 写到外面。【必须SA和SA之间是抢占的，不能在一个SA内部抢占，一个SA内部无法抢占】

###################### Global Set Up######################################
#import numpy as np
#import torch
#import cupy as np
import pandas as pd
import sympy
import time
import datetime
# from sko.tools import set_run_mode
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
#from sko.SA import SABoltzmann
from SA_PT import SABoltzmann as SABoltzmann
import multiprocessing
from multiprocessing.dummy import Pool

# auction settings
b1 = 0.75
v1 = 27.99
d1 = 0.01

b2 = 0.75
v2 = 199.99
d2 = 0.01

# cnt_row = data_i.shape[0]
cnt_row = 5
# cnt_n_2 = data_i['cnt_n_2']
cnt_n_2 = [1,1,2,3,1]
#N_i = data_i['N']
N_i = [500,50,100,200,300]
# max duration
max_T1 = 831
max_T2 = 600

# initial params
table_5_M = [0.025,0.85,3.72]
# lower/ upper bound
lb = [-0.3,0.01,0.01]
ub = [0.3, 2, 16]

###################### Global Set Up######################################


def task(args):

    #set_run_mode(loss_func, 'cached')
    #set_run_mode(loss_func, 'multithreading')
    params,settings = args
    sa_boltzmann = SABoltzmann(x0=params, settings=settings,T_max=1000, T_min=1e-5, learn_rate=0.01, L=20, max_stay_counter=5,
                                lb=lb, ub=ub)
    print("> Now do SA....... \n")

    sa_boltzmann.run()

    print("SA ends \n")

#多线程测试
if __name__ == '__main__':
    multiprocessing.freeze_support() # 在Windows下编译需要加这行
    print("The num of CPU of this computer is : ",multiprocessing.cpu_count())
    # 生成进程池
    pool = Pool(4)
    # 添加进程

    params1 = table_5_M
    settings1 = [v1,b1,d1, max_T1, cnt_row, N_i, cnt_n_2]
    params2 = table_5_M
    settings2 = [v2,b2,d2, max_T2, cnt_row, N_i, cnt_n_2]
    p1 = pool.map_async(task, [(params1,settings1),(params2,settings2)])

    # 不再添加进程
    pool.close()

    # 主程序等待所有进程完成
    pool.join()

    # 获取进程函数返回值(如果任务存在返回值)
    p1_value = p1.get()

    print(f'进程1返回值：{p1_value}')