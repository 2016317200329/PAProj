#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 11:02
# @Author  : Wang Yujia
# @File    : my_collate_fn.py
# @Description : 为dataloader整理数据

import torch
import numpy as np

def my_collate_fn(data):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data.sort(key=lambda x: len(x[1]), reverse=False)   # 按照targets数据长度升序排序
    max_len = len(data[-1][1])                         # 选取最长的targets数据长度

    data_list = []
    target_list = []

    # # padding with 0
    batch = 0
    while data[batch][1].shape[0] < max_len:
        tmp = np.array([[0,0]]* (max_len - data[batch][1].shape[0]))
        data_list.append(data[batch][0])                # 原样保存training data
        print(f"compare {data[batch][1].shape} with {tmp.shape}")
        target_list.append(np.concatenate([data[batch][1], tmp], axis=0 ))
        batch += 1

    while batch < len(data):
        data_list.append(data[-1][0])
        target_list.append(data[-1][1])
        batch += 1

    data_tensor = torch.from_numpy(np.stack(data_list)).float()
    target_tensor = torch.from_numpy(np.stack(target_list)).float()

    return (data_tensor, target_tensor)