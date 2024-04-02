#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 11:02
# @Author  : Wang Yujia
# @File    : my_collate_fn.py
# @Description : 为dataloader整理数据

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
# from Config.config_base import BaseConfig
# opt = BaseConfig()

SAFETY = 1e-30

### padding with opt.SAFETY and rescale
# 1. padding 太多0不利于学习，
# 2. Standardize and sum=1
def my_collate_fn_3(data, INPUT_LIST=[1,2,3,4]):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, (TARGET=1)
    target_loss_list = []       # target data for computing loss of NN, (TARGET=5)
    setting_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:
        # print("shape:",data[batch][0].shape) #shape: (3, 300)

        # 对于所有用0填充的data来说，最好用一个数字补齐这些0
        data[batch][0][0,np.where(data[batch][0][0]==0)] = SAFETY
        data[batch][0][1,np.where(data[batch][0][1]==0)] = SAFETY
        data[batch][0][2,np.where(data[batch][0][2]==0)] = SAFETY

        # 归一化
        data[batch][0][0] = data[batch][0][0]/sum(data[batch][0][0])
        data[batch][0][1] = data[batch][0][1]/sum(data[batch][0][1])
        data[batch][0][2] = data[batch][0][2]/sum(data[batch][0][2])

        # Select dim of data as need.
        data_tmp = np.array([data[batch][0][dim - 1] for dim in INPUT_LIST])
        data_list.append(torch.tensor(data_tmp))

        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()

    #index = data[:][5]
    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor


def my_collate_fn_4(data):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, (TARGET=1)
    target_loss_list = []       # target data for computing loss of NN, (TARGET=5)
    setting_list = []
    metric_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:

        data_list.append(torch.tensor(data[batch][0]))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor

### 当只有1个GT input时 (GT_CHOSEN)：
def my_collate_fn_1GT(data, GT_CHOSEN):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, (TARGET=1)
    target_loss_list = []       # target data for computing loss of NN, (TARGET=5)
    setting_list = []
    metric_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:

        # 对于所有用0填充的data来说，最好用一个数字补齐这些0
        data[batch][0][GT_CHOSEN,np.where(data[batch][0][GT_CHOSEN]==0)] = SAFETY
        # 向量归一化
        data[batch][0][GT_CHOSEN] = data[batch][0][GT_CHOSEN]/sum(data[batch][0][GT_CHOSEN])

        # 然后再和embedding拼接
        data_tmp = np.array([data[batch][0][GT_CHOSEN,:],data[batch][0][3,:]])
        data_list.append(torch.tensor(data_tmp))

        # data_list.append(torch.tensor(data[batch][0]))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor

# 当输入没有emd时
def my_collate_fn_woemd(data):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, (TARGET=1)
    target_loss_list = []       # target data for computing loss of NN, (TARGET=5)
    setting_list = []
    metric_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:
        # print("shape:",data[batch][0].shape) #shape: (3, 300)

        # 对于所有用0填充的data来说，最好用一个数字补齐这些0
        data[batch][0][0,np.where(data[batch][0][0]==0)] = SAFETY
        data[batch][0][1,np.where(data[batch][0][1]==0)] = SAFETY

        # 向量归一化
        data[batch][0][0] = data[batch][0][0]/sum(data[batch][0][0])
        data[batch][0][1] = data[batch][0][1]/sum(data[batch][0][1])

        data_tmp = np.array([data[batch][0][0,:],data[batch][0][1,:]])
        data_list.append(torch.tensor(data_tmp))

        # data_list.append(torch.tensor(data[batch][0]))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor

# InferNet
def my_collate_fn_GT2(data):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN
    target_loss_list = []       # target data for computing loss of NN, (ls T)

    target_params_list = []       # target data for computing loss of NN, (TARGET=5 or 1)
    setting_list = []
    metric_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:
        data_list.append(torch.tensor(data[batch][0]))
        # target_data只保留N这一列
        # print("target[0] shape:",torch.tensor(data[batch][1][:,0]).shape)
        target_metric_list.append(torch.tensor(data[batch][1][:,0], dtype=torch.int64))
        target_loss_list.append(torch.tensor(data[batch][2][:,0], dtype=torch.int64))
        target_params_list.append(torch.tensor(data[batch][3]))
        setting_list.append(torch.tensor(data[batch][4]))
        metric_list.append(torch.tensor(data[batch][5]))
        batch += 1


    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_tensor = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    target_params_tensor = torch.stack(target_params_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    ## 省点内存
    del target_metric_padded,target_loss_padded, data_list, target_metric_list,target_loss_list,setting_list,metric_list
    return data_tensor, target_metric_tensor, target_loss_tensor, target_params_tensor, setting_tensor, metric_tensor


# 只提取GT1(2)或者embedding，such that input is 1-dim
def my_collate_fn_0GT(data, which_dim):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, (TARGET=1)
    target_loss_list = []       # target data for computing loss of NN, (TARGET=5)
    setting_list = []
    metric_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:
        if which_dim == 1 or which_dim == 0:
            # 对于所有用0填充的data来说，最好用一个数字补齐这些0
            data[batch][0][which_dim,np.where(data[batch][0][which_dim]==0)] = SAFETY
            # sum==1
            data[batch][0][which_dim] = data[batch][0][which_dim]/sum(data[batch][0][which_dim])


        data_list.append(torch.tensor(data[batch][0][which_dim]).unsqueeze(0))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor



def my_collate_fn_mask(data):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, (TARGET=1)
    target_loss_list = []       # target data for computing loss of NN, (TARGET=5)
    setting_list = []
    metric_list = []
    val_idx_list = []
    test_idx_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:
        # print("shape:",data[batch][0].shape) #shape: (3, 300)

        # 对于所有用0填充的data来说，最好用一个数字补齐这些0
        data[batch][0][0,np.where(data[batch][0][0]==0)] = SAFETY
        data[batch][0][1,np.where(data[batch][0][1]==0)] = SAFETY

        # 归一化
        data[batch][0][0] = data[batch][0][0]/sum(data[batch][0][0])
        data[batch][0][1] = data[batch][0][1]/sum(data[batch][0][1])


        data_list.append(torch.tensor(data[batch][0]))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        val_idx_list.append(torch.tensor(data[batch][5]))
        test_idx_list.append(torch.tensor(data[batch][6]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    val_idx_padded = pad_sequence(val_idx_list,batch_first=True)
    test_idx_padded = pad_sequence(test_idx_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()
    val_tensor = val_idx_padded.int()
    test_tensor = test_idx_padded.int()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()


    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor,val_tensor,test_tensor



def my_collate_fn_er(data):
    er_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:

        er_list.append(torch.tensor(data[batch]))
        batch += 1

    er_tensor = torch.stack(er_list).float()

    return er_tensor

