#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 13:28
# @Author  : Wang Yujia
# @File    : mydataset.py
# @Description : 重写Dataset.同时读取并返回Train和Target data.


from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

class myDataset(Dataset):
    def __init__(self, train_path, target_path, key_path, metric_path):
        """
        读取data path

        """
        self.train_root_path = train_path
        self.target_root_path = target_path

        # all_path里有全部的data地址作为list
        self.train_all_path = os.listdir(train_path)
        # target_path里有全部的target data地址
        self.target_all_path = os.listdir(target_path)
        self.key_path = key_path
        self.metric_path = metric_path

        # [Obsolete] Rescale range
        self.a = 0
        self.b = 255

    def __len__(self):
        """
        返回dataset的规模/ 有多少个settings可以用来学习
        :return: the num of files in the dataset
        """
        return len(self.target_all_path)

    def rescale(self,data,a,b) :
        """
        [Obsolete] 对data重新rescale to range of [a,b]

        :param data: [Chanel,Dim]. Data to be rescaled
        :param a: Lower range
        :param b: Upper range
        :return: Rescaled data
        """
        # Get the max/min in each chanel and broadcast it by hand
        # print("before scale:",data)
        data_max = np.tile(pd.DataFrame(np.max(data,axis=1)), (1, data.shape[1]))
        data_min = np.tile(pd.DataFrame(np.min(data,axis=1)), (1, data.shape[1]))
        # Rescale
        data_rescaled = a + (data-data_min)*(b-a) / (data_max - data_min)

        return data_rescaled

    def __getitem__(self, index):
        """
        返回第index个 training data，target data, setting data
        :param index: 文件序
        :return: train_data, target_data
        """
        # print(f"target file is {self.target_all_path[index]}")
        train_path_i_path = os.path.join(self.train_root_path,self.train_all_path[index])
        target_path_i_path = os.path.join(self.target_root_path,self.target_all_path[index])

        train_df = pd.read_csv(train_path_i_path,encoding="utf-8")
        target_df = pd.read_csv(target_path_i_path,encoding="utf-8")
        settings = pd.read_csv(self.key_path,encoding="utf-8")
        metric = pd.read_csv(self.metric_path,encoding="utf-8")

        settings_df = settings.iloc[index,:]
        metric_df = metric.iloc[index,:]

        # print("dtrain_df shape",train_df.shape)   （3，300）

        # [Obsolete] Rescale
        # train_df.iloc[0:2,:] = self.rescale(train_df.iloc[0:2,:],self.a,self.b)

        # Transform into numpy (not tensor!)
        train_data = np.array(train_df.values,dtype=float)
        target_data = np.array(target_df.values,dtype=float)
        settings_data = np.array(settings_df.values,dtype=float)
        metric_data = np.array(metric_df.values,dtype=float)

        return train_data, target_data,settings_data, metric_data


######################### TEST USE ########################

# train_pct = 0.7
#
# # training data
# # train_path = "../data/train_100"
# train_path = "../data/train_300_v1"
#
# # target data
#
# target_path = r"../data/targets_5"
# # data keys (for target)
# data_key_path = r"../data/target_datakey.csv"
# # NLL metric
# NLL_metric_path = "../data/NLL_metric_GT.csv"
#
# if __name__ == '__main__':
#     dataset = myDataset(train_path, target_path, data_key_path,NLL_metric_path)
#     # print(type(dataset))  # <class '__main__.myDataset'>
#     # train_data, target_data = dataset.__getitem__(2)
#     # print(train_data,target_data.shape)
#
#     for i in range(1000,1001):
#         train_data, target_data,settings_data,metric_df = dataset.__getitem__(i)
#         print(i,metric_df)
#
#         # assert train_data.shape==(3,100),"AAA"
#
#     # for data in train_loader:
#     #     print(data)