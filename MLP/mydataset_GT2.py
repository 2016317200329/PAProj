#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/22 23:47
# @Author  : Wang Yujia
# @File    : mydataset_GT2.py
# @Description : Dataset for GT2 & GT3 Inference

from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

class myDataset(Dataset):
    def __init__(self, train_path, target_path_metric,target_path_loss, key_path):
        """
        读取data path
        target_path: target data path
        target_params_path: data path of GT-2 params
        """
        self.train_root_path = train_path
        self.target_metric_root_path = target_path_metric
        self.target_loss_root_path = target_path_loss

        # all_path里有全部的data地址作为list
        self.train_all_path = os.listdir(train_path)
        self.target_all_path = os.listdir(target_path_metric)

        self.key_path = key_path

    def __len__(self):
        """
        返回dataset的规模/ 有多少个settings可以用来学习
        :return: the num of files in the dataset
        """
        return len(self.train_all_path)

    def __getitem__(self, index):
        """
        返回第index个 training data，target data, setting data
        :param index: 文件序
        :return:
        """
        # print(f"index={index}")
        train_path_i_path = os.path.join(self.train_root_path,self.train_all_path[index])
        target_loss_path_i_path = os.path.join(self.target_loss_root_path,self.target_all_path[index])
        target_metric_i_path = os.path.join(self.target_metric_root_path,self.target_all_path[index])

        train_df = pd.read_csv(train_path_i_path,encoding="utf-8")
        target_loss_df = pd.read_csv(target_loss_path_i_path,encoding="utf-8")
        target_metric_df = pd.read_csv(target_metric_i_path,encoding="utf-8")

        settings = pd.read_csv(self.key_path,encoding="utf-8")
        settings_df = settings.iloc[index,:]

        # Transform into numpy (not tensor!)
        train_data = np.array(train_df.values,dtype=float)
        target_loss_data = np.array(target_loss_df.values,dtype=float)
        target_metric_data = np.array(target_metric_df.values,dtype=float)

        settings_data = np.array(0.,dtype=float)            # 'desc'无法处理, 本来应该是 settings_df

        return train_data, target_metric_data, target_loss_data, settings_data

######################### TEST USE ########################

# train_pct = 0.7
#
# # training data
# train_path = "../data/train_8_all"
#
# # target data
# target_path = r"../data/targets_all"
# params_opitim_path = r"../data/SA_PT/params_opitim_delta_T.csv"
#
# # data keys (for target)
# data_key_path = r"../data/target_datakey_all.csv"
#
# # NLL metric
# NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30_all.csv"

# if __name__ == '__main__':
#     dataset = myDataset(train_path, target_path, params_opitim_path, data_key_path, NLL_metric_path)
#
#     i=100
#     train_data, target_data, params_opitim,settings_data,metric_df = dataset.__getitem__(i)
#     print(target_data.shape)
#     print(train_data.shape)
#     print(params_opitim.shape)
