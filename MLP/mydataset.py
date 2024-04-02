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
from torchvision import transforms

def custom_transform(data):
    # 对第一列进行归一化，第二列不变
    data = (data - data.mean()) / data.std()
    return data

# 定义一个包含自定义transform操作的Transform对象
transform = transforms.Compose([
    transforms.Lambda(custom_transform), # 自定义transform
])

class myDataset(Dataset):
    def __init__(self, train_path, target_path_metric, target_path_loss, key_path, transform_flag = False):
        """
        读取data path
        target_path_metric: 计算metric用的original data
        target_path_loss： 计算loss / training用的data
        transform_flag： 是否执行归一化操作
        """
        self.train_root_path = train_path
        self.target_metric_root_path = target_path_metric
        self.target_loss_root_path = target_path_loss

        # all_path里有全部的data地址作为list
        self.train_all_path = os.listdir(train_path)
        # target_path里有全部的target data地址
        self.target_all_path = os.listdir(target_path_metric)
        self.key_path = key_path


        self.transform = transform

    def __len__(self):
        """
        返回dataset的规模/ 有多少个settings可以用来学习
        :return: the num of files in the dataset
        """
        return len(self.target_all_path)

    def __getitem__(self, index):
        """
        返回第index个 training data，target data, setting data
        :param index: 文件序
        :return: train_data, target_data
        """
        # print(f"target file is {self.target_all_path[index]}")
        train_path_i_path = os.path.join(self.train_root_path,self.train_all_path[index])
        target_loss_path_i_path = os.path.join(self.target_loss_root_path,self.target_all_path[index])
        target_metric_i_path = os.path.join(self.target_metric_root_path,self.target_all_path[index])

        train_df = pd.read_csv(train_path_i_path,encoding="utf-8")
        target_loss_df = pd.read_csv(target_loss_path_i_path,encoding="utf-8")
        target_metric_df = pd.read_csv(target_metric_i_path,encoding="utf-8")

        settings = pd.read_csv(self.key_path,encoding="utf-8")

        settings_df = settings.iloc[index,1:]         # 'desc'无法处理, 去掉这一列

        # Transform into numpy (not tensor!)
        train_data = np.array(train_df.values,dtype=float)
        target_loss_data = np.array(target_loss_df.values,dtype=float)
        target_metric_data = np.array(target_metric_df.values,dtype=float)
        settings_data = np.array(settings_df.values,dtype=float)

        return train_data, target_metric_data, target_loss_data, settings_data

class myDataset_revenue(Dataset):
    def __init__(self, target_path, exp_revenue_path):
        """
        读取data path
        target_path_metric: 计算metric用的original data
        target_path_loss： 计算loss / training用的data
        transform_flag： 是否执行归一化操作
        """
        self.target_root_path = target_path
        self.exp_revenue_path = exp_revenue_path

        self.target_all_path = os.listdir(target_path)

    def __len__(self):
        """
        返回dataset的规模/ 有多少个settings可以用来学习
        :return: the num of files in the dataset
        """
        return len(self.target_all_path)

    def __getitem__(self, index):
        """
        返回第index个 training data，target data, setting data
        :param index: 文件序
        :return: train_data, target_data
        """

        er = pd.read_csv(self.exp_revenue_path,encoding="utf-8")
        er_df = er.iloc[index,:]

        # Transform into numpy (not tensor!)
        er_data = np.array(er_df.values,dtype=float)

        return er_data



class myDataset_mask(Dataset):
    def __init__(self, train_path, target_path_metric, target_path_loss, key_path, metric_path, val_idx_path, test_idx_path, transform_flag = False):
        """
        读取data path
        target_path_metric: 计算metric用的original data
        target_path_loss： 计算loss / training用的data
        val_idx_path: myDataset没有，记录了mask的idx
        test_idx_path： myDataset没有
        transform_flag： 是否执行归一化操作
        """
        self.train_root_path = train_path
        self.target_metric_root_path = target_path_metric
        self.target_loss_root_path = target_path_loss
        self.val_idx_root_path = val_idx_path
        self.test_idx_root_path = test_idx_path

        # all_path里有全部的data地址作为list
        self.train_all_path = os.listdir(train_path)
        # target_path里有全部的target data地址
        self.target_all_path = os.listdir(target_path_metric)
        self.idx_path = os.listdir(val_idx_path)

        self.key_path = key_path
        self.metric_path = metric_path

        self.transform = transform

    def __len__(self):
        """
        返回dataset的规模/ 有多少个settings可以用来学习
        :return: the num of files in the dataset
        """
        return len(self.target_all_path)

    def __getitem__(self, index):
        """
        返回第index个 training data，target data, setting data
        :param index: 文件序
        :return: train_data, target_data
        """
        # print(f"target file is {self.target_all_path[index]}")
        train_path_i_path = os.path.join(self.train_root_path,self.train_all_path[index])
        target_loss_path_i_path = os.path.join(self.target_loss_root_path,self.target_all_path[index])
        target_metric_i_path = os.path.join(self.target_metric_root_path,self.target_all_path[index])
        val_idx_i_path = os.path.join(self.val_idx_root_path,self.idx_path[index])
        test_idx_i_path = os.path.join(self.test_idx_root_path,self.idx_path[index])

        train_df = pd.read_csv(train_path_i_path,encoding="utf-8")
        target_loss_df = pd.read_csv(target_loss_path_i_path,encoding="utf-8")
        target_metric_df = pd.read_csv(target_metric_i_path,encoding="utf-8")
        val_idx_df = pd.read_csv(val_idx_i_path,encoding="utf-8")
        test_idx_df = pd.read_csv(test_idx_i_path,encoding="utf-8")

        settings = pd.read_csv(self.key_path,encoding="utf-8")
        FLAG_SKIP_METRIC = 0
        if os.path.exists(self.metric_path):
            metric = pd.read_csv(self.metric_path, encoding="utf-8")
        else:
            print("SKIP self.metric_path")
            FLAG_SKIP_METRIC = 1
        # metric = pd.read_csv(self.metric_path,encoding="utf-8")

        settings_df = settings.iloc[index,1:]         # 'desc'无法处理, 去掉这一列
        metric_df = metric.iloc[index,:]


        # Transform into numpy (not tensor!)
        train_data = np.array(train_df.values,dtype=float)
        target_loss_data = np.array(target_loss_df.values,dtype=float)
        target_metric_data = np.array(target_metric_df.values,dtype=float)
        val_idx_data = np.array(val_idx_df.values,dtype=int)
        test_idx_data = np.array(test_idx_df.values,dtype=int)

        settings_data = np.array(settings_df.values,dtype=float)
        metric_data = np.array(metric_df.values,dtype=float)

        if FLAG_SKIP_METRIC:
            return train_data, target_metric_data, target_loss_data, settings_data, val_idx_data, test_idx_data
        else:
            return train_data, target_metric_data, target_loss_data, settings_data, metric_data,val_idx_data, test_idx_data

######################### TEST USE ########################

# train_pct = 0.7
#
# # training data
# # train_path = "../data/train_100"
# train_path = "../data/train_300_all"
#
# # target data
#
# target_path = r"../data/targets_all"
# # data keys (for target)
# data_key_path = r"../data/target_datakey_all.csv"
# # NLL metric
# NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30_all.csv"
#
# if __name__ == '__main__':
#     dataset = myDataset(train_path, target_path,target_path, data_key_path,NLL_metric_path)
#
#     for i in range(1300,1306):
#         train_data, target_data,_,settings_data,metric_df = dataset.__getitem__(i)
#         print(settings_data)
#     # for data in train_loader:
#     #     print(data)