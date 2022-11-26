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

class myDataset(Dataset):
    def __init__(self, train_path, target_path, key_path):
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

        self.target_features = ['N','P','cnt_n_2']

    def __len__(self):
        """
        返回dataset的规模/ 有多少个settings可以用来学习
        :return: the num of files in the dataset
        """
        return len(self.target_all_path)

    def __getitem__(self, index):
        """
        返回第index个 training data，target data
        :param index: 文件序
        :return: train_data, target_data_withkey
        """
        train_path_i_path = os.path.join(self.train_root_path,self.train_all_path[index])
        target_path_i_path = os.path.join(self.target_root_path,self.target_all_path[index])
        train_df = pd.read_csv(train_path_i_path,encoding="utf-8")
        target_df = pd.read_csv(target_path_i_path,encoding="utf-8")

        # transform into tensor
        train_data = torch.tensor(train_df.values)
        target_data = torch.tensor(target_df.values)

        return train_data, target_data

# training data
train_path = r"../data/train"

# target data
target_path = r"../data/targets"
# data keys (for target)
data_key_path = r"../data/target_datakey.csv"

if __name__ == '__main__':
    dataset = myDataset(train_path, target_path, data_key_path)
    # train_data, target_data = dataset.__getitem__(0)
    # print(train_data.shape,target_data.shape)