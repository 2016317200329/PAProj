#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 13:28
# @Author  : Wang Yujia
# @File    : mydataset.py
# @Description : 重写过的Dataset方案都在这里。
#               方案1：[有idx问题]重写2个类，TrainDataset要返回2个值：data和feature_keays，TargetDataset根据feature_keays去返回target data
#               方案2：[√]写一个类，同时读取并返回Train和Target data. 注意，当loss为MLE的时候，只需要提取target的最后一列就好
import pandas as pd

# 方案1：Train和Target各自写一个class，Target要利用Train给的keys
from BasicInfo.common_settings import *
from torch.utils.data import Dataset
import torch
import numpy as np

features = ['product_id','bidincrement','bidfee','retail']
#
# class TrainDataset(Dataset):
#     def __init__(self, data_dir):
#         """
#         读取data_dir文件夹下所有data的filename
#         :param data_dir: the root path of dataset
#         """
#         self.file_name = os.listdir(data_dir)
#         self.data_path = []
#         for index in range(len(self.file_name)):
#             self.data_path.append(os.path.join(data_dir, self.file_name[index]))
#
#     def __len__(self):
#         """
#         返回整个dataset的规模/ 有多少个files
#         :return: the num of files in the dataset
#         """
#         return len(self.file_name)
#
#     def __getitem__(self,index):
#         """
#         返回第index个文件的数据
#         :param index:
#         :return: data和data对应的feature keys
#         """
#         data_df = pd.read_csv(self.data_path[index])
#         data = torch.tensor(np.array(data_df))
#         # key不需要是tensor
#         key = data_df[:,2:5].unique()
#         return data, key
#
# class TargetDataset(Dataset):
#     def __init__(self, data_path,key):
#         """
#         读取data_withnp_1_selected.csv
#         :param data_path:
#         """
#         self.data_path = data_path
#         self.key = key
#
#     def __getitem__(self):
#         """
#         根据key返回对应的target data
#         :param index:
#         :return:
#         """
#         data = pd.read_csv(self.data_path)
#         data = data.iloc[:,1:data.shape[1]]
#         # 读取key
#         data_withkey = data[(data.product_id == self.key[0]) & (data.bidincrement == self.key[1]) & (data.bidfee == self.key[2])]
#         data_p = data_withkey['p_1']
#         data_p = torch.tensor(np.array(data_p))
#
#         return data_p

#%% md

# 方案2：写一个类，同时读取并返回Train和Target data
class myDataset(Dataset):
    def __init__(self, train_data_dir, target_data_path,data_key_path):
        """
        读取data_dir路径下所有训练data files的path

        """
        self.train_data_dir = train_data_dir
        self.target_data_path = target_data_path
        self.data_key_path = data_key_path

    def __len__(self):
        """
        返回dataset的规模/ 有多少个settings可以做infer
        :return: the num of files in the dataset
        """
        data_key = pd.read_csv(self.data_key_path)
        return data_key.shape[0]

    def transform(self,str_p):
        """
        transform str 'P' into array 'P'
        :return:
        """
        a = np.array(np.mat(str_p))
        a_vec = a.flatten()
        return a_vec

    def __getitem__(self, index):
        """
        返回第index个 key，training data，target data
        :param index: 文件序
        :return: train_data, target_data_withkey
        """
        # 还是要把这个4个files 读进内存里
        train_data_all_1 = pd.read_csv(self.train_data_dir[0])
        train_data_all_2 = pd.read_csv(self.train_data_dir[1])
        target_data_all = pd.read_csv(self.target_data_path)
        data_key_all = pd.read_csv(self.data_key_path)

        # get the key from the 'index'
        data_key = data_key_all[index,:]

        # select target data with the key
        train_data_i_1_df = train_data_all_1[(train_data_all_1['bidincrement'] == data_key[1]) &
                                           (train_data_all_1['bidfee'] == data_key[2]) &
                                           (train_data_all_1['retail'] == data_key[2])].copy()

        train_data_i_2_df = train_data_all_2[(train_data_all_2['bidincrement'] == data_key[1]) &
                                           (train_data_all_2['bidfee'] == data_key[2]) &
                                           (train_data_all_2['retail'] == data_key[3])].copy()

        target_data_i_df = target_data_all[(target_data_all['product_id'] == data_key[0]) &
                                           (target_data_all['bidincrement'] == data_key[1]) &
                                           (target_data_all['bidfee'] == data_key[2]) &
                                           (target_data_all['retail'] == data_key[3])].copy()

        # transform P into array vector
        p_1_i = self.transform(train_data_i_1_df.loc[:, 'P'])
        p_2_i = self.transform(train_data_i_2_df.loc[:, 'P'])

        #
        # # 当loss为MLE的时候，只需要提取target的'P'列就好
        # target_data_withkey = target_data_withkey.iloc[:,(target_data_withkey.shape[1]-1)]
        # train_data = torch.tensor(np.array(train_data_df))
        # target_data_withkey = torch.tensor(np.array(target_data_withkey))

        return data_key, (p_1_i, p_2_i),
