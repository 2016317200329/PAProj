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
    def __init__(self, path_all):
        """
        读取data path

        """
        self.GT_1_data_path = path_all[0]
        self.GT_2_data_path = path_all[1]
        self.target_data_path_head = path_all[2]
        self.data_key_path = path_all[3]
        self.target_features = ['N','P','cnt_n_2']

    def __len__(self):
        """
        返回dataset的规模/ 有多少个settings可以做infer
        :return: the num of files in the dataset
        """
        data_key = pd.read_csv(self.data_key_path)
        return data_key.shape[0]

    def __getitem__(self, index):
        """
        返回第index个 key，training data，target data
        :param index: 文件序
        :return: train_data, target_data_withkey
        """
        # 把这4个files 读进内存里
        # read in
        train_data_all_1 = pd.read_csv(self.GT_1_data_path)
        train_data_all_2 = pd.read_csv(self.GT_2_data_path)
        data_key_all = pd.read_csv(self.data_key_path)
        target_data_i = pd.read_csv(self.target_data_path_head + str(index) + ".csv",encoding="utf-8")

        # get the key from the 'index'
        data_key_i = data_key_all.iloc[index,:]
        data_key_i = data_key_i.copy()

        # select data with the key or the feature
        train_data_i_1_df = train_data_all_1[(train_data_all_1['bidincrement'] == data_key_i[1]) &
                                            (train_data_all_1['bidfee'] == data_key_i[2]) &
                                            (train_data_all_1['retail'] == data_key_i[3])].copy()

        train_data_i_2_df = train_data_all_2[(train_data_all_2['bidincrement'] == data_key_i[1]) &
                                            (train_data_all_2['bidfee'] == data_key_i[2]) &
                                            (train_data_all_2['retail'] == data_key_i[3])].copy()
        target_data_i_df = target_data_i[self.target_features].copy()

        # transform P&N into tensor
        p_1 = torch.from_numpy( np.array(train_data_i_1_df.iloc[:,len(data_key_i)-1:]) )
        p_2 = torch.from_numpy( np.array(train_data_i_2_df.iloc[:,len(data_key_i)-1:]) )
        target_data = torch.from_numpy(np.array(target_data_i_df))
        # data_key = torch.tensor(data_key_i)

        # train_data = torch.cat([p_1,p_2])
        return p_1, p_2, target_data, data_key

# input data
GT_1_data_path = "../data/info_asymm/results/asc_symmetry/GT_asc_symmetry_P2_K=300.csv"
GT_2_data_path = "../data/SA_PT/results/PT_oneforall_P_K=300.csv"

# target data
target_output_head = "../data/targets/target_data_NP_"
# data keys (for target)
data_key_path = "../data/targets/target_datakey.csv"

if __name__ == '__main__':
    test_dta = myDataset([GT_1_data_path,GT_2_data_path, target_output_head, data_key_path])
    p_1, p_2, target_data, data_key =  test_dta.__getitem__(0)

