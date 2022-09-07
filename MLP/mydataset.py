#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 13:28
# @Author  : Wang Yujia
# @File    : mydataset.py
# @Description : 重写过的Dataset方案都在这里。
#               方案1：[有idx问题]重写2个类，TrainDataset要返回2个值：data和feature_keays，TargetDataset根据feature_keays去返回target data
#               方案2：[√]写一个类，同时读取并返回Train和Target data. 注意，当loss为MLE的时候，只需要提取target的最后一列就好

# 方案1：Train和Target各自写一个class，Target要利用Train给的keys
from BasicInfo.common_settings import *
from torch.utils.data import Dataset
import torch

class TrainDataset(Dataset):
    def __init__(self, data_dir):
        """
        读取data_dir文件夹下所有data的filename
        :param data_dir: the root path of dataset
        """
        self.file_name = os.listdir(data_dir)
        self.data_path = []
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(data_dir, self.file_name[index]))

    def __len__(self):
        """
        返回整个dataset的规模/ 有多少个files
        :return: the num of files in the dataset
        """
        return len(self.file_name)

    def __getitem__(self,index):
        """
        返回第index个文件的数据
        :param index:
        :return: data和data对应的feature keys
        """
        data_df = pd.read_csv(self.data_path[index])
        data = torch.tensor(np.array(data_df))
        # key不需要是tensor
        key = data_df[:,2:5].unique()
        return data, key

class TargetDataset(Dataset):
    def __init__(self, data_path,key):
        """
        读取data_withnp_1_selected.csv
        :param data_path:
        """
        self.data_path = data_path
        self.key = key

    def __getitem__(self):
        """
        根据key返回对应的target data
        :param index:
        :return:
        """
        data = pd.read_csv(self.data_path)
        data = data.iloc[:,1:data.shape[1]]
        # 读取key
        data_withkey = data[(data.product_id == self.key[0]) & (data.bidincrement == self.key[1]) & (data.bidfee == self.key[2])]
        data_p = data_withkey['p_1']
        data_p = torch.tensor(np.array(data_p))

        return data_p

# 方案2：写一个类，同时读取并返回Train和Target data
class myDataset(Dataset):
    def __init__(self, train_data_dir, target_data_path):
        """
        读取data_dir路径下所有训练data files的path
        :param data_dir: data root path
        """
        self.file_name = os.listdir(train_data_dir)
        self.data_path = []
        self.target_data_path = target_data_path
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(train_data_dir, self.file_name[index]))

    def __len__(self):
        """
        返回整个dataset的规模/ 有多少个files
        :return: the num of files in the dataset
        """
        return len(self.file_name)

    def __getitem__(self, index):
        """
        返回第index个训练集的数据，提取第index个训练集的key并筛选对应的target data
        :param index: 文件序
        :return: train_data, target_data_withkey
        """
        train_data_df = pd.read_csv(self.data_path[index])
        key = train_data_df.iloc[0, 2:5]
        target_data_df = pd.read_csv(self.target_data_path)
        # select target data with a specific key
        # 加上.copy()可以避免后面drop时报错
        target_data_withkey = target_data_df[(target_data_df.product_id == key[0]) & (target_data_df.bidincrement == key[1]) & (target_data_df.bidfee == key[2])].copy()
        # 注意，当loss为MLE的时候，只需要提取target的最后一列就好
        target_data_withkey = target_data_withkey.iloc[:,(target_data_withkey.shape[1]-1)]
        train_data = torch.tensor(np.array(train_data_df))
        target_data_withkey = torch.tensor(np.array(target_data_withkey))
        return train_data, target_data_withkey