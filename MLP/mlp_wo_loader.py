#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 14:08
# @Author  : Wang Yujia
# @File    : mlp.py
# @Description : mlp，但是没有dataloader
# @TODO:

from BasicInfo.common_settings import *
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

################# Global Params ###########################
# dataset划分
batch_size = 16
train_percentage = 0.7
vali_percentage = 0.3
# test_percentage = 0.1
# 训练优化
learning_rate = 0.01
total_train_step = 0
total_test_step = 0
epoch = 10
# 路径
data_withnp_1_selectedkeys_path = "data/data_withnp_1_selectedkeys.csv"
train_data_file_path = "../data/sim_data"
target_data_path = "../data/data_withnp_1_selected.csv"
data_file_head = "../data/sim_data/data_sampled_"
data_file_tail = ".csv"
datakey_path = "../data/data_withnp_1_selectedkeys.csv"
################# Global Params ###########################

writer = SummaryWriter("logs-MLP")

# 1. data读取与划分：
# 1.1
def len_datafile(data_dir):
    """
    返回整个训练集dataset的规模/ 有多少个files
    :return: the num of files in the dataset
    """
    file_name = os.listdir(data_dir)
    return len(file_name)

def readData(data_dir,index):
    """
    读取第index个data文件
    :param data_dir: root path
    :param index: 下标
    :return: 第index个数据文件(tensor)
    """
    file_name = os.listdir(data_dir)
    data_path = os.path.join(data_dir, file_name[index])
    data_df = pd.read_csv(data_path)
    # 转成张量！
    data = torch.tensor(np.array(data_df))
    return data

def readTargetData(data_path,index):
    """
    返回第index个key对应的target data
    :param index: 下标
    :param data_path: target data的路径
    :return:
    """

    data = pd.read_csv(data_path)
    data = data.iloc[:,1:data.shape[1]]
    # 读取第index个key的内容
    datakey = pd.read_csv(datakey_path)
    datakey = datakey.iloc[index,:]
    data_withkey = data[(data.product_id == datakey[0]) & (data.bidincrement == datakey[1]) & (data.bidfee == datakey[2])]
    features = ['product_id', 'bidincrement', 'bidfee']
    data_featurex = [['product_id', 'bidincrement', 'bidfee','n_1']]
    data_featurex = torch.tensor(np.array(data_featurex))
    data_p = data_withkey['p_1']
    data_p = torch.tensor(np.array(data_p))

    return data_featurex, data_p


# 2. MLP
# 2.1 MLP: 一个2+1层的MLP
# input 目前是大小为160*5的data
class Mlp(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians) -> None:
        super().__init__()
        self.mlp_call = nn.Sequential(
            # 因为input只有5个features
            # nn.Flatten(start_dim=0),
            nn.Linear(5, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 625),
            nn.ReLU(inplace=True),
            nn.Linear(625, 10),
            nn.Tanh()
        )
        # π μ σ for MDN
        self.z_pi = nn.Sequential(
            nn.Linear(10, n_gaussians),
            nn.Softmax(dim=1)
        )
        self.z_mu = nn.Linear(10, n_gaussians)
        self.z_sigma = nn.Linear(10, n_gaussians)

    def forward(self, x):
        mlp_output = self.mlp_call(x)
        # 输出n_gaussians个高斯的参数
        pi = self.z_pi(mlp_output)
        mu = self.z_mu(mlp_output)
        sigma = torch.exp(self.z_sigma(mlp_output))
        return pi, mu, sigma

# 2.2 实例化mlp
mlp = Mlp(3)

# 3.优化设置
# 3.1 损失函数: 最大似然
def loss_fn(mu, sigma, pi, y):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    # log_prob：Returns the log of the prob. density/mass function evaluated at value.
    # log_prob(value)是计算value在定义的正态分布m中对应的概率的对数!!
    # 这里的y是横轴上的数值，求纵轴上对应钟形曲线的概率loss_1
    loss_1 = torch.exp(m.log_prob(y))
    # 返回输入张量给定维度上每行的和,dim为1会计算每个维度上的和
    # 用pi加权求和，求和的是概率
    # 对应Bishop论文式22！
    loss_2 = torch.sum(loss_1 * pi, dim=1)
    # 把概率变负的log，方便梯度下降
    # loss公式对应Bishop论文式290

    loss = -torch.log(loss_2)
    # 注意这里的loss是一个多维的loss，求mean
    return torch.mean(loss)

# 3.2 优化器
# learning_rate = 0.01
optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

# 4.训练与测试
# 4.1 开始训练
for i in range(epoch):
    print("---------第{}轮训练开始----------".format(i + 1))
    mlp.train()
    for data in train_iter:
        #TODO: 1. target读入 2. output成分布然后sample
        input = data
        pi, mu, sigma = mlp(input)
        output = torch.distributions.Normal(loc=mu, scale=sigma)
        loss = loss_fn(pi, mu, sigma, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        # print
        if total_train_step % 10 == 0:
            # 一般不写loss，而是loss.item()
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 4.3 测试
    mlp.eval()
    total_test_loss = 0
    total_acc = 0
    # 这里可以使用验证集validation set
    with torch.no_grad():
        for data in vali_iter:
            imgs, targets = data
            outputs = mlp(imgs)
            loss = loss_fn(outputs, targets)
            # 累积loss
            total_test_loss += loss.item()
            # 计算准确率的写法见how to train 2.py
            acc = (outputs.argmax(1) == targets).sum()
            total_acc += acc

    print("整体测试集上的Loss:{}".format(total_test_loss))
    # print("整体测试集上的正确率{}".format(total_acc/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_acc",(total_acc/test_data_size))

    # writer记录完再累加次数，tensorboard是从0开始的
    total_test_step = total_test_step + 1

    # 保存每一个epoch的模型结果
    torch.save(mlp, "mlp_model_{}.pth".format(i))

print("SUCCESS\n")
