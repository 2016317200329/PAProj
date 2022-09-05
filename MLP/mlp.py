#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 15:17
# @Author  : Wang Yujia
# @File    : mlp.py
# @Description : 搭一个基本的mlp，用于测试思路
# @TODO: 画图细化一下training的流程：怎么用target data/ NN的规模/ batch size等

from BasicInfo.common_settings import *
import mydataset
import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

################# Global Params ###########################
N_gaussians = 3
# dataset划分
batch_size = 1
train_percentage = 0.8
vali_percentage = 0.2
# 训练优化
learning_rate = 0.01
total_train_step = 0
total_test_step = 0
epoch = 5
# 路径
data_withnp_1_selectedkeys_path = "data/data_withnp_1_selectedkeys.csv"
train_data_file_path = "../data/sim_data"
target_data_path = "../data/data_withnp_1_selected.csv"
data_file_head = "../data/sim_data/data_sampled_"
data_file_tail = ".csv"
################# Global Params ###########################

writer = SummaryWriter("logs-MLP")

# 1. Data读取与划分：70%为训练集,20%为验证集，10%为测试集
# 1.1 读取并划分数据集，保存大小. *目前是读取sample data
data = mydataset.myDataset(train_data_file_path,target_data_path)
# 方案一Dataset写法
# train_data = mydataset.TrainDataset(train_data_file_path)
# target_data = mydataset.TargetDataset(target_data_path)
train_size = int(train_percentage * data.__len__())
vali_size = data.__len__() - train_size
train_data, vali_data= \
    torch.utils.data.random_split(data, [train_size, vali_size])

# 1.2 加载dataloader
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
vali_iter = DataLoader(vali_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

# 2. MLP：一个3+1层的MLP
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
            nn.Linear(25, 225),
            nn.ReLU(inplace=True),
            nn.Linear(225, 9),
            nn.Tanh()
        )
        # π μ σ for MDN
        self.z_pi = nn.Sequential(
            nn.Linear(9, n_gaussians),
            nn.Softmax(dim=1),
        )
        self.z_mu = nn.Linear(9, n_gaussians)
        self.z_sigma = nn.Linear(9, n_gaussians)

    def forward(self, x):
        x = torch.squeeze(x)
        mlp_output = self.mlp_call(x)
        # 输出n_gaussians个高斯的参数
        tmp = self.z_pi(mlp_output)
        pi = torch.mean(tmp,dim=0)
        tmp = self.z_mu(mlp_output)
        mu = torch.mean(tmp,dim=0)
        tmp = torch.exp(self.z_sigma(mlp_output))
        sigma = torch.mean(tmp,dim=0)
        return pi, mu, sigma

# 3.损失函数: 最大似然
# 计算loss的时候，需要用NN【额外】做一下infer，然后和target比较
def loss_fn(mu, sigma, pi, y):
    y = torch.squeeze(y)
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    # log_prob：Returns the log of the prob. density/mass function evaluated at value.
    # log_prob(value)是计算value在定义的正态分布m中对应的概率的对数!!
    loss_1 = torch.exp(m.log_prob(y))
    # 返回输入张量给定维度上每行的和,dim为1会计算每个维度上的和
    # 用pi加权求和，被求和的是概率。TODO：check一下这个值是不是直接可以和预测值作比
    # 对应Bishop论文式22
    loss_2 = torch.sum(loss_1 * pi, dim=1)
    # 把概率变负的log，方便梯度下降
    # 对应Bishop论文式29
    loss = -torch.log(loss_2)
    # 注意这里的loss是一个多维的loss，求mean
    return torch.mean(loss)

# 4.训练与测试
# 4.1 实例化mlp和优化器
mlp = Mlp(N_gaussians)
# learning_rate = 0.01
optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

# 4.2 训练与测试
for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i + 1))
    mlp.train()
    for data in train_iter:
        input, target = data
        pi, mu, sigma = mlp(input)
        output = torch.distributions.Normal(loc=mu, scale=sigma)
        loss = loss_fn(pi, mu, sigma, target)
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
            input, target = data
            pi, mu, sigma = mlp(input)
            loss = loss_fn(pi, mu, sigma, target)
            # 累积loss
            total_test_loss += loss.item()
            # 计算准确率的写法见how to train 2.py
            # TODO: 准确率计算

    print("整体测试集上的Loss:{}".format(total_test_loss))
    # print("整体测试集上的正确率{}".format(total_acc/test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_acc",(total_acc/test_data_size))

    # writer记录完再累加次数，tensorboard是从0开始的
    total_test_step = total_test_step + 1

    # 保存每一个epoch的模型结果
    torch.save(mlp, "mlp_model_{}.pth".format(i))

print("SUCCESS\n")
