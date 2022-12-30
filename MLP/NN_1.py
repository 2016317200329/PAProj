#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 21:52
# @Author  : Wang Yujia
# @File    : NN_1.py
# @Description : 第一版本的NN设计

import pandas as pd
import numpy as np
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tensorboardX import SummaryWriter
from torchsummary import summary
from visdom import Visdom

from set_random_seed import *
from mydataset import *
from my_collate_fn import *

# nums of Gaussian kernels
N_gaussians = 3

# dataset划分
batch_size = 5
train_pct = 0.7
vali_pct = 0.2
test_pct = 0.1

# train and optim.
learning_rate = 0.01
total_train_step = 0
total_test_step = 0
EPOCH_NUM = 5

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training data
train_path = r"../data/train"
# target data
target_path = r"../data/targets"
# data keys
data_key_path = r"../data/target_datakey.csv"

# set the random seed
setup_seed(7)

# 读取dataset
dataset = myDataset(train_path, target_path, data_key_path)

# 乱序抽取dataset
shuffled_indices = np.random.permutation(dataset.__len__())
train_idx = shuffled_indices[:int(train_pct*dataset.__len__())]
tmp = int((train_pct+vali_pct)*dataset.__len__())
val_idx = shuffled_indices[int(train_pct*dataset.__len__()):tmp]
test_idx = shuffled_indices[tmp:]

# dataloader
train_loader = DataLoader(dataset = dataset,batch_size = batch_size, shuffle=False, num_workers=0, drop_last=False, sampler=SubsetRandomSampler(train_idx), collate_fn = my_collate_fn)
val_loader = DataLoader(dataset = dataset,batch_size = batch_size, shuffle=False, num_workers=0, drop_last=False, sampler=SubsetRandomSampler(val_idx),collate_fn = my_collate_fn)
test_loader = DataLoader(dataset = dataset,batch_size = batch_size, shuffle=False, num_workers=0, drop_last=False, sampler=SubsetRandomSampler(test_idx),collate_fn = my_collate_fn)

# define the Net
class MLP(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians) -> None:
        super().__init__()

        self.mlp_call = nn.Sequential(
            nn.BatchNorm2d(num_features=3,affine=False),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,3), stride=(1,3), padding=0,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3,affine=False),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,2), stride=(1,2), padding=0,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3,affine=False),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,5), stride=(1,5), padding=0,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3,affine=False),
            nn.Flatten(),
            nn.Linear(30, 9)
        )
        # π μ σ for MDN
        self.z_pi = nn.Sequential(
            nn.Linear(9, n_gaussians),  # 30个params要learn
            nn.Softmax(dim=1)
        )
        self.z_mu = nn.Linear(9, n_gaussians)
        self.z_sigma = nn.Linear(9, n_gaussians)

    def forward(self, x):
        # 加一个height维度
        x.unsqueeze_(dim=2)
        mlp_output = self.mlp_call(x)
        # print("mlp_output is :", mlp_output)
        # 输出n_gaussians个高斯的参数
        tmp = self.z_pi(mlp_output)
        pi = torch.mean(tmp,dim=0)
        tmp = self.z_mu(mlp_output)
        mu = torch.mean(tmp,dim=0)
        tmp = torch.exp(self.z_sigma(mlp_output))
        # sigma has to be positive
        torch._assert((torch.nonzero(tmp<0, as_tuple=False).shape[0]<=0),"Sigma is less than zero!")
        sigma = torch.mean(tmp,dim=0)
        return pi, mu, sigma
        # return mlp_output


# define the loss
def loss_fn(pi, mu, sigma, target, device, total_train_step):

    # loss2 = ce(logit, label)

    target = torch.squeeze(target)
    duration = torch.flatten(target[:,:,0])
    duration = torch.repeat_interleave(duration.unsqueeze(dim=1), repeats=3, dim=1).to(device)
    m = torch.distributions.Normal(loc=mu.T, scale=sigma.T)
    # log_prob：Returns the log of the prob. density/mass function evaluated at value.
    # log_prob(value)是计算value在定义的正态分布m中对应的概率的对数!!
    loss_1 = torch.exp(m.log_prob(duration))
    # print("Is there any 0 in loss_1: ",torch.any(torch.sum(loss_1==0)).item())
    # 返回输入张量给定维度上每行的和,dim为1会计算每个维度上的和
    # 用pi加权求和，被求和的是概率。对应Bishop论文式22
    loss_2 = torch.sum(loss_1 * pi, dim=1)  # loss_2有0值, 是因为loss_1有0值？

    # draw the distrb.
    x_0 = torch.arange(0,torch.max(duration).item()).to(device)
    x = torch.repeat_interleave(x_0.unsqueeze(dim=1), repeats=3, dim=1)
    y = torch.exp(m.log_prob(x)).to(device)
    y_sum = torch.unsqueeze(torch.sum(pi*y,dim=1),dim=1)   # 维度相等才能cat
    win_str = "total_train_step-"+str(total_train_step)
    viz.line(X = x_0,Y= torch.cat([y,y_sum],dim = 1), env="001", win=win_str,
        opts= dict(title=win_str, legend=['N1', 'N2', 'N3','NNN']))

    # 把概率变负的log，方便梯度下降
    # 对应Bishop论文式29
    # 注意只对非零值进行log的操作
    loss_3 = loss_2[torch.nonzero(loss_2)].squeeze_()
    loss = -torch.log(loss_3)
    # 注意这里的loss是一个多维的loss，求mean
    return torch.mean(loss)


if __name__ == '__main__':

    mlp = MLP(N_gaussians).to(device=device)

    # summary(mlp, (3, 300))
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
    viz = Visdom(env="001")

    for epoch in range(0,1):
        mlp.train()
        for batch_id,data in enumerate(train_loader):
            input, target = data
            print(f"---- {batch_id} batch----")
            input = input.to(device)
            pi, mu, sigma = mlp(input)
            print(f"The [pi,mu,sigma] is : \n")
            print(pi,"\n",mu,"\n",sigma)

            loss = loss_fn(pi, mu, sigma, target, device, total_train_step)
            print("反向传播前：{}，Loss：{}".format(total_train_step, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            print("反向传播后：{}，Loss：{}".format(total_train_step, loss))
            # if total_train_step % 10 == 0:
                # 一般不写loss，而是loss.item()
            # print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)