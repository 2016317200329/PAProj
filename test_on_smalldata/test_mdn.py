#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 15:11
# @Author  : Wang Yujia
# @File    : test_mdn.py
# @Description :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

n_samples = 1000
# 生成一个1*n_samples大小的tensor
epsilon = torch.randn(n_samples)
# 从[-10,10]生成一个有均匀步长的 1*n_samples大小的tensor
x_data = torch.linspace(-10, 10, n_samples)
y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon

y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)

plt.figure(figsize=(8, 8))
plt.scatter(x_data, y_data, alpha=0.4)
plt.show()

n_input = 1
n_hidden = 20
n_output = 1

x_test = torch.linspace(-15, 15, n_samples).view(-1, 1)

class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        # π μ σ
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h))
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma


model = MDN(n_hidden=20, n_gaussians=5)

optimizer = torch.optim.Adam(model.parameters())

def mdn_loss_fn(y, mu, sigma, pi):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)

for epoch in range(10000):
    pi, mu, sigma = model(x_data)
    loss = mdn_loss_fn(y_data, mu, sigma, pi)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(loss.data.tolist())

pi, mu, sigma = model(x_test)

k = torch.multinomial(pi, 1).view(-1)
y_pred = torch.normal(mu, sigma)[np.arange(n_samples), k].data

plt.figure(figsize=(8, 8))
plt.scatter(x_data, y_data, alpha=0.4)
plt.scatter(x_test, y_pred, alpha=0.4, color='red')
plt.show()

y= tensor.zero