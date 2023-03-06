#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 09:09
# @Author  : Wang Yujia
# @File    : loss.py
# @Description : 记录各种loss function
import geomloss
import torch
import torch.nn as nn



def cal_metric(Pi, Mu, Sigma, Duration, N_gaussians, vali_setting,MIN_LOSS,device):
    """
    计算 NLL metric of MDN output
    Parameters
    ----------
    Pi
    Mu
    Sigma
    Duration
    N_gaussians
    vali_setting

    Returns
    -------

    """
    NLL = torch.tensor(0.,device=device,requires_grad=True)
    # print("vali_setting: ",vali_setting.shape)   torch.Size([40, 4])
    # [id,bidincrement,bidfee,retail]
    for i in range(len(Pi)):

        # d = vali_setting[i,1]
        # b = vali_setting[i,2]
        # v = vali_setting[i,3]
        # T = int((v-b)/d)

        target = Duration[i,:,0]
        pi = Pi[i,:]
        # m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])
        m = torch.distributions.Laplace(loc=Mu[i,:], scale=Sigma[i,:])

        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        # target_nonzero_2 = torch.repeat_interleave(target_nonzero.unsqueeze(dim=1), repeats=N_gaussians, dim=1).to(device)
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        # loss_1 是高斯分布的单点a处的概率密度value
        # loss_1 = torch.exp(m.log_prob(target_nonzero_2))

        # loss_1 是高斯分布在[a-0.5,a+0.5]上的cdf作为a的“单点分布”
        loss_1 = (m.cdf(target_nonzero_2+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)
        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)

        ## 两种截断的方式都ok：
        ## 方法一：在MDN的prob上设定最小值；
        # loss_3 = torch.clamp(loss_2,min=MIN_LOSS)
        # loss_4 = torch.log(loss_3)
        # # loss_sum = -torch.mean(loss_4) + loss_sum
        # loss_sum = -torch.sum(loss_4) + loss_sum
        assert torch.all(loss_2)>=0,"in metric, loss_2<0"
        ## 方法二：在prob of each sample后加一个safety数 MIN_LOSS
        # loss_2 shape: torch.Size([40, 1])
        loss_3 = -torch.log(loss_2+MIN_LOSS)
        # 除以idx的长度，平均到每个auction
        NLL = torch.sum(loss_3)/len(idx) + NLL

    # 求metric不需要除以batch_size，需要累积起来所有vali data的nll
    return NLL


def loss_fn_v2(Pi, Mu, Sigma, Duration, N_gaussians,TARGET, eps, device):
    """
    计算NLL loss
    Parameters
    ----------
    Pi
    Mu
    Sigma
    Duration
    N_gaussians
    eps: SAFETY

    Returns
    -------

    """
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)

    for i in range(len(Pi)):

        target = Duration[i,:,0]
        pi = Pi[i,:]
        m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])
        # m = torch.distributions.Laplace(loc=Mu[i,:], scale=Sigma[i,:])

        # with torch.no_grad():
        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        # loss_1 是高斯分布的概率密度value
        # loss_1 = torch.exp(m.log_prob(target_nonzero_2))

        loss_1 = (m.cdf(target_nonzero_2+ TARGET/2) - m.cdf(target_nonzero_2 - TARGET/2)).to(device=device)

        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)

        assert torch.all(loss_2) >= 0, "in loss, loss_2<0"

        ## 两种截断的方式都ok：
        ## 方法一：在MDN的prob上设定最小值；
        loss_3 = torch.clamp(loss_2,min=eps)
        loss_4 = torch.log(loss_3)
        loss_sum = -torch.sum(loss_4) + loss_sum

        # ## 方法二：在prob of each sample后加一个safety数 SAFETY
        # loss_3 = -torch.log(loss_2+SAFETY)

        ############ Punishment ############
        #### 统计每个mu距离其他mu的距离，然后sum
        # a = Mu[i,:].reshape(1,-1)
        # b = Pi[i,:].reshape(1,-1)
        # loss_dist = -torch.sum(torch.sqrt((a- a.T)**2))
        # loss_dist = -torch.sum(torch.sqrt((b- b.T)**2))

        # loss_sum = torch.sum(loss_3) + loss_sum

        # print("loss_dist:",loss_dist)
        # print("loss_3:",torch.sum(loss_3))

    # 最后loss求一下平均
    return (loss_sum)/len(Pi)


def log_sum_exp(x):
    """Log-sum-exp trick implementation"""

    # x.shape: torch.Size([35, 3])
    # x_max: torch.Size([35, 1])
    # torch.max()[0]， 只返回最大值的每个数,[1]则返回下标

    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x_min = torch.min(x, dim=1, keepdim=True)[0]
    c = torch.abs(x_min)
    tmp = torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)

    # tmp有很多是1的值，说明exp(0)
    # print("tmp:",tmp)
    # print("x_max: ",x_max)
    with torch.no_grad():
        assert torch.all(x_max<0),"x_max>0!!!"
    return torch.log(tmp) + x_max

def loss_fn_v3(Pi, Mu, Sigma, Target, N_gaussians,device):
    """
    计算loss with log-exp-sum trick. Didn't work well.
    Parameters
    ----------
    Pi
    Mu
    Sigma
    Target
    N_gaussians

    Returns
    -------

    """
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)

    for i in range(len(Pi)):

        target = Target[i,:,0]
        pi = Pi[i,:]
        sigma = Sigma[i,:]
        mu = Mu[i,:]

        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians)

        # loss_1 是高斯分布的概率密度value
        # if(torch.isnan(torch.log(sigma).detach()).any()):
        #     print("ALERT!!!!!!!!!!!!!!!!!: torch.log(sigma)")
        # if(torch.isnan(torch.log(pi).detach()).any()):
        #     print("ALERT!!!!!!!!!!!!!!!!!: torch.log(pi)")
        v1 = .5 * torch.log(2 * torch.tensor(math.pi)) -torch.log(sigma)
        v2 = (target_nonzero_2 - mu)
        # v1 torch.Size([3])
        # v2 torch.Size([44, 3])
        exponent = torch.log(pi) - .5 * torch.log(2 * torch.tensor(math.pi)) -torch.log(sigma) - .5*((target_nonzero_2 - mu)**2)/(sigma**2)

        log_gauss = -log_sum_exp(exponent)

        # 在samples维度上继续sum一次
        loss_sum = torch.sum(log_gauss) + loss_sum

    loss_ts = loss_sum/len(Pi)
    return loss_ts



# cdf loss
###################
def loss_fn_cdf(Pi, Mu, Sigma, Target, N_gaussians,device):
    """
    计算cdf loss
    Parameters
    ----------
    Pi
    Mu
    Sigma
    Target
    N_gaussians

    Returns
    -------

    """
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)

    # for each GMM
    for i in range(len(Pi)):
        target = Target[i,:,0]
        prob_target = Target[i,:,1]
        pi = Pi[i,:]
        m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])

        with torch.no_grad():
            # Drop padded data and Expanded to the same dim
            #### 写法一：work。无原地变换
            idx = torch.nonzero(target)
            target_nonzero = target[idx]
            prob_target_nonzero = torch.squeeze(prob_target[idx])

            target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        # loss_1 是高斯分布的概率密度value
        loss_1 = torch.exp(m.cdf(target_nonzero_2))

        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)

        # loss_3 是cdf的差值的一范数or二范数，表示的是MDN和target之间的总体差异
        loss_3 = torch.norm((loss_2 - prob_target_nonzero), 2)

        # loss_sum = torch.log(loss_3) + loss_sum
        loss_sum = loss_3 + loss_sum

    # Batch平均loss
    return loss_sum/len(Pi)

# # 当input的shape是[50,3]时，输出应该是50个GMM
# # 对这50个GMM看能生成什么output

def loss_fn_CE(Pi, Mu, Sigma, Target, N_gaussians,device):
    """
    计算交叉熵loss
    Parameters
    ----------
    Pi
    Mu
    Sigma
    Target
    N_gaussians

    Returns
    -------

    """
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)

    # for each GMM
    for i in range(len(Pi)):
        target = Target[i,:,0]
        prob_target = Target[i,:,1]
        pi = Pi[i,:]
        m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])

        # Drop padded data and Expanded to the same dim
        #### 写法一：work。无原地变换
        idx = torch.nonzero(target)
        target_nonzero = torch.squeeze(target[idx])
        prob_target_nonzero = torch.squeeze(prob_target[idx])

        target_nonzero_2 = torch.repeat_interleave(target_nonzero.unsqueeze(dim=1), repeats=N_gaussians, dim=1).to(device)

        # loss_1 是高斯分布的概率密度value
        loss_1 = torch.exp(m.log_prob(target_nonzero_2))

        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)

        # loss_4是KL loss
        # loss_4 = -prob_target_nonzero*torch.log(loss_2) + prob_target_nonzero*torch.log(prob_target_nonzero)
        # print("prob_target_nonzero shape:",prob_target_nonzero.shape)
        # print("loss_2 shape:",loss_2.shape)

        loss_4 = -prob_target_nonzero*torch.log(loss_2+MIN_LOSS)+ prob_target_nonzero*torch.log(prob_target_nonzero)
        loss_sum = torch.sum(loss_4)+loss_sum

    loss_ts = loss_sum/len(Pi)
    return loss_ts


p = 1
entreg = .1 # entropy regularization factor for Sinkhorn
factor = 1  # prob的放大系数
# 若以欧式距离为metric，则cost function可以直接用geomloss提供的 Sinkhorn快速解
# debias 可以tune一下
OTLoss = geomloss.SamplesLoss(
    loss='sinkhorn', p=p)    # blur=entreg**(1/p) backend='tensorized',

def loss_fn_WD(Pi, Mu, Sigma, Target, N_gaussians,TARGET,device):
    """
    计算Wasserstein loss(Sinkhorn)
    Parameters
    ----------
    Pi
    Mu
    Sigma
    Target
    N_gaussians

    Returns
    -------

    """
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)

    for i in range(len(Pi)):

        target = torch.unique(Target[i],dim=0)
        n_target = target[:,0]
        non_zero_idx = torch.nonzero(n_target)
        n = n_target[non_zero_idx]
        p_target = target[:,1]
        p = p_target[non_zero_idx]*factor

        # (n,p) pair
        y_target = torch.cat([n,p],dim=1)

        pi = Pi[i,:]
        m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])
        x = torch.repeat_interleave(n, repeats=N_gaussians, dim=1)   # expand dim
        y = (m.cdf(x+TARGET) - m.cdf(x)).to(device=device)
        y_cdf = torch.sum(pi*y,dim=1).unsqueeze(dim=1)*factor
        y_pred = torch.cat([n,y_cdf],dim=1)

        loss_sum = OTLoss(y_pred,y_target) + loss_sum

    return loss_sum/len(Pi)


def validate(mlp,val_loader,N_gaussians, MIN_LOSS, device):
    total_vali_metric = 0   # vali metric
    cnt = 0                 # vali set size
    GT_metric = torch.tensor([0.,0.]).reshape(1,2)
    with torch.no_grad():
        for vali_batch_id, vali_data in enumerate(val_loader):

            vali_input_data, vali_target, _, vali_setting , vali_metric = vali_data
            vali_input_data = vali_input_data.to(device)
            vali_target = vali_target.to(device)
            cnt = cnt + len(vali_input_data)
            vali_pi, vali_mu, vali_sigma = mlp(vali_input_data)

            # Compute the error/ metric
            vali_nll = cal_metric(vali_pi, vali_mu, vali_sigma, vali_target, N_gaussians, vali_setting, MIN_LOSS,device)
            total_vali_metric += vali_nll

            # Sum up NLL of all vali data
            GT_metric += torch.sum(vali_metric,dim=0)

        # Get metric of GT model
        GT_metric = GT_metric/cnt

        total_vali_metric = total_vali_metric/cnt

    return total_vali_metric,GT_metric

