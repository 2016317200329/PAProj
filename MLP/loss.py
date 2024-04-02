#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 09:09
# @Author  : Wang Yujia
# @File    : loss.py
# @Description : 记录各种loss function
import geomloss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(filename="loss_info",filemode="w",level=logging.DEBUG)
logger = logging.getLogger(__name__)

def cal_metric(Pi, Mu, Sigma, Duration, N_gaussians, vali_setting, MIN_LOSS, device):
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

    # [id,bidincrement,bidfee,retail]
    for i in range(len(Pi)):
        target = Duration[i,:,0]
        pi = Pi[i,:]

        # m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])
        # Note: Mu == scale, Sigma = shape
        m = torch.distributions.Weibull(Mu[i,:], Sigma[i,:])

        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)   # sample size of this auction
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
        if torch.isnan(torch.sum(loss_3)):
            print("pi:",pi)
            print("Mu[i,:]:",Mu[i,:])
            print(" Sigma[i,:]:", Sigma[i,:])
            print("loss_1:",loss_1)
            print("loss_2:", loss_2)
            print("loss_3:",loss_3)
        # 除以samples数量，平均到每个记录上
        NLL = torch.sum(loss_3)/len(idx) + NLL

    # 累积起来所有vali data的nll
    return NLL

def cal_metric_KL(Pi, Mu, Sigma, Duration, N_gaussians, vali_setting, MIN_LOSS,device):
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
    NLL_sum = torch.tensor(0.,device=device,requires_grad=False)
    NLL_batchmean = torch.tensor(0.,device=device,requires_grad=False)

    # "mean": 除以samples数量，平均到每个记录上; input进去的值必须log化
    # 'mean' divides the total loss by both the batch size and the support size.
    # 'batchmean' divides only by the batch size, and aligns with the KL div math definition.
    # 'mean' will be changed to behave the same as 'batchmean' in the next major release.

    # input should be a distribution in the log space!!
    kl_loss_sum = nn.KLDivLoss(reduction="sum",log_target=False)
    kl_loss_batchmean = nn.KLDivLoss(reduction="batchmean",log_target=False)

    # [id,bidincrement,bidfee,retail]
    # For testing, len(Pi)==1
    for i in range(len(Pi)):
        Duration_uniq = torch.unique(Duration,dim=1)  # 注意KL-D必须有这一步，去掉N值重复的sample
        target = Duration_uniq[i,:,0]
        p_target = Duration_uniq[i,:,1]
        pi = Pi[i,:]

        # Note: Mu == scale, Sigma = shape
        m = torch.distributions.Weibull(Mu[i,:], Sigma[i,:])

        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)   # sample size of this auction
        target_nonzero = target[idx]
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        # loss_1 是分布在[a-0.5,a+0.5]上的cdf作为a的“单点分布”
        loss_1 = (m.cdf(target_nonzero_2+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)
        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1,keepdim=True)

        assert torch.all(loss_2)>=0,"in metric, loss_2<0"

        # Make sure they were in the same minimum shape!
        p_target.squeeze_()
        loss_2.squeeze_()
        loss_3_sum = kl_loss_sum(torch.log(loss_2+MIN_LOSS), p_target)
        loss_3_batchmean = kl_loss_batchmean(torch.log(loss_2+MIN_LOSS), p_target)

        # 手工算是一样的
        # loss_sum = p_target * (torch.log(p_target) - torch.log(loss_2+MIN_LOSS))  # KL散度计算和torch中的保持一致
        # print(f"loss_sum.shape = {loss_sum.shape}")
        # loss_sum_ = loss_sum.sum()
        # print(f"loss_sum_ = {loss_sum_}")
        # print(f"loss_3_sum = {loss_3_sum}")
        #
        # print(f"loss_3_sum = {loss_3_sum}")
        # print(f"loss_3_batchmean = {loss_3_batchmean}")

        NLL_batchmean = NLL_batchmean + loss_3_batchmean
        NLL_sum = NLL_sum + loss_3_sum

    # 累积起来所有vali data的nll
    return NLL_sum

def cal_metric_q(Pi, Mu, Sigma, Duration, N_gaussians, vali_setting, MIN_LOSS, device, q=0.8):
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
    # [id,bidincrement,bidfee,retail]
    for i in range(len(Pi)):
        # print(Duration.shape)
        target = Duration[i,:,0]
        # print(target.shape)
        pi = Pi[i,:]
        # m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])
        # Note: Mu == scale, Sigma = shape
        m = torch.distributions.Weibull(Mu[i,:], Sigma[i,:])
        # m = torch.distributions.Laplace(loc=Mu[i,:], scale=Sigma[i,:])

        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]

        # Get those >= q-value
        qvalue = torch.quantile(target_nonzero, q=q,keepdim=True)
        target_nonzero_q = target_nonzero[torch.gt(target_nonzero, qvalue)]
        target_nonzero_q = target_nonzero_q[:, None]        # Expand dim

        # Expand by repeating
        # target_nonzero_2 = torch.repeat_interleave(target_nonzero.unsqueeze(dim=1), repeats=N_gaussians, dim=1).to(device)
        target_nonzero_2 = target_nonzero_q.repeat(1,N_gaussians).to(device)

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
        ## 这种方法和GT的metric计算方法一样s
        # loss_2 shape: torch.Size([40, 1])

        loss_3 = -torch.log(loss_2+MIN_LOSS)
        if torch.isnan(torch.sum(loss_3)):
            print("pi:",pi)
            print("Mu[i,:]:",Mu[i,:])
            print(" Sigma[i,:]:", Sigma[i,:])
            print("loss_1:",loss_1)
            print("loss_2:", loss_2)
            print("loss_3:",loss_3)

        # 除以samples数量，平均到每个记录上
        NLL = torch.sum(loss_3)/len(target_nonzero_q) + NLL

    # 累积起来所有vali data的nll
    return NLL

def validate(mlp,data_loader,N_gaussians, MIN_LOSS, device, detail = False,THRESHOLD=0):
    total_metric = 0   # vali metric
    cnt = 0                 # vali set size

    for batch_id, data in enumerate(data_loader):
        input_data, target, _, setting = data
        input_data = input_data.to(device)
        target = target.to(device)

        if target.shape[1] >= THRESHOLD:
            cnt = cnt + len(input_data)  # len(input_data) = 1
        else:   # 不计算少于THRESHOLD的auction
            continue

        pi, mu, sigma = mlp(input_data)

        # Compute the error/ metric
        # Note: Mu == scale, Sigma = shape
        nll = cal_metric(pi, mu, sigma, target, N_gaussians, setting, MIN_LOSS, device)
        total_metric += nll.detach()


    # print(f"THRESHOLD = {THRESHOLD}, Number of auction config is {cnt}")

    total_metric = total_metric.detach().cpu().item()/cnt

    return total_metric

def validate_KL(mlp,data_loader,N_gaussians, MIN_LOSS, device, THRESHOLD=0):
    '''
    KL Divergence as a metric
    Args:
        mlp:
        data_loader:
        N_gaussians:
        MIN_LOSS:
        device:
    Returns:
    '''

    total_metric = 0        # vali metric
    cnt = 0                 # data set size: Number of auction configs that contains samples > THRESHOLD

    for batch_id, data in enumerate(data_loader):
        input_data, target, _, setting= data
        input_data = input_data.to(device)
        target = target.to(device)

        if target.shape[1] >= THRESHOLD:
            cnt = cnt + len(input_data)  # len(input_data) = 1, 因为
        else:   # 不计算少于THRESHOLD的auction
            continue
        pi, mu, sigma = mlp(input_data)

        # Compute the error/ metric
        # Note: Mu == scale, Sigma = shape
        nll = cal_metric_KL(pi, mu, sigma, target, N_gaussians, setting, MIN_LOSS, device)

        total_metric += nll.detach()


    print(f"THRESHOLD = {THRESHOLD}, Number of auction config is {cnt}")
    # Get metric of GT model
    # GT_metric = GT_metric.detach()/cnt

    total_metric = total_metric.detach().cpu().item()/cnt

    # Winning rate
    return total_metric

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
    # logger.info('Start loss')
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

        loss_1 = (m.cdf(target_nonzero_2+ TARGET) - m.cdf(target_nonzero_2 - TARGET)).to(device=device)
        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)
        assert torch.all(loss_2) >= 0, "in loss, loss_2<0"
        # logger.debug(f'loss: {loss_2}')

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

def loss_fn_CE(Pi, Mu, Sigma, Target, N_gaussians, TARGET, eps, device):
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
    kl_loss = nn.KLDivLoss(reduction="sum",log_target=True)     # input进去的值必须log化

    # For each GMM
    for i in range(len(Pi)):
        target = torch.unique(Target[i],dim=0)
        n_target = target[:,0]
        p_target = target[:, 1]

        non_zero_idx = torch.nonzero(n_target)
        n = n_target[non_zero_idx]              # 去掉padding的zero? 现在还有zero吗? 取决于collate方式
        p = p_target[non_zero_idx]

        pi = Pi[i,:]
        m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])
        x = torch.repeat_interleave(n, repeats=N_gaussians, dim=1)                 # expand dim
        loss_1 = (m.cdf(x+ TARGET/2) - m.cdf(x - TARGET/2)).to(device=device)      # loss_1 是高斯分布的概率密度value
        loss_2 = torch.sum(pi*loss_1,dim=1).unsqueeze(dim=1)                       # loss_2 是MDN的概率密度value

        assert torch.all(loss_2) >= 0, "in loss, loss_2<0"
        # 一般p不会是zero
        # 就是怕y_pred出现zero
        loss_3 = kl_loss(torch.log(loss_2),torch.log(p))
        # print("loss_3: ",loss_3)
        print("p.log(): ",p.log())
        print("loss_2.log(): ",loss_2.log())
        loss_sum = loss_sum + loss_3

        # loss_4是KL loss
        # loss_4 = -prob_target_nonzero*torch.log(loss_2) + prob_target_nonzero*torch.log(prob_target_nonzero)
        # print("loss_2 shape:",loss_2.shape)
        #
        # loss_4 = -prob_target_nonzero*torch.log(loss_2+MIN_LOSS)+ prob_target_nonzero*torch.log(prob_target_nonzero)
        # loss_sum = torch.sum(loss_4)+loss_sum

    loss_ts = loss_sum/len(Pi)
    return loss_ts

p = 2
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
        # m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])
        m = torch.distributions.Weibull(Mu[i,:],Sigma[i,:])

        x = torch.repeat_interleave(n, repeats=N_gaussians, dim=1)   # expand dim
        y = (m.cdf(x+TARGET) - m.cdf(x)).to(device=device)
        y_cdf = torch.sum(pi*y,dim=1).unsqueeze(dim=1)*factor
        y_pred = torch.cat([n,y_cdf],dim=1)

        loss_sum = OTLoss(y_pred,y_target) + loss_sum

    return loss_sum/len(Pi)
def save_grad(name):
    def hook(grad):
        print(f"name={name}, grad={grad}")
    return hook


def loss_fn_wei(Pi, Mu, Sigma, Duration, N_gaussians, TARGET, eps, device):
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
    # Pi.register_hook(save_grad('Pi'))
    # Mu.register_hook(save_grad('Mu'))
    # Sigma.register_hook(save_grad('Sigma'))
    for i in range(len(Pi)):

        target = Duration[i,:,0]
        pi = Pi[i,:]
        # p = Duration[i,:,1]
        m = torch.distributions.Weibull(Mu[i,:],Sigma[i,:])

        # with torch.no_grad():
        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        # p_nonzero = torch.flatten(p[idx])
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        # loss_1 是高斯分布的概率密度value
        # loss_1 = torch.exp(m.log_prob(target_nonzero_2))

        # print("target_nonzero_2: ",target_nonzero_2)

        # loss_1 = (m.cdf(target_nonzero_2+ TARGET/2) - m.cdf(torch.clamp((target_nonzero_2 - TARGET/2),target_nonzero_2))).to(device=device)
        loss_1 = (m.cdf(target_nonzero_2 + TARGET+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)
        # logger.debug(f'loss 1: {loss_1}')
        # print("loss_1", loss_1)
        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)
        assert torch.all(loss_2) >= 0, "in loss, loss_2<0"


        ## 两种截断的方式都ok：
        ## 方法一：在MDN的prob上设定最小值；
        loss_3 = torch.clamp(loss_2,min=eps)
        loss_4 = torch.log(loss_3)


        ##### TV
        # loss_5 = torch.nn.functional.mse_loss(loss_2, p_nonzero)
        # print(loss_5.detach().cpu())
        loss_sum = -torch.sum(loss_4) + loss_sum # + loss_5

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

def loss_fn_wei_2(Pi, Mu, Sigma, Duration, N_gaussians, TARGET, eps, device):
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
    # Pi.register_hook(save_grad('Pi'))
    # Mu.register_hook(save_grad('Mu'))
    # Sigma.register_hook(save_grad('Sigma'))
    for i in range(len(Pi)):

        target = Duration[i,:,0]
        pi = Pi[i,:]
        # p = Duration[i,:,1]
        m = torch.distributions.Weibull(Mu[i,:],Sigma[i,:])

        # with torch.no_grad():
        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        # p_nonzero = torch.flatten(p[idx])
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        # loss_1 是高斯分布的概率密度value
        # loss_1 = torch.exp(m.log_prob(target_nonzero_2))

        # print("target_nonzero_2: ",target_nonzero_2)

        # loss_1 = (m.cdf(target_nonzero_2+ TARGET/2) - m.cdf(torch.clamp((target_nonzero_2 - TARGET/2),target_nonzero_2))).to(device=device)
        loss_1 = (m.cdf(target_nonzero_2 + TARGET+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)
        # logger.debug(f'loss 1: {loss_1}')
        # print("loss_1", loss_1)
        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1**2 * pi, dim=1)
        assert torch.all(loss_2) >= 0, "in loss, loss_2<0"


        ## 两种截断的方式都ok：
        ## 方法一：在MDN的prob上设定最小值；
        loss_3 = torch.clamp(loss_2,min=eps)
        loss_4 = torch.log(loss_3)


        ##### TV
        # loss_5 = torch.nn.functional.mse_loss(loss_2, p_nonzero)
        # print(loss_5.detach().cpu())
        loss_sum = -torch.sum(loss_4) + loss_sum # + loss_5

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


def loss_fn_mse(Pi, Mu, Sigma, Duration, N_gaussians, TARGET, eps, device):
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)

    for i in range(len(Pi)):
        target = Duration[i, :, 0]
        pi = Pi[i, :]
        p = Duration[i, :, 1]
        m = torch.distributions.Weibull(Mu[i, :], Sigma[i, :])

        # with torch.no_grad():
        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        p_nonzero = torch.flatten(p[idx])
        target_nonzero_2 = target_nonzero.repeat(1, N_gaussians).to(device)

        pi = Pi[i,:]
        # m = torch.distributions.Normal(loc=Mu[i,:], scale=Sigma[i,:])
        m = torch.distributions.Weibull(Mu[i,:],Sigma[i,:])

        loss_1 = (m.cdf(target_nonzero_2 + TARGET+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)

        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)

        # loss_3 = torch.mean( torch.log((loss_2-p_nonzero)**2) )
        loss_3 = torch.nn.functional.mse_loss(torch.log(loss_2), torch.log(p_nonzero),reduction="mean")

        loss_sum = torch.sum(loss_3)

    return loss_sum/len(Pi)


def loss_fn_wei_mask(Pi, Mu, Sigma, Duration, N_gaussians, TARGET, Val_idx, Test_idx, eps, device):
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

        m = torch.distributions.Weibull(Mu[i,:],Sigma[i,:])

        # with torch.no_grad():
        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]

        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        # loss_1 是高斯分布的概率密度value
        # loss_1 = torch.exp(m.log_prob(target_nonzero_2))

        # print("target_nonzero_2: ",target_nonzero_2)

        loss_1 = (m.cdf(target_nonzero_2 + TARGET+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)


        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)
        assert torch.all(loss_2) >= 0, "in loss, loss_2<0"

        ## 两种截断的方式都ok：
        ## 方法一：在MDN的prob上设定最小值；
        loss_3 = torch.clamp(loss_2,min=eps)
        loss_4 = torch.log(loss_3)

        # Set the mask index
        target_idx = torch.ones_like(idx)
        val_idx = torch.nonzero(Val_idx[i,:,:])
        test_idx = torch.nonzero(Test_idx[i,:,:])

        target_idx[val_idx] = 0
        target_idx[test_idx] = 0
        print("val_idx:", val_idx)
        print("target_idx:",target_idx.shape)

        loss_4_mask = loss_4.view(-1,1) * target_idx
        loss_sum = -torch.sum(loss_4) + loss_sum

    # 最后loss求一下平均
    return (loss_sum)/len(Pi)

def loss_fn_wei_crps(Pi, Scale, Shape, Duration, N_gaussians, TARGET, eps, device):
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
    # Pi.register_hook(save_grad('Pi'))
    # Mu.register_hook(save_grad('Mu'))
    # Sigma.register_hook(save_grad('Sigma'))
    for i in range(len(Pi)):

        target = Duration[i,:,0]
        pi = Pi[i,:]
        m = torch.distributions.Weibull(Scale[i,:],Shape[i,:])

        # with torch.no_grad():
        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        # p_nonzero = torch.flatten(p[idx])
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        # loss_1 是高斯分布的概率密度value
        # loss_1 = torch.exp(m.log_prob(target_nonzero_2))

        # print("target_nonzero_2: ",target_nonzero_2)

        # loss_1 = (m.cdf(target_nonzero_2+ TARGET/2) - m.cdf(torch.clamp((target_nonzero_2 - TARGET/2),target_nonzero_2))).to(device=device)
        loss_1 = (m.cdf(target_nonzero_2 + TARGET+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)
        (m.cdf())
        # logger.debug(f'loss 1: {loss_1}')
        # print("loss_1", loss_1)
        # loss_2 是MDN的概率密度value
        loss_2 = torch.sum(loss_1 * pi, dim=1)
        assert torch.all(loss_2) >= 0, "in loss, loss_2<0"


        ## 两种截断的方式都ok：
        ## 方法一：在MDN的prob上设定最小值；
        loss_3 = torch.clamp(loss_2,min=eps)
        loss_4 = torch.log(loss_3)


        ##### TV
        # loss_5 = torch.nn.functional.mse_loss(loss_2, p_nonzero)
        # print(loss_5.detach().cpu())
        loss_sum = -torch.sum(loss_4) + loss_sum # + loss_5

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


# 衡量upper tail的performance, 默认衡量后面20%的data
def validate_q(mlp,data_loader,N_gaussians, MIN_LOSS, device, detail = False, q = 0.8):
    total_metric = 0   # vali metric
    cnt = 0                 # data set size
    odds_cnt_1 = 0          # times that NN behaves better than GT-1
    odds_cnt_2_common = 0          # times that NN behaves better than GT-2(common)
    odds_cnt_2_SA = 0          # times that NN behaves better than GT-2(SA)
    odds = [-1,-1,-1]            # winning rate of NN
    GT_metric = torch.tensor([0.,0.,0.,0.]).reshape(1,-1)

    for batch_id, data in enumerate(data_loader):
        input_data, target, _, setting , metric = data
        input_data = input_data.to(device)
        target = target.to(device)
        cnt = cnt + len(input_data)

        pi, mu, sigma = mlp(input_data)

        # Compute the error/ metric
        # Note: Mu == scale, Sigma = shape
        nll = cal_metric_q(pi, mu, sigma, target, N_gaussians, setting, MIN_LOSS, device, q)
        total_metric += nll.detach()

        # Sum up NLL of all vali data
        GT_metric += torch.sum(metric,dim=0)

        if detail:                                  # 优胜率：在多少个setting上NN可以表现更好
            if nll.detach() < metric.detach()[0,0]:
                odds_cnt_1 += 1
            if nll.detach() < metric.detach()[0,1]:
                odds_cnt_2_common += 1
            if nll.detach() < metric.detach()[0,2]:
                odds_cnt_2_SA += 1

    # Get metric of GT model
    GT_metric = GT_metric/cnt

    total_metric = total_metric/cnt

    # Winning rate
    odds = [odds_cnt_1 / cnt, odds_cnt_2_common / cnt, odds_cnt_2_SA / cnt]


    return total_metric, GT_metric, odds

def rmse_revenue(mlp,data_loader, er_loader, N_gaussians, device):
    cnt = len(data_loader)     # Total test amount
    rmse = 0.
    mse = 0.

    # For each config.
    for batch_id, (data, target_er) in enumerate(zip(data_loader, er_loader)):
        input_data, target, _, setting , metric = data
        input_data = input_data.to(device)

        pi, mu, sigma = mlp(input_data)

        er_duration = 0.    # Exp of duration.

        # For each distribution
        for j in range(N_gaussians):
            pi_j = pi[0][j]
            scale_j = mu[0][j]
            shape_j = sigma[0][j]

            er_duration = er_duration + pi_j * scale_j * torch.exp(torch.lgamma(1 + 1 / shape_j))

        # er_i or target_er_i = expected revenue / retail
        er_i = ( setting[0][1] + setting[0][3]) * er_duration/setting[0][2]

        target_er_i = target_er[0][-1]

        mse = mse + torch.pow(target_er_i-er_i,2)

    rmse = torch.sqrt(mse/cnt)

    return rmse



