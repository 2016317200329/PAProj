#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:47
# @Author  : Wang Yujia
# @File    : SA_for_PT_funcs_delta_eq1.py
# @Description : SA_for_PT_model_delta_eq1.ipynb需要的functions。注意是delta=1的情况下的，有所简化

import numpy as np
import torch
import scipy.stats

from torch.cuda.amp import GradScaler,autocast
# torch.set_default_dtype(torch.float64)  # 设为之后，在tmp>0下不会发生NaN的问题，但是推理速度极慢

def save_grad(name):
    def hook(grad):
        print("*******")
        print(f"name={name}, grad={grad}")
    return hook



# prob. weighting func。根据Eq(5)
def OMEGA(p):
    return p


def C(t,b):
    return 0.2*t*b

# the valuation function
def f(x, alpha):
    return (1-np.exp(-alpha*x))

def f_2(x,alpha):
    # y = torch.min(-alpha*x,torch.tensor(85.))       # torch.exp(x), x<85
    return np.exp(-alpha*x)
def f_Equi(t,v,d,b,alpha,labda):

    tmp = v - d * (t - 1) - C(t - 2, b) - b
    if (tmp>0):

        root = (labda*f(C(t-2,b),alpha) + f(tmp,alpha)) / (labda*f(C(t-2,b)+b,alpha) + f(tmp,alpha))

        # if np.isnan(root):
        #     root = 1.
    else:

        # root = (f(C(t-1,b),alpha) - f(-tmp,alpha)) / (f(C(t-1,b)+b,alpha) - f(-tmp,alpha))
        # 化简：此时和C无关，sunk cost不影响
        # 分母有可能为0

        root = (1- f_2(-(v-(t-1)*d-b),alpha)) / (f_2(b,alpha) - f_2(-(v-(t-1)*d-b),alpha))
        if np.isnan(root):      # 240331 set
            root = 1.

    # AssertionError: Before clip, U[t] < 0 , tmp = -0.07000000000025464, U[t] = nan, t,V,d,b,alpha,labda = (19998, 2799.99, 0.02, 0.6, 0.3000000119209289,0.9323378801345824)
    # assert root >= 0, f"Before clip, U[t] < 0 , tmp = {tmp}, U[t] = {root}, t,v,d,b,alpha,labda = {t,v,d,b,alpha,labda}"
    # assert root <= 1, f"Before clip, U[t] > 1 , tmp = {tmp}, U[t] = {root}, t,v,d,b,alpha,labda = {t,v,d,b,alpha,labda}"

    return root


def get_LEN_T(v,b,d,duration_max):
    """
    计算LEN和T
    Args:
        v:
        b:
        d:
        T_i: 需要计算的最大的duration：合理 而且 不计算多余的

    Returns:
            LEN: U length aka. max duration GT should calculate
            T: duration limitation

    """

    T = 0
    LEN = 0

    if d == 0:      # fixed-price
        T = np.inf
        LEN = int(duration_max)         # 最多计算target data需要的duration
    else:           # asc-price
        T = np.floor((v - b) / d)
        LEN = int(T)                    # 把能计算的都算了

    return LEN, T


def U_GT2(t,v, d, b,alpha = -0.0135, labda = 3.3124,eps = 1e-10):
    return np.clip(f_Equi(t, v, d, b, alpha, labda),eps,1-eps)  # 注意设置LEN之内，eps < U < 1-eps

def get_U_GT2(LEN,v, d, b, alpha = -0.0135, labda = 3.3124,eps = 1e-10):
    """
    计算LEN之内的U. 注意：eps < U[t] < 1-eps, when 1 <= t <= LEN, and U[LEN+1] = 0
    Args:
        LEN: 可计算的最大的auction duration
        v:
        d:
        b:
        alpha:
        labda:
        eps = 1e-10,表示U最小为1e-10，方便计算避免log0

    Returns:
        U: prob vector and U[t] represents the prob that someone bids at period 't'

    """

    U = [0] * (LEN + 2)  # U: the prob. that someone offers a bid in t_th round
    U[0], U[1] = 1., 1.  # u[0]用不到,u[1]=1保证auction至少1轮

    # U1 = [0] * (LEN + 2)  # U: the prob. that someone offers a bid in t_th round
    # U1[0], U1[1] = 1., 1.  # u[0]用不到,u[1]=1保证auction至少1轮


    # 使用 vectorize
    t = np.arange(2,LEN+1)
    U_GT2_vec = np.vectorize(U_GT2)
    U[2:-1] = U_GT2_vec(t, v, d, b,alpha,labda,eps)

    # # 不使用 vectorize
    # t = 2
    # while t <=LEN:      # 不超过理论上限，可计算
    #     U1[t] = np.clip(f_Equi(t, v, d, b, alpha, labda),eps,1-eps)  # 注意设置LEN之内，eps < U < 1-eps
    #     t += 1

    # 超过理论上限T_i，不可计算，严格置为0
    # U[LEN+1]表示auction在t = LEN+1时 no bid will occur
    U[-1] = 0
    # U1[t] = 0

    # assert U == U1, "U != U1"
    return U


def U_GT1(t, v, d, b,eps = 1e-10):
    return np.clip(1 - b / (v - (t - 1) * d), eps, 1 - eps)  # 注意设置LEN之内，eps < U < 1-eps


def get_U_GT1(LEN,v, d, b,eps = 1e-10):
    U = [0] * (LEN + 2)  # U: the prob. that someone offers a bid in t_th round
    U[0], U[1] = 1., 1.  # u[0]用不到,u[1]=1保证auction至少1轮

    # U1 = [0] * (LEN + 2)  # U: the prob. that someone offers a bid in t_th round
    # U1[0], U1[1] = 1., 1.  # u[0]用不到,u[1]=1保证auction至少1轮

    # 使用 vectorize
    t = np.arange(2,LEN+1)
    U_GT1_vec = np.vectorize(U_GT1)
    U[2:-1] = U_GT1_vec(t, v, d, b,eps)

    # # 不使用 vectorize
    # t = 2
    # while t <=LEN:      # 不超过理论上限，可计算
    #     U1[t] = np.clip(1- b/(v-(t-1)*d),eps,1-eps)  # 注意设置LEN之内，eps < U < 1-eps
    #     t += 1

    # 超过理论上限T_i，不可计算，严格置为0
    # U[LEN+1]表示auction在t = LEN+1时 no bid will occur
    U[-1] = 0
    # U1[t] = 0
    # assert U == U1, "U != U1"
    return U
def get_nll_loss(T_target, U, LEN, eps = 1e-10):
    """
    计算NLL loss for GT-2 inference, 注意可能出现两个log 0的问题 都已排除.
    另外计算loss和计算metric采取的是不同的处理方式
    Args:
        T_target:
        U:
        LEN:
        eps: 1e-10

    Returns:

    """
    nll = 0.0

    # For each observed duration T_target[idx]
    for idx in range(0,len(T_target)):
        # Only take into consideration situations: Recorded in U (可以计算的)
        if T_target[idx] <= LEN :
            # # U_t=1表示拍卖在第t轮一定有人bid，此时拍卖不可能结束在第t-1轮
            # if U[T_target[idx]+1] == 1:
                # nll += ( np.sum( np.log( U[1:(T_target[idx])+1] ) ) +
                #     np.where((np.isinf( np.log(1-U[T_target[idx]+1]) )), 0., np.log(1-U[T_target[idx]+1])) )
            nll += ( np.sum( np.log( U[1:(T_target[idx])+1] ) ) + np.log(1-U[T_target[idx]+1]))
        else:   # 不可以计算，相当于整个U vector == eps
            nll += T_target[idx] * np.log(eps)
    return float(-nll)

def get_nll_meric(T_target, U, LEN, TARGET = 1,eps = 1e-30,q=1):
    """

    Args:
        T_target:
        U:
        LEN:
        TARGET:
        eps:

    Returns:
    NLL metric: 正值,并且已经除以sample数量
    """

    nll = 0.
    # Solve for P with length of LEN
    P = np.array([0.0] * (LEN + 1))
    P[0] = 0.0
    tmp = np.array([0.0] * (LEN + 3))  # tmp的大小不需要太精确
    tmp[0] = 1.0
    # 注意：P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    for t in range(1, len(P)):
        tmp[t] = tmp[t - 1] * U[t]  # tmp[t]存了U从1到(t)的连乘积
        P[t] = (1 - U[t + 1]) * tmp[t]

    # DO NOT DELETE the P[0] here to keep with actual meaning,和实际意义保持一致
    # P = np.delete(P,[0],axis=0)

    # # To make sure sum=1
    # p[T-1] = max(1-sum(p[0:T-1]),0)

    # According to the target data, compute the NLL value

    P_dict = {}  # 把P转化成dict，方便计算
    # Sum prob in every interval of TARGET
    # 注意P_dict从1开始计数，和实际意义保持一致
    # print(P)
    for i in range(1, LEN + 1, TARGET):  # 当TARGET=1时，其实没有什么影响
        j = min(LEN, i + TARGET)
        P_dict[i] = np.sum(P[i:j])

    # nll_i = sum(logP1+logP2+...)
    # Sum up all prob if GT gives one
    if q==1:
        for i in range(0, len(T_target)):
            if T_target[i] in P_dict:  # 如果target_i可以计算，
                nll += -np.log(P_dict[T_target[i]] + eps)
            else:  # 不可以计算，prob=0
                nll += -np.log(0. + eps)
        # 返回值是正值
        nll = nll / len(T_target)

    elif q<1:       # 只统计percentile后面的data
        qvalue = np.quantile(T_target,q=q)
        T_target_Q = np.array(T_target)[T_target>qvalue]
        for i in range(0, len(T_target_Q)):
            if T_target_Q[i] in P_dict:  # 如果target_i可以计算，
                nll += -np.log(P_dict[T_target_Q[i]] + eps)
            else:  # 不可以计算，prob=0
                nll += -np.log(0. + eps)
        nll = nll / len(T_target_Q)

    for i in range(0, len(T_target)):
        if T_target[i] in P_dict:  # 如果target_i可以计算，
            nll += -np.log(P_dict[T_target[i]] + eps)
        else:  # 不可以计算，prob=0
            nll += -np.log(0. + eps)
    # 返回值是正值
    nll = nll / len(T_target)

    return nll

def get_P(U,LEN):
    """
    返回P vector, 且P[0]表示duration=1，目前用于画图
    Args:
        U:
        LEN:
        v:
        d:
        b:
    """
    # Solve for P with length of LEN
    P = np.array([0.0]*(LEN+1))
    P[0] = 0.0                                            # auction duration==0的概率=0
    tmp = np.array([0.0]*(LEN+3))                         # tmp的大小不需要太精确
    tmp[0] = 1.0

    # 注意：P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    for t in range(1,len(P)):
        tmp[t] = tmp[t-1]*U[t]                          # tmp[t]存了U从1到(t)的连乘积
        P[t] = (1-U[t+1])*tmp[t]

    # 去掉P[0]
    return P[1:]


############# Tensor #############:

def f_ts(x, alpha, device):
    # -alpha*x不能超过85，否则overflow
    y1 = torch.clamp(-alpha*x,max=torch.tensor(85.,device=device)).to(device)       # torch.exp(x), x<85
    y2 = (1-torch.exp(y1)).to(device)
    # 'register_hook' won’t only in two cases:
    # It was registered on a Tensor for which the gradients was never computed.
    # The register_hook function is some part of your code that did not run during the forward.
    y1.register_hook(save_grad('y1'))
    y2.register_hook(save_grad('y2'))


    return y2

def f_2_ts(x, alpha,device):
    # -alpha*x不能超过85
    # y1 = torch.min(-alpha*x,torch.tensor(85.)).to(device)       # torch.exp(x), x<85
    y1 = (-alpha*x).to(device)       # torch.exp(x), x<85
    y2 = (torch.exp(y1)).to(device)

    y1.register_hook(save_grad('y1'))   ###??????
    y2.register_hook(save_grad('y2'))

    return y2

# root = U_{t-1}
def f_Equi_ts(t, T, v, d, b, alpha, labda, device):

    tmp = (v-d*(t-1)-C((t-2),b) - b).to(device)

    if (tmp>0):

        # root = (labda*f_ts(C(t-2,b),alpha,device) + f_ts(((v-d*(t-1)-C((t-2),b) - b)),alpha,device)) / (labda*f_ts(C(t-2,b)+b,alpha,device) + f_ts(((v-d*(t-1)-C((t-2),b) - b)),alpha,device))
        root = alpha*labda

    else:

        # root = (1- f_2_ts(-(v-(t-1)*d-b),alpha,device)) / (f_2_ts(b,alpha,device) - f_2_ts(-(v-(t-1)*d-b),alpha,device))
        root = -alpha * labda
    # root.register_hook(save_grad('root'))
    # alpha.register_hook(save_grad('alpha'))
    # labda.register_hook(save_grad('labda'))
    # print("root:", root)

    return root

# if __name__ == '__main__':
#
#     # t, v, d, b, alpha, labda = (17480, 1299.99, 0.01, 0.75, 0.025, 3.72)
#     # t, v, d, b, alpha, labda = (263, 39.95, 0.15, 0.75, -0.0631152460302136, 5.710295596229588)
#     t, v, d, b, alpha, labda = (4988, 898, 0.06, 0.6, 0.3, 0.01)
#
#     # t = torch.tensor(4623.)
#     #
#     # v = torch.tensor(799.9900)
#     # d = torch.tensor(0.0600)
#     # b = torch.tensor(0.6000)
#     # T = np.floor((v - b) / d)
#     # alpha = torch.tensor([0.2555], device='cuda:0',requires_grad=True)
#     # labda  = torch.tensor([0.0100], device='cuda:0',requires_grad=True)
#     # t = 260        # 8123开始m<0
#     # root = f_Equi_ts(t, T, v, d, b, alpha, labda, 'cuda:0')
#
#     # root = (1- f_2(-(v-(t-1)*d-b),alpha)) / (f_2(b,alpha) - f_2(-(v-(t-1)*d-b),alpha))
#     root = (f_2(b,alpha) - f_2(-(v-(t-1)*d-b),alpha))
#     print("root:",root)
#
#     # while t <= 265:
#     #     # root = f_Equi(t,v,d,b,alpha,labda)
#     #     tmp = v - d * (t - 1) - C(t - 2, b) - b
#     #
#     #     print(f"m:{tmp}")
#     #
#     #     # # 精度问题
#     #     # K = f(C(t-1,b)+b,alpha)
#     #     # print(f"K = {K}")
#     #     # B = f(-tmp,alpha)
#     #     # print(f"B = {B}")
#     #
#     #     root = (f(C(t-1,b),alpha) - f(-tmp,alpha)) / (f(C(t-1,b)+b,alpha) - f(-tmp,alpha))
#     #     # root_42 = (f(C(t-2,b),alpha) - f(-tmp,alpha)) / (f(C(t-2,b)+b,alpha) - f(-tmp,alpha))
#     #     root_43 = (1- f_2(-(v-(t-1)*d-b),alpha)) / (f_2(b,alpha) - f_2(-(v-(t-1)*d-b),alpha))
#     #
#     #     print(f"t={t}: root(4.2)={root}")
#     #     print(f"t={t}: root(4.3)={root_43}")
#     #     # print(f"A = {f(C(t-2,b)+b,alpha)}")
#     #     # print(f"B = {f(-tmp,alpha)}")
#     #     print("-------------")
#     #     t += 1



def get_KL_meric(T_target, target_p, U, LEN, TARGET = 1, eps = 1e-30):
    """

    Args:
        target_p:
        U:
        LEN:
        TARGET:
        eps:

    Returns:
    KL metric
    """

    kl = 0.
    # Solve for P with length of LEN
    P = np.array([0.0] * (LEN + 1))
    P[0] = 0.0
    tmp = np.array([0.0] * (LEN + 3))  # tmp的大小不需要太精确
    tmp[0] = 1.0
    # 注意：P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    for t in range(1, len(P)):
        tmp[t] = tmp[t - 1] * U[t]  # tmp[t]存了U从1到(t)的连乘积
        P[t] = (1 - U[t + 1]) * tmp[t]

    # DO NOT DELETE the P[0] here to keep with actual meaning,和实际意义保持一致
    # P = np.delete(P,[0],axis=0)

    # # To make sure sum=1
    # p[T-1] = max(1-sum(p[0:T-1]),0)

    # According to the target data, compute the NLL value

    P_dict = {}  # 把P转化成dict，方便计算
    # Sum prob in every interval of TARGET
    # 注意P_dict从1开始计数，和实际意义保持一致
    # print(P)
    for i in range(1, LEN + 1, TARGET):  # 当TARGET=1时，其实没有什么影响
        j = min(LEN, i + TARGET)
        P_dict[i] = np.sum(P[i:j])

    T_target = [int(i) for i in T_target]

    # Sum up all prob if GT gives one
    for i in range(0, len(T_target)):
        if T_target[i] in P_dict:  # 如果target_i可以计算， #
            kl += target_p[i] * (np.log(target_p[i]) - np.log(P_dict[T_target[i]] + eps))    # KL散度计算和torch中的保持一致
        else:  # 不可以计算，prob=0
            kl += target_p[i] * (np.log(target_p[i]) - np.log(0. + eps))           # KL散度计算和torch中的保持一致

    else:       # 只统计percentile后面的data
        assert "No such functions"


    return kl

