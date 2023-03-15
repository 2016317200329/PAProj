import pandas as pd
import numpy as np
import math
import random
import torch.utils.data
from mydataset import *
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from visdom import Visdom
from graphviz import Digraph
import torchvision
from torchviz import make_dot
from torch.nn.utils.rnn import pad_sequence
import pygraphviz as pgv
import networkx as nx
import matplotlib.pyplot as plt
import geomloss
import time
import optuna
from importlib import reload
from functools import partial

######### Ray Tune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import scipy
from torch.nn import KLDivLoss
from torch.autograd.gradcheck import gradcheck
from config import bcolors

import config
import loss
import plot
# import models

reload(config)  # 必须reload！！
reload(loss)    # 必须reload！！
reload(plot)
# reload(models)

from config import DefaultConfig
# from models import MLP_1_1
from loss import cal_metric
from loss import loss_fn_v2
from loss import loss_fn_wei
from loss import loss_fn_WD
from loss import validate
from plot import plot_conv_weight
from plot import plot_mu_weight
from plot import plot_pi_weight
from plot import plot_sigma_weight
from plot import plot_net

opt = DefaultConfig()

def setup_seed(seed):
    """
    Set seed
    Args:
        seed:
    """
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

# # trainable params
# parameters = dict(
#     lr=[.01,.001],
#     batch_size = [100,1000],
#     shuffle = [True,False]
# )
# #创建可传递给product函数的可迭代列表
# param_values = [v for v in parameters.values()]
# #把各个列表进行组合，得到一系列组合的参数
# #*号是告诉乘积函数把列表中每个值作为参数，而不是把列表本身当做参数来对待
# for lr,batch_size,shuffle in product(*param_values):
#     comment  = f'batch_size={batch_size}lr={lr}shuffle={shuffle}'
#     #这里写你调陈的主程序即可
#     print(comment)


def my_collate_fn(data):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, (TARGET=1)
    target_loss_list = []       # target data for computing loss of NN, (TARGET=5)
    setting_list = []
    metric_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:
        # print("shape:",data[batch][0].shape) #shape: (3, 300)

        # 所有GT model
        data_list.append(torch.tensor(data[batch][0]))

        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor

def my_collate_fn_2(data):
# 这里的data是一个list， list的元素是元组: (self.data, self.label)
# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,
# data[索引到index(batch)][索引到data或者label][索引到channel]

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, (TARGET=1)
    target_loss_list = []       # target data for computing loss of NN, (TARGET=5)
    setting_list = []
    metric_list = []

    data_len = len(data)        # 读进来的data batch的大小

    batch = 0

    while batch < data_len:
        # print("shape:",data[batch][0].shape) #shape: (3, 300)

        # 对于所有用0填充的data来说，最好用一个数字补齐这些0
        # 补齐方法1:
        data[batch][0][0,np.where(data[batch][0][0]==0)]= min(min(data[batch][0][0]),opt.SAFETY)
        data[batch][0][1,np.where(data[batch][0][1]==0)]= min(min(data[batch][0][1]),opt.SAFETY)

        data_list.append(torch.tensor(data[batch][0]))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor

def save_data_idx(dataset,opt):
    """
    因为objective function会被执行很多次，所以这里先保存一下idx，使得所有objective在同一组dataset上进行。
    然后在tuning时，会在shuffle_time组不同的dataset split上进行，用以取平均
    Args:
        dataset:
        opt:
        shuffle_time: The num of list of index to be generated.
    """
    shuffled_indices = []
    # 使用全部的data
    if not opt.arr_flag:
        DATA_len = dataset.__len__()
        shuffled_indices = np.random.permutation(DATA_len)

    # 使用指定的data
    if opt.arr_flag:
        shuffled_indices = np.load(opt.arr_path)
        # DATA_len = len(shuffled_indices)
        np.random.shuffle(shuffled_indices)

    return shuffled_indices

def get_data_idx(shuffled_indices,opt):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        opt:

    Returns:
    """

    DATA_len = len(shuffled_indices)

    train_idx = shuffled_indices[:int(opt.train_pct * DATA_len)]
    tmp = int((opt.train_pct + opt.vali_pct) * DATA_len)
    val_idx = shuffled_indices[int(opt.train_pct * DATA_len):tmp]
    test_idx = shuffled_indices[tmp:]

    return train_idx,val_idx,test_idx

def get_data_loader(dataset, shuffled_indices, opt):
    """
    To get dataloader according to shuffled 'shuffled_indices'
    Args:
        dataset:
        shuffled_indices:
        opt:

    Returns:

    """
    train_idx,val_idx,test_idx = get_data_idx(shuffled_indices,opt)

    train_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(train_idx), collate_fn=my_collate_fn_2)
    val_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx), collate_fn=my_collate_fn_2)
    # 注意test_loader的batch size
    test_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(test_idx), collate_fn=my_collate_fn_2)

    return train_loader,val_loader,test_loader

def get_params(mlp,config,opt):
    """
    Set learning rates for different layers and return params for training
    Args:
        mlp:
        config:
        opt:

    Returns: params

    """
    shape_params = list(map(id, mlp.z_shape.parameters()))
    scale_params = list(map(id, mlp.z_scale.parameters()))
    pi_params = list(map(id, mlp.z_pi.parameters()))

    params_id = shape_params + scale_params + pi_params

    base_params = filter(lambda p: id(p) not in params_id, mlp.parameters())
    params = [{'params': base_params},  # 如果对某个参数不指定学习率，就使用最外层的默认学习率
            {'params': mlp.z_pi.parameters(), 'lr': config['lr_for_pi']},
            {'params': mlp.z_shape.parameters(), 'lr': config['lr_for_shape']},
            {'params': mlp.z_scale.parameters(), 'lr': config['lr_for_scale']}]

    return params

# Not Sequential

# 1569
class Conv_block_1(nn.Module):
    def __init__(self, ch_out=1,kernel_size=9, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.ac_func = nn.Softplus()

        self.conv = nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=self.kernel_size, stride=self.stride, padding=0,bias=True)
        self.BN_aff1 = nn.BatchNorm1d(num_features=1,affine=True)
        self.BN_aff2 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)      # works better

    def forward(self, x):
        # Conv=>BN=>AC
        x = self.conv(x)
        # 方法一：
        # x = torch.squeeze(x,dim=2)
        # x = self.ac_func(self.BN_aff1(x))

        # 方法二：works better
        x = torch.squeeze(x)
        x = self.ac_func(self.BN_aff2(x))

        return x

class Conv_1_1(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=9, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)

        self.layer_pi = Conv_block_1(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_mu = Conv_block_1(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_sigma = Conv_block_1(ch_out=1,kernel_size=kernel_size,stride=stride)

        self.ac_func = nn.Softplus()

        self.z_pi = nn.Sequential(
            nn.Linear(self.ln_in, n_gaussians),
            nn.Softmax(dim=1)
        )

        self.z_mu = nn.Linear(self.ln_in, n_gaussians)
        self.z_sigma = nn.Linear(self.ln_in, n_gaussians)


    def forward(self, x):

        x = self.BN1(x)
        x = torch.unsqueeze(x,dim=1)                     # torch.Size([B, 1, 3, 300])

        x_pi = torch.squeeze(self.layer_pi(x))          # 不加squeeze不行
        x_mu = torch.squeeze(self.layer_mu(x))
        x_sigma = torch.squeeze(self.layer_sigma(x))

        pi = self.z_pi(x_pi)
        mu = self.z_mu(x_mu)
        sigma = self.z_sigma(x_sigma)

        sigma = torch.exp(sigma)
        sigma = torch.clamp(sigma,1e-4)

        # return x1_3
        return pi, mu, sigma

# 1569
class Conv_block_4(nn.Module):
    def __init__(self, ch_out=1,kernel_size=12, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.ac_func = nn.Softplus()

        self.conv = nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=self.kernel_size, stride=self.stride, padding=0,bias=True)
        self.BN_aff1 = nn.BatchNorm1d(num_features=1,affine=True)
        self.BN_aff2 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)      # works better

    def forward(self, x):
        # Conv=>BN=>AC
        x = self.conv(x)
        # 方法一：
        # x = torch.squeeze(x,dim=2)
        # x = self.ac_func(self.BN_aff1(x))

        # 方法二：works better
        x = torch.squeeze(x)
        x = self.ac_func(self.BN_aff2(x))

        return x

class Conv_1_4(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=12, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)

        self.layer_pi = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_scale = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_shape = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)

        self.ac_func = nn.Softplus()

        self.z_pi = nn.Sequential(
            nn.Linear(self.ln_in, n_gaussians),
            nn.Softmax(dim=1)
        )
        self.z_scale = nn.Linear(self.ln_in, n_gaussians)
        self.z_shape = nn.Linear(self.ln_in, n_gaussians)

    def forward(self, x):

        x = self.BN1(x)
        x = torch.unsqueeze(x,dim=1)                     # torch.Size([B, 1, 3, 300])

        x_pi = torch.squeeze(self.layer_pi(x))          # 不加squeeze不行
        x_scale = torch.squeeze(self.layer_scale(x))
        x_shape = torch.squeeze(self.layer_shape(x))

        pi = self.z_pi(x_pi)
        scale = torch.exp(self.z_scale(x_scale))
        scale = torch.clamp(scale,1e-4)
        shape = torch.exp(self.z_shape(x_shape))
        shape = torch.clamp(shape,1e-4)

        return pi,scale,shape



def trainer(train_loader, val_loader, mlp, config, opt, device):
    """
    Main body of a training process. Called by objective function
    Args:
        train_loader:
        val_loader:
        mlp:
        config: params to be tuned
        opt:    params to be held
        device:

    Returns: performance(avg NLL in last 5 epoch) in validation set

    """
    params = get_params(mlp,config,opt)
    optimizer = torch.optim.Adam(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.StepLR_step_size, gamma=opt.StepLR_gamma)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)

    total_train_step = 0
    EPOCH_NUM = 70

    val_metric = []
    GT_metric_1 = 0     # 记录GT-1的metric
    for epoch in range(EPOCH_NUM):
        mlp.train()
        epoch_train_loss = 0

        for batch_id, data in enumerate(train_loader):
            input_data, _, target_loss, setting, _ = data
            # Do the inference
            input_data = input_data.to(device)
            target_loss = target_loss.to(device)
            pi, mu, sigma = mlp(input_data)

            # loss = loss_fn_v2(pi, mu, sigma, target_loss, opt.N_gaussians, opt.TARGET, opt.SAFETY, device)
            loss = loss_fn_wei(pi, mu, sigma, target_loss, opt.N_gaussians, opt.TARGET, opt.SAFETY, device)
            epoch_train_loss += loss.item()
            # draw_loss(total_train_step, loss.item(), win_train_loss_str)
            optimizer.zero_grad()
            loss.backward()   # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.

            total_train_step += 1
        scheduler.step()

        ########### Do validation
        mlp.eval()
        with torch.no_grad():
            total_vali_metric, GT_metric = validate(mlp, val_loader, opt.N_gaussians, opt.MIN_LOSS, device)
            GT_metric_1 = GT_metric[0, 0]
            val_metric.append(total_vali_metric.cpu())

            # draw_metric(epoch, total_vali_metric.cpu(), GT_metric)
            # writer.add_scalars("metric/" + opt.tag_str, {"pred": total_vali_metric.cpu(),
            #                                             "GT-1": GT_metric[0, 0],
            #                                             "GT-2": GT_metric[0, 1]}, epoch)

    # 最后5次training的均值
    avg_val_metric = sum(val_metric[-5:])/5
    # 和GT-1的差值：越大越好
    metric_diff = GT_metric_1 - avg_val_metric
    return metric_diff


class Objective:
    def __init__(self, dataset, opt):
        # Hold this implementation specific arguments as the fields of the class.
        self.dataset = dataset
        self.opt = opt
        # Hold the data split!!
        self.shuffled_indices = save_data_idx(dataset,opt)

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        config = {
            'lr_for_pi': trial.suggest_float('lr_for_pi',1e-4,1e-2),
            # 'kernel_size': trial.suggest_int('kernel_size',6,15),
            # 'stride': trial.suggest_int('stride',3,6)
            'lr_for_shape': trial.suggest_float('lr_for_shape',1e-4,1e-2),
            'lr_for_scale': trial.suggest_float('lr_for_scale',1e-4,1e-2)
        }

        train_loader, val_loader, test_loader = get_data_loader(self.dataset, self.shuffled_indices, self.opt)

        model = Conv_1_4(self.opt.N_gaussians,kernel_size=9,stride=3).to(device)  # put your model and data on the same computation device.
        # Use the same init params
        model_path_init = 'mlp_init.pth'
        model_data = torch.load(model_path_init)
        model.load_state_dict(model_data)

        performance = trainer(train_loader, val_loader, model, config, self.opt, device)

        return performance

######## main #########

setup_seed(6)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#### optuna-dashboard sqlite:///study_name
study_name = 'wei-size-lr'  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
# 里面所有的组合被cover之后会自动stop
# sampler = optuna.samplers.GridSampler(search_space={
#             'lr_for_sigma':[1e-3,5e-3,1e-2],
#             'ac_func3':["rreLU","preLU","leakyrelu"],
#             'ac_str_sigma':["relu","exp","elu"]
#         })

sampler = optuna.samplers.QMCSampler()
pruner = optuna.pruners.NopPruner()
study = optuna.create_study(study_name=study_name,direction='maximize',storage=storage_name,load_if_exists=True,sampler=sampler,pruner=pruner)

dataset = myDataset(opt.train_path, opt.target_path_metric, opt.target_path_loss, opt.data_key_path,
                    opt.NLL_metric_path)

# n_trials表示会尝试n_trials次params的组合方式
# objective会被调用这么多次
study.optimize(Objective(dataset,opt), n_trials=36,show_progress_bar=True)

results = study.trials_dataframe(attrs=("number", "value", "params", "state"))

# 输出最优的超参数组合和性能指标
print('Best hyperparameters: {}'.format(study.best_params))
print('Best performance: {:.6f}'.format(study.best_value))
