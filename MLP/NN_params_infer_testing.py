# Read in .pth and then calculate metrics. for testing set?

import pandas as pd
import numpy as np
import os
import random
from models import MLP_2_1
from models import MLP_GT3_2
import torch.utils.data
import mydataset_GT2
from functorch import vmap
from mydataset_GT2 import myDataset
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary
from visdom import Visdom
import torchvision
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
import time
# from GT_model.GT_2.SA_for_PT_funcs_delta_eq1 import *
from GT_model.GT_2 import SA_for_PT_funcs_delta_eq1
from importlib import reload
from utils import get_data_loader

from torch.cuda.amp import GradScaler,autocast
import pickle

scaler = GradScaler(backoff_factor = 0.1)
import scipy
from torch.autograd.gradcheck import gradcheck
from Config import config_GT2
from Config import config_GT3

def get_model_opt(MODEL_NAME):
    if MODEL_NAME == 'InferNet_GT2':
        return {
            'model' : MLP_2_1(),
            'opt': config_GT2.DefaultConfig()
        }
    elif MODEL_NAME == 'InferNet_GT3':
        return {
            'model': MLP_GT3_2(),
            'opt': config_GT3.DefaultConfig()
        }
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

######################### ↓↓↓ Carefully Check ↓↓↓ ####################

from NN_params_infer_GT2 import validate_params
MODEL_NAME = "InferNet_GT2"

# from NN_params_infer_GT3 import validate_params
# MODEL_NAME = "InferNet_GT3"

opt = get_model_opt(MODEL_NAME)['opt']

seed = 626
######################### ↑↑↑ Carefully Check ↑↑↑ ####################


import loss
import plot
import my_collate_fn
from my_collate_fn import my_collate_fn_GT2

from utils import setup_seed
from utils import save_data_idx
from utils import get_data_idx
from utils import get_InferNet_save_path
from utils import load_checkpoint




reload(config_GT2)  # 必须reload！！
reload(config_GT3)  # 必须reload！！
reload(loss)        # 必须reload！！
reload(plot)
reload(mydataset_GT2)
reload(SA_for_PT_funcs_delta_eq1)
reload(my_collate_fn)

total_train_step = 0
total_test_step = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

bound_alpha = torch.tensor([-0.3,0.3],device=device)
bound_labda = torch.tensor([0.01,18],device=device)


if __name__ == '__main__':

    setup_seed(seed)
    print(f"========== seed = {seed} ==========")
    print(f"========== seed = {seed} ==========")
    print(f"========== seed = {seed} ==========")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 数据设置
    dataset = myDataset(opt.train_path, opt.target_path_metric, opt.target_path_loss, opt.params_opitim_path,
                        opt.data_key_path, opt.NLL_metric_path)
    shuffled_indices = save_data_idx(dataset, opt.arr_flag)
    train_idx, val_idx, test_idx = get_data_idx(shuffled_indices,opt.train_pct, opt.vali_pct,opt.SET_VAL)
    train_loader,val_loader,test_loader = get_data_loader(dataset, opt.batch_size, train_idx, val_idx, test_idx, my_collate_fn_GT2)
    print(f"test_loader size: {test_loader.__len__() / 1276}")

    model = get_model_opt(MODEL_NAME)['model'].to(device)


    ########### Load .pth and do testing
    model.eval()
    with torch.no_grad():
        model_path = get_InferNet_save_path(opt.ARTIFICIAL, seed, opt.net_root_path, opt.noise_pct, MODEL_NAME)
        mlp,_ = load_checkpoint(model_path, model)
        mlp.eval()                  # In case
        total_test_metric, GT_metric = validate_params(mlp,test_loader,opt.MIN_LOSS,device)

        print(f"========== MODEL_NAME =  {MODEL_NAME} | SEED = {seed} ==========")
        print(f"========== IN Test dataset, InferNet of {MODEL_NAME}:  {total_test_metric} ==========")
        print(f"========== IN Test dataset, the GTs: {GT_metric} ==========")
        print(f"========== IN Test dataset, the GTs: GT1,GT2(common),GT2(SA) ==========")
