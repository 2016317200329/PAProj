#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 10:57
# @Author  : Wang Yujia
# @File    : set_random_seed.py
# @Description :

import torch
import numpy as np
import random

# 随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True