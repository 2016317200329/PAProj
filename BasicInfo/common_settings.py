#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 21:24
# @Author  : Wang Yujia
# @File    : common_settings.py
# @Description : 涉及的各种python包，输出路径，输入路径，print范围

import os
import pandas as pd
import torch

# print输出在:
txt = "../BasicInfo_outcomes.txt"
doc = open(txt,'w')

# 设置长输出
pd.set_option('display.max_columns', 1000000)   # 可以在大数据量下，没有省略号
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_colwidth', 1000000)
pd.set_option('display.width', 1000000)

# 原data路径
outcomes_orignal_path = "../data/outcomes.tsv"
traces_original_path = "../data/traces.tsv"

#
torch.set_default_tensor_type(torch.DoubleTensor)