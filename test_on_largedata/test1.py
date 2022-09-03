#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 19:52
# @Author  : Wang Yujia
# @File    : test1_outcomes20.py
# @Description : 1. 读dta数据并且转换成csv

import pandas as pd
# data = pd.io.stata.read_stata("C:\\Users\\Wang Yujia\\Desktop\\Yu\\Datasets\\LargeDataset\\rdv037 supplementary data\\DataPackage\\MainData\\auctions.dta")
# utf-8可能乱码。换成ansi就好了
# data.to_csv("../data/auctions.csv", encoding = 'utf-8')

# 1. 读dta数据并且转换成csv
data = pd.io.stata.read_stata("C:\\Users\\Wang Yujia\\Desktop\\Yu\\Datasets\\LargeDataset\\rdv037 supplementary data\\DataPackage\\MainData\\auctions_reshaped.dta")
data.to_csv("../data/auctions_reshaped.csv", encoding = 'utf-8')



