#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 13:49
# @Author  : Wang Yujia
# @File    : stat_large_dataset.py
# @Description : 对大数据集的一些统计：1.

from BasicInfo.common_settings import *

largedata_root_path = "../data/"
filename = "auctions.csv"
datapath_auction = largedata_root_path + filename
auction_data_df = pd.read_csv(datapath_auction)
print("auction.csv的shape：",auction_data_df.shape)

N = auction_data_df.shape[0]

# 1. what do ‘y’ and 'k' columns repesent?
data_df = pd.DataFrame([auction_data_df.auctionid,auction_data_df.y, auction_data_df.k]).T
print("提前3列作为df：shape：",data_df.shape)
print(data_df.head(5))
N = data_df.shape[0]
ans = pd.value_counts(auction_data_df.k) / N
# 'k'的统计数据和原文对得上，因此k是bid_inc

# 2. fixedprice和pennyauction这两列表示auction的属性？
# 2.1 判断和是不是N: No,
# 而且两列之和和N的比例为：tmp = 9960+10709, tmp/N=0.124
print("N:",N)
data_df = pd.DataFrame([auction_data_df.fixedprice,auction_data_df.pennyauction]).T
tmp = data_df.sum().sum()
print("这两列1的个数：",tmp)
# 2.2 这两列是不是只有1和0两个数字: Yes
ans_1 = pd.value_counts(data_df.fixedprice).shape
ans_2 = pd.value_counts(data_df.pennyauction).shape
# 两列是不是有同时为1的：Yes但是不多
tmp = data_df[(data_df.fixedprice == 1) & (data_df.pennyauction == 1)].copy()
print(tmp.shape)
# 2.3 两列是不是有同时为0的：Yes而且数量很多
tmp = data_df[(data_df.fixedprice == 0) & (data_df.pennyauction == 0)].copy()
print(tmp.shape)

# 3. paper里写09年之后的data都记录了top-10 auction情况，所以auction_id是唯一的吗:Yes
ans = pd.value_counts(auction_data_df.auctionid)
ans = pd.DataFrame(auction_data_df.auctionid).nunique()
print("auction_id是唯一的吗:", ans==N)