#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/15 15:35
# @Author  : Wang Yujia
# @File    : test2_traces20.py
# @Description : 1. 用traces20.tsv测试，输出了一些基本信息


import pandas as pd

#为了测试生成的只有20行的小数据
outcome_path = "../data/outcomes20.tsv"
trace_path = "../data/traces20.tsv"


def AuctionBasicInfo() -> None:
    """
    输出dataset关于Auction的（统计）信息

    :return: None
    """

    # 读取tsv
    trace = pd.read_csv(trace_path, sep='\t')

    # 统计auction个数
    trace_grouped = trace.groupby('auction_id')
    auction_num = trace_grouped.ngroups
    print("\n -----trace.tsv记录了{}场auction".format(auction_num))

    return


def BasicInfo(trace_path: str) -> None:
    """
    输出dataset的一些基本信息

    :param trace20_path: dataset路径
    :return: None
    """

    trace = pd.read_csv(trace_path, sep='\t')

    # 输出数据集大小以及sample()
    print("-----trace.tsv有{}行".format(trace.shape[0]))
    print("-----trace.tsv数据集是这个样子：\n {}".format(trace.sample(5)))

    print("\n**********下面是trace20.tsv的【auction信息】**********\n")

    AuctionBasicInfo()

    return

if __name__ == '__main__':

    # 读取data，输出一些基本信息
    BasicInfo(trace_path)

