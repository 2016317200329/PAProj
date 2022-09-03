#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/15 19:26
# @Author  : Wang Yujia
# @File    : test3.py
# @Description : 1. outcomes和traces的对比 2. 提取traces中记录的auction的bidfee, bidincrement

import pandas as pd
import numpy as np
import csv

#为了测试生成的只有20行的小数据
outcome_path = "../data/outcomes20.tsv"
trace_path = "../data/traces20.tsv"

def compare_2dataset() -> None:
    """
    比较2个数据集

    :return: None
    """

    # 读取tsv
    outcome = pd.read_csv(outcome_path, sep='\t')
    trace = pd.read_csv(trace_path, sep='\t')

    # traces里的拍卖数据是否在outcomes里都出现了?
    auction_id_traces = trace['auction_id'].unique().tolist()
    auction_id_outcomes = outcome['auction_id'].unique()
    auction_id_comp = np.isin(auction_id_traces, auction_id_outcomes)
    print("-----traces.tsv详细数据集里有{}场拍卖数据".format(len(auction_id_traces)))
    print("-----outcomes.tsv详细数据集里有{}场拍卖数据".format(auction_id_outcomes.size))
    print("-----traces.tsv详细数据集里的拍卖数据，有{}个拍卖未出现在outcomes.tsv里".format(outcome.shape[0]-auction_id_comp.size))

    print("-----info:\n",outcome.info)
    return

def contract_info() -> object:
    """
    提取outcomes的settings，为traces中拍卖数据补充setting信息,从而计算n

    :return：a sub_df of traces:['auction_id', 'product_id', 'bidincrement', 'bidfee']
    """

    # 读取tsv，并且outcomes只保留需要的几列数据
    outcomes = pd.read_csv(outcome_path, sep='\t')
    traces = pd.read_csv(trace_path, sep='\t')
    outcomes = outcomes[['auction_id', 'product_id', 'bidincrement', 'bidfee']]

    # 提取2个dataset共有的auction_id: `common_auction_id`
    common_auction_id = traces[traces['auction_id'].isin(outcomes['auction_id'])]
    common_auction_id = common_auction_id['auction_id'].unique()

    #从outcomes中提取共有的auction_id的所有setting: common_auction_settings
    common_auction_settings = outcomes[outcomes['auction_id'].isin(common_auction_id)]

    # traces数据groupby，数n，并且保留需要的列: trace_groupby_auctionid
    trace_groupby_auctionid = traces.groupby('auction_id')
    trace_groupby_auctionid = trace_groupby_auctionid.count()['bid_number'].to_frame()
    temp = trace_groupby_auctionid.index.to_frame()
    trace_groupby_auctionid['auction_id'] = temp
    trace_groupby_auctionid.columns=['n','auction_id']

    # 合并两表：data_withn
    # reset index，不然报错：“'auction_id' is both an index level and a column label, which is ambiguous.”
    trace_groupby_auctionid.reset_index(drop=True, inplace=True)
    data_withn = pd.merge(trace_groupby_auctionid, common_auction_settings, on='auction_id', how="left")

    # 以csv的形式,续写入 "../data/method_1_data_withn.csv"
    output_file = "../data/method_1_data_withn.csv"
    data_withn.to_csv(output_file, index=None, header=True, encoding="utf-8")

    # return data_withn


if __name__ == '__main__':
    compare_2dataset()