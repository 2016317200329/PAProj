#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 17:02
# @Author  : Wang Yujia
# @File    : BasicInfo_outcomes.py
# @Description : 关于outcomes数据集的一些信息,参考test1.py test2_traces20.py test3.py
# TODO：

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print输出在:
txt = "../BasicInfo_outcomes.txt"


# 设置长输出
pd.set_option('display.max_columns', 1000000)   # 可以在大数据量下，没有省略号
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_colwidth', 1000000)
pd.set_option('display.width', 1000000)

# 原data路径
outcomes_orignal_path = "../data/outcomes.tsv"
traces_original_path = "../data/traces.tsv"

# 读取tsv
outcomes = pd.read_csv(outcomes_orignal_path, sep='\t')
traces = pd.read_csv(traces_original_path, sep='\t')


def ProductBasicInfo() -> None:
    """
    输出dataset关于product的（统计）信息

    :return: None
    """

    # 1.统计auction with unique setting的个数
    # unique setting表示的是['product_id', 'bidincrement', 'bidfee']均一样
    outcomes_grouped = outcomes.groupby(['product_id', 'bidincrement', 'bidfee'])
    # 把这2列变成tuple形式('bidincrement','bidfee')
    outcomes_groupby_productid = outcomes[['bidincrement','bidfee']].apply(tuple, axis=1).to_frame()
    outcomes_groupby_productid.insert(0, 'product_id', outcomes[['product_id']].astype(int))
    outcomes_groupby_productid.columns=['product_id','(bidincrement, bidfee)']
    outcomes_groupby_productid = outcomes_groupby_productid.groupby('product_id')
    n_productid = outcomes_groupby_productid.ngroups
    # 只取auction_id的统计信息就够了,做一个count就好
    outcomes_grouped_info = outcomes_grouped['auction_id'].describe()['count']
    outcomes_grouped_info = outcomes_grouped_info.to_frame()
    print(40 * "- ", "\n", file = doc)
    print("-----对于outcomes.tsv, setting指这3个属性：['product_id', 'bidincrement','bidfee']", file = doc)
    print("-----在outcomes.tsv，有 {} 个'product_id'".format(n_productid), file = doc)
    print("-----在outcomes.tsv，有 {} 个unique setting.\n".format(outcomes_grouped_info.shape[0]), file = doc)

    # 2. 各个unique setting的样本数有多少个？
    # 添上一列作为临时的id
    outcomes_grouped_info['id'] = list(range(1, outcomes_grouped_info.shape[0] + 1))
    outcomes_grouped_info = outcomes_grouped_info[['count', 'id']]
    outcomes_grouped_info = outcomes_grouped_info.astype({"count": int})
    outcomes_grouped_info_size = outcomes_grouped_info.groupby(['count']).size().to_frame()
        # When we reset the index, the old index is added as a column
    outcomes_grouped_info_size.reset_index(inplace=True)
    outcomes_grouped_info_size.columns = ['count of auctions', 'count of unique setting']
    outcomes_grouped_info_size.sort_values(by=['count of auctions'], ascending=True, inplace=True)
    print("-----一个setting下的样本数过少不利于学习,因此对每个setting下的样本数做一个统计：", file = doc)
    print("-----如下2列，表示样本数为'count of auctions'的setting有'count of unique setting'个\n", file = doc)
    print(outcomes_grouped_info_size, file = doc)
    print(sum(outcomes_grouped_info_size['count of unique setting'][0:20]))

    # 画图, 效果不好
    fig = plt.figure()
    n = outcomes_grouped_info_size.shape[0]
    ax = fig.add_subplot(1, 1, 1)
    ax.set(ylabel='count of unique setting', xlabel='count of auctions')
    plt.scatter(outcomes_grouped_info_size['count of auctions'], outcomes_grouped_info_size['count of unique setting'], s = 3)
    plt.show()

    # 3. 统计在同一个`product_id`下，`['bidincrement','bidfee']`有多少不同的组合以及样本数
    # 主要是看有多少`product_id`，只有一组['bidincrement','bidfee']
    print(40 * "- ", "\n", file = doc)
    outcomes_groupby_productid = outcomes_groupby_productid.nunique()
    outcomes_groupby_productid.sort_values(by = ['(bidincrement, bidfee)'], ascending=True, inplace=True)
    print("-----在outcomes.tsv，有 {} 个'product_id'".format(n_productid), file = doc)
        # 条件计数
    n_one_product_id_setting = sum(1 for i in outcomes_groupby_productid['(bidincrement, bidfee)'] if i==1 )
    print("-----其中有 {} 个'product_id'只有1组(bidincrement, bidfee)".format(n_one_product_id_setting), file = doc)
    print("-----以下是各个 'product_id' 对应的 [bidincrement, bidfee] 个数", file = doc)
    outcomes_groupby_productid.reset_index(inplace=True)
    print(outcomes_groupby_productid, file = doc)
    print(40 * "- ", "\n", file = doc)

    return

def FlgInfo() -> None:
    """
    输出一些关于flag的统计信息

    :return: None
    """

    # flg_endprice为1的商品的finalprice都是0吗
    # flg_endprice: A binary flag indicating a 100%-off auction
    # finalprice: The price charged to the winner in dollars
    outcomes = pd.read_csv(outcomes_orignal_path, sep='\t')
    flg_endprice_notsame_df_1 = outcomes.query("flg_endprice == 1 & finalprice != 0")
    flg_endprice_notsame_df_2 = outcomes.query("flg_endprice != 1 & finalprice == 0")
    print("-----outcomes中，正确的数据应该是：flg_endprice为1 && finalprice为0 \n",file=doc)
    print("-----outcomes中，不符合上述规则的有： \n",file=doc)
    print(flg_endprice_notsame_df_1,file=doc)
    print(flg_endprice_notsame_df_2,file=doc)

    return

def AuctionBasicInfo() -> None:
    """
    输出dataset关于Auction的（统计）信息

    :return: None
    """

    # 统计auction个数
    traces_grouped = traces.groupby('auction_id')
        # auction_num = len(traces_grouped)
    auction_num = traces_grouped.ngroups
    print("-----traces.tsv记录了{}场auction".format(auction_num),file=doc)

    # 每场auction记录了多少条信息/ 每场auction经历了多少次bid
    bids_num_byauction = traces_grouped.count()['bid_number'].to_frame()
    bids_num_byauction.reset_index(inplace=True)
    bids_num_byauction.columns=['auction_id', 'bid_count']
    bids_num_byauction.sort_values(by='bid_count', ascending=False, inplace=True)
    print("-----traces.tsv中，标号为'auction_id'的竞拍进行了'bid_count'轮出价\n",file=doc)
    print(bids_num_byauction,file=doc)
    print("-----traces中，describe()：\n", file=doc)
    print(bids_num_byauction.describe(), file=doc)

    return

def Compare2Dataset() -> None:
    """
    比较2个数据集

    :return: None
    """

    print("\n**********下面是traces.tsv和outcomes.tsv的【auction信息】**********\n", file=doc)

    # traces里的拍卖数据是否在outcomes里都出现了?
    auction_id_traces = traces['auction_id'].unique().tolist()
    auction_id_outcomes = outcomes['auction_id'].unique()
    auction_id_comp = np.isin(auction_id_traces, auction_id_outcomes)
    print("-----traces.tsv详细数据集里有{}场拍卖数据".format(len(auction_id_traces)),file=doc)
    print("-----outcomes.tsv详细数据集里有{}场拍卖数据".format(auction_id_outcomes.size),file=doc)
    print("-----traces.tsv详细数据集里的拍卖数据，有{}个拍卖未出现在outcomes.tsv里".format(outcomes.shape[0]-auction_id_comp.size),file=doc)

    print("-----info:\n",outcomes.info,file=doc)
    return


def BasicInfo() -> None:
    """
    输出dataset的一些基本信息

    :return: None
    """

    # 输出数据集大小以及sample()
    print("-----outcomes.tsv有{}行，aka. auction数量".format(outcomes.shape[0]),file=doc)
    print("-----outcomes.tsv数据集是这个样子：\n {}".format(outcomes.sample(5)),file=doc)

    # 输出数据集大小以及sample()
    print("-----traces.tsv有{}行".format(traces.shape[0]),file=doc)
    print("-----traces.tsv数据集是这个样子：\n {}".format(traces.sample(5)),file=doc)


    print("\n**********下面是outcomes.tsv的【商品信息】**********\n",file=doc)
    ProductBasicInfo()

    print("\n**********下面是outcomes.tsv的【flag信息】**********\n",file=doc)
    FlgInfo()

    print("\n**********下面是traces.tsv的【auction信息】**********\n",file=doc)

    AuctionBasicInfo()

    return


if __name__ == '__main__':
    doc = open(txt, 'w')
    BasicInfo()
    # Compare2Dataset()
    doc.close()
    print("SUCCESS")