#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 16:18
# @Author  : Wang Yujia
# @File    : test1_outcomes20.py
# @Description : 用outcomes20.tsv测试，输出了
#                1.outcomes20的基本信息 2.关于product的一些计数信息 3.关于flag的计数信息
#                4. 统计unique product

# TODO： 1. 规范命名： 2. 是否有进行了0轮的auction？

import pandas as pd
import matplotlib.pyplot as plt

# 原data路径
outcomes_original_path = "../data/outcomes.tsv"
traces_original_path = "../data/traces.tsv"

#为了测试生成的只有20行的小数据
outcomes_path = "../data/outcomes20.tsv"
trace_path = "../data/traces20.tsv"

# 读取tsv
outcomes = pd.read_csv(outcomes_path, sep='\t')


def GenData20(outcomes_orignal_path: str, outcomes_path: str) -> None:
    """
    为outcomes数据集生成一个用于测试代码的小数据集，截取前20行。实际上该代码没用到

    :param outcomes_orignal_path: 原tsv大文件的路径
    :param outcome20_path: 待生成的20行小文件的路径
    :return: None
    """

    outcomes_orignal = pd.read_csv(outcomes_orignal_path, sep='\t')
        #  index=False 不保留行索引，因为原数据集自带一个index
    outcomes_orignal.head(20).to_csv(outcomes_path, sep='\t', index=False)
    outcomes20 = pd.read_csv(outcomes_path, sep='\t')

    print("-----生成了20行小数据集的shape：{}".format(outcomes20.shape))
    print("-----生成了20行小数据集如下：\n {}".format(outcomes20))
    return


def ProductBasicInfo() -> None:
    """
    输出dataset关于product的（统计）信息

    :return: None
    """

    # 统计auction with unique setting的个数
    outcomes_grouped = outcomes.groupby(['product_id', 'bidincrement','bidfee'])
    outcomes_groupby_productid = outcomes[['bidincrement','bidfee']].apply(tuple, axis=1).to_frame()
    outcomes_groupby_productid.insert(0, 'product_id', outcomes[['product_id']].astype(int))
    outcomes_groupby_productid.columns=['product_id','(bidincrement, bidfee)']
    outcomes_groupby_productid = outcomes_groupby_productid.groupby('product_id')
    n_productid = outcomes_groupby_productid.ngroups
        # 只取auction_id的统计信息就够了,做一个count就好
    outcomes_grouped_info = outcomes_grouped['auction_id'].describe()['count']
    outcomes_grouped_info = outcomes_grouped_info.to_frame()
    n_unique_setting = outcomes_grouped_info.shape[0]
    # outcomes_grouped_info.sort_values(by=['count'], ascending=False, inplace=True)
    print(40*"- ","\n")
    print("-----对于outcomes.tsv，setting指这3个属性：['product_id', 'bidincrement','bidfee'] \n")
    print("-----在outcomes.tsv，有 {} 个'product_id'".format(n_productid))
    print("-----在outcomes.tsv，有 {} 个unique setting.\n".format(n_unique_setting))

    # 各个unique setting的样本数有多少个？
    # 添上一列作为临时的id
    outcomes_grouped_info['id'] = list(range(1,n_unique_setting+1))
    outcomes_grouped_info = outcomes_grouped_info[['count','id']]
    outcomes_grouped_info = outcomes_grouped_info.astype({"count": int})
    outcomes_grouped_info_size = outcomes_grouped_info.groupby(['count']).size().to_frame()
        # When we reset the index, the old index is added as a column
    outcomes_grouped_info_size.reset_index(inplace=True)
    outcomes_grouped_info_size.columns = ['count of auctions','count of unique setting']
    outcomes_grouped_info_size.sort_values(by=['count of auctions'], ascending=False, inplace=True)
    print("-----一个setting下的样本数过少不利于学习,因此对每个setting下的样本数做一个统计：")
    print("-----如下2列，表示样本数为'count of auctions'的setting有'count of unique setting'个\n")
    print(outcomes_grouped_info_size)

    # 画图，但是在大数据集上效果不好
    fig = plt.figure()
    n = outcomes_grouped_info_size.shape[0]
    ax = fig.add_subplot(1, 1, 1)
    ax.set(ylabel='count of unique setting', xlabel='count of auctions')
    plt.scatter(outcomes_grouped_info_size['count of auctions'], outcomes_grouped_info_size['count of unique setting'], s = 3)
    plt.show()

    # 统计在同一个`product_id`下，`['bidincrement','bidfee']`有多少不同的组合以及样本数
    # 有多少`product_id`，只有一组['bidincrement','bidfee']
    print(40 * "- ", "\n")
    outcomes_groupby_productid = outcomes_groupby_productid.nunique()
    outcomes_groupby_productid.sort_values(by = ['(bidincrement, bidfee)'], ascending=True, inplace=True)


    print("-----在outcomes.tsv，有 {} 个'product_id'".format(n_productid))
        # 条件计数
    n_one_product_id_setting = sum(1 for i in outcomes_groupby_productid['(bidincrement, bidfee)'] if i==1 )
    print("-----其中有 {} 个'product_id'只有1组(bidincrement, bidfee)".format(n_one_product_id_setting))
    print("-----以下是各个 'product_id' 对应的 [bidincrement, bidfee] 个数")
    outcomes_groupby_productid.reset_index(inplace=True)
    print(outcomes_groupby_productid)



    print(40 * "- ", "\n")


    #############################
    ###### 220426：下面这些统计是建立在product_id之上的，不能保证bid fee, bid inc的唯一性，因此不具有参考
    #############################
    # 每个product有几种
        # 注意['item']提取后会变成series格式
    outcomes_count_by_productid = outcomes_grouped.count()['item'].to_frame()
    outcomes_count_by_productid.columns=['count_by_product']
    # 排序
    outcomes_count_by_productid.sort_values(by=['count_by_product'], ascending=False, inplace=True)
    print("-----outcomes按照product_id分类，每一类商品被拍卖的次数：\n")
    print(outcomes_count_by_productid)


    # 出现x次的product有几个

    outcomes_count_by_productid.reset_index(inplace=True)
    outcomes_grouped_by_auctiontimes = outcomes_count_by_productid.groupby('count_by_product')
    outcomes_count_by_auctiontimes = outcomes_grouped_by_auctiontimes.count()
    outcomes_count_by_auctiontimes.reset_index(inplace=True)
    outcomes_count_by_auctiontimes.columns=['auction_times','product_count']
    outcomes_count_by_auctiontimes.sort_values(by = 'product_count',ascending=False, inplace=True)

    print("-----outcomes中，被拍卖'auction_times'次的商品有'product_count'个：\n")
    print(outcomes_count_by_auctiontimes)
    print("-----outcomes中，describe-2()：\n")
    print(outcomes_count_by_auctiontimes.describe())

    del outcomes

    return

def FlgInfo() -> None:
    """
    输出一些关于flag的统计信息

    :return: None
    """

    # flg_endprice为1的商品的finalprice都是0吗
    # flg_endprice: A binary flag indicating a 100%-off auction
    # finalprice: The price charged to the winner in dollars
    outcomes = pd.read_csv(outcomes_path, sep='\t')
    flg_endprice_notsame_df_1 = outcomes.query("flg_endprice == 1 & finalprice != 0")
    flg_endprice_notsame_df_2 = outcomes.query("flg_endprice != 1 & finalprice == 0")
    print("-----outcomes中，正确的数据应该是：flg_endprice为1 && finalprice为0 \n")
    print("-----outcomes中，不符合上述规则的有： \n")
    print(flg_endprice_notsame_df_1)
    print(flg_endprice_notsame_df_2)









    return


def BasicInfo(outcomes_path: str) -> None:
    """
    输出dataset的一些基本信息

    :param outcomes_path: dataset路径
    :return: None
    """

    outcomes = pd.read_csv(outcomes_path, sep='\t')

    # 输出数据集大小以及sample()
    print("-----outcomes.tsv有{}行，aka. auction数量".format(outcomes.shape[0]))
    print("-----outcomes.tsv数据集是这个样子：\n {}".format(outcomes.sample()))

    print("\n**********下面是outcomes.tsv的【商品信息】**********\n")
    ProductBasicInfo()

    print("\n**********下面是outcomes.tsv的【flag信息】**********\n")
    FlgInfo()

    return


if __name__ == '__main__':

    # 生成20行小数据，测试用，跑一次就够了
    # 代码正确，实际上没有用到
    # GenData20(outcomes_original_path, outcomes20_path)

    # 读取data，输出一些基本信息
    BasicInfo(outcomes_path)


