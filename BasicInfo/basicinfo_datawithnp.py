#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 15:46
# @Author  : Wang Yujia
# @File    : basicinfo_datawithnp.py
# @Description : 主要对于处理后（指计算出n与p）的data，进行一些分析
import pandas as pd

from BasicInfo.common_settings import *

data_withnp_1_path = "../data/data_withnp_1.csv"
data_withnp_2_path = "../data/data_withnp_2.csv"

def analysis(data_path):
    """
    对data_withnp_1.csv进行一些分析
    :return:
    """
    with open(data_path, mode='r', encoding="utf-8") as f:
        data_withnp = pd.read_csv(f)
        data_withnp.rename(columns={'Unnamed: 0': 'cnt_sample'}, inplace=True)

        # 1. 统计每个unique setting[product_id,bidincrement,bidfee]下的样本数: data_ordered['size']
        # 实际上和calculate_n.py里面的'cnt_uniq'一样，统计的在某个setting下(n,p)的数量
        data_withnp_grouped = data_withnp.groupby(by=['product_id','bidincrement','bidfee'],as_index=False)
        tmp = data_withnp_grouped.size()
        data_ordered = tmp.sort_values(by=['size'], ascending=True)
        data_ordered.reset_index(inplace=True)

        # 2. Trick: 根据柱状图可以看到基本样本数基本都不超过100
        print(data_ordered.describe())
        # 2.1 有多少>=100样本数的setting？: 8个/ 8个
        print(data_ordered['size'].map(lambda x: x >= 100).sum())
        # 2.2 [50,100)样本数的呢？: 4个/ 3个
        print(data_ordered['size'].map(lambda x: 50 <= x <100).sum())
        # 2.3 [40,50)样本数的呢？: 26个/ 28个
        print(data_ordered['size'].map(lambda x: 30 <= x <50).sum())
        # 2.4 [20,)样本数的呢？: 82个/ 83个
        print(data_ordered['size'].map(lambda x: x >= 20).sum())

        # 3. 画图看一看这些data在样本数量的分布情况
        plt.bar(x=data_ordered['index'], height=data_ordered['size'])
        plt.show()

    return 1

if __name__ == '__main__':
    flag = analysis(data_withnp_1_path)
    if (flag):
        print("\n analysis() which analysis 'data_withnp' is SUCCESSFUL \n")