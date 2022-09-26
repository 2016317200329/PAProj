#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 20:46
# @Author  : Wang Yujia
# @File    : calculate_n.py
# @Description : 1. method_1: 通过traces.tsv计算（count）`n` 2. method_2:通过outcomes计算`n`
#               3. 计算n对应的p 4. 根据阈值选data

from BasicInfo.common_settings import *


# 用到的各种data地址
output_file = "../data/data_withn.csv"
data_withn_path = "../data/data_withn.csv"
data_withnp_1_path = "../data/data_withnp_1.csv"
data_withnp_2_path = "../data/data_withnp_2.csv"
common_auction_id_path = "../data/common_auction_id.csv"
data_selected_key_1_path = "../data/data_withnp_1_selectedkeys.csv"
data_selected_key_2_path = "../data/data_withnp_2_selectedkeys.csv"
# 打算提取出来的data features
features_extracted = ['auction_id','product_id','bidincrement','bidfee']
# 样本数在threshold之下的settings不予考虑
threshold = 16


def method_1():
    """
    计算n的第一个方法：通过计算traces.tsv中的auction条目数量，来计算

    :return: flag
    """

    # 读取tsv，并且outcomes只保留需要的几列数据
    outcomes = pd.read_csv(outcomes_orignal_path, sep='\t')
    traces = pd.read_csv(traces_original_path, sep='\t')
    outcomes = outcomes[features_extracted]

    # 提取2个dataset共有的auction_id: `common_auction_id`
    common_auction_id = traces[traces['auction_id'].isin(outcomes['auction_id'])]
    common_auction_id = pd.DataFrame(common_auction_id['auction_id'].unique())
    # 以csv的形式保存下`common_auction_id`： "../data/method_1_data_withn.csv"
    common_auction_id.columns = ['auction_id']
    common_auction_id.to_csv(common_auction_id_path, encoding="utf-8")

    # 从outcomes中提取共有的auction_id的所有setting: common_auction_settings
    common_auction_settings = outcomes[outcomes['auction_id'].isin(common_auction_id['auction_id'])]

    # traces数据groupby，数n，并且保留需要的列: trace_groupby_auctionid
    trace_grouped = traces.groupby('auction_id')
    trace_groupby_auctionid = trace_grouped.count()['bid_user'].to_frame()
    # 整理以下data，index是'auction_id'，把index变成一列
    # reset index，不然报错：“'auction_id' is both an index level and a column label, which is ambiguous.”
    trace_groupby_auctionid.reset_index(drop=False, inplace=True)
    trace_groupby_auctionid.columns = ['auction_id', 'n_1']

    # 左连接两表，扩充traces数据：data_withn
    data_withn = pd.merge(trace_groupby_auctionid, common_auction_settings, on='auction_id', how="left")

    # 以csv的形式写入 "../data/method_1_data_withn.csv"
    data_withn.to_csv(output_file, header=True, encoding="utf-8")

    return 1

def method_2():
    """
    第2个计算n的方法：通过2个dataset共有的auction_id，在outcomes数据集中通过(finalprice-price)/bidincrement来计算

    :return: flag
    """

    # 读取outcomes和common_auction_id, outcomes只保留需要的列
    outcomes = pd.read_csv(outcomes_orignal_path, sep='\t')
    outcomes = outcomes[['auction_id', 'bidincrement', 'price']]
    common_auctionid_path = "../data/common_auction_id.csv"
    common_auction_id = pd.read_csv(common_auctionid_path)

    # 通过common_auction_id筛选outcomes数据：outcomes_filtered
    outcomes_filtered = outcomes[outcomes['auction_id'].isin(common_auction_id['auction_id'])]

    # 通过(price-0)/bidincrement来计算n，并添加进data里作为一列: outcomes_method2
    outcomes_method2 = outcomes_filtered.copy()
    outcomes_method2.loc[:,'n_2'] = \
        outcomes_filtered.loc[:,'price'] / (outcomes_filtered.loc[:,'bidincrement']*0.01)

    # 合并之前，整理数据
    outcomes_method2.loc[:, 'n_2'] = outcomes_method2['n_2'].astype('int')
    outcomes_method2.drop(axis=1,columns=['bidincrement','price'], inplace=True)

    # 续写在 "../data/data_withn.csv" 里，新建列
    with open(output_file, mode='r+', encoding="utf-8") as f:
        outcomes_method1 = pd.read_csv(f)
        data_withn = pd.merge(outcomes_method1, outcomes_method2, on='auction_id', how="left")

        # 重排数据列然后输出
        data_withn = data_withn.loc[:,['n_1','n_2','auction_id','product_id','bidincrement','bidfee']]
        data_withn.to_csv(output_file, header=True, encoding="utf-8")

    return 1

def diff():
    """
    计算2个方法的异同

    :return:
    """

    with open(data_withn_path,mode='r+',encoding="utf-8") as f:
        data_withn = pd.read_csv(f)

        # 计算2个n值的差值: df_temp
        data_withn.loc[:,'diff'] = data_withn.loc[:,'n_1'] - data_withn.loc[:, 'n_2']
        def cnt0(x):
            return 1 if (x==0) else 0
        df_temp = data_withn['diff'].apply(cnt0)

        # 计算2个n值不同的比例: percentage
        cnt_diff_equals0 = df_temp.sum()
        percentage = float(cnt_diff_equals0)/ data_withn.shape[0]

    return percentage

def cal_p():
    """
    对2个方法算出来的‘n’对应的‘p’进行计算，最后输出2个csv文档

    :return: flag
    """
    # output_file20 = "../data/data_withn20.csv"
    with open(output_file, mode='r', encoding="utf-8") as f:
        data_withn = pd.read_csv(f)
        data_withn.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

        # 1. 计算unique setting的auction总数'cnt_uniq': data_withn_cnt
        # unique setting表示的是['product_id', 'bidincrement', 'bidfee']均一样
        # 注意'cnt_uniq'并不需要出现在最后的data中
        data_grouped_tmp = data_withn.groupby(['product_id', 'bidincrement', 'bidfee'],as_index=False)
        tmp = data_grouped_tmp.size()
        data_withn_cnt = pd.merge(data_withn, tmp, on=['product_id', 'bidincrement', 'bidfee'], how="left")
        data_withn_cnt.rename(columns={'size': 'cnt_uniq'},inplace=True)
        data_withn_cnt.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

        # 2. 计算p_1和p_2: data_withn_cnt_n12
        # 2.1 先计算cnt_n_1，并添加到data_withn的一列: data_withn_cnt_n1
        # cnt_n_1表示某个setting下的n_1数值出现了几次/ cnt_n_1=2时，在某个setting下，一样的竞拍轮数n_1出现了2次
        data_grouped_tmp = data_withn.groupby(['product_id', 'bidincrement', 'bidfee', 'n_1'],as_index=False)
        tmp = data_grouped_tmp.size()
        data_withn_cnt_n1 = pd.merge(data_withn_cnt, tmp, on=['product_id', 'bidincrement', 'bidfee', 'n_1'], how="left")
        data_withn_cnt_n1.rename(columns={'size': 'cnt_n_1'},inplace=True)
        # 2.2 计算cnt_n_2，并添加到data_withn_cnt_n1的一列: data_withn_cnt_n12
        data_grouped_tmp = data_withn.groupby(['product_id', 'bidincrement', 'bidfee', 'n_2'], as_index=False)
        tmp = data_grouped_tmp.size()
        data_withn_cnt_n12 = pd.merge(data_withn_cnt_n1, tmp, on=['product_id', 'bidincrement', 'bidfee', 'n_2'], how="left")
        data_withn_cnt_n12.rename(columns={'size': 'cnt_n_2'}, inplace=True)
        del(data_withn_cnt_n1)

        # 2.3 计算p_1: p_1 = cnt_n_1 / cnt_uniq: data_withn_cnt_n12
        # `data_withn_cnt_n12`中包含了所需要的3个值：cnt_n_1 cnt_n_2 cnt_uniq
        tmp = data_withn_cnt_n12['cnt_n_1'] / data_withn_cnt_n12['cnt_uniq']
        data_withn_cnt_n12['p_1'] = tmp
        # 2.4 计算p_2: data_withn_cnt_n12: data_withn_cnt_n12
        tmp = data_withn_cnt_n12['cnt_n_2'] / data_withn_cnt_n12['cnt_uniq']
        data_withn_cnt_n12['p_2'] = tmp

        # 3 整理数据，然后输出: data_withnp_1, data_withnp_2
        # 3.1 'index'和'auction_id'是非重复的而且不需要记录，去掉这两列
        data_withn_cnt_n12.drop(columns=['auction_id','index'],inplace=True)
        # 3.2 2个表格，分别记录2种方法的n与p: data_withnp_1, data_withnp_2
        data_withnp_1 = data_withn_cnt_n12[['product_id', 'bidincrement', 'bidfee', 'n_1', 'cnt_n_1', 'p_1']]
        data_withnp_2 = data_withn_cnt_n12[['product_id', 'bidincrement', 'bidfee', 'n_2', 'cnt_n_2', 'p_2']]
        # 3.3 去重与输出: data_withnp_1.csv, data_withnp_2.csv
        data_withnp_1.drop_duplicates(inplace=True)
        data_withnp_2.drop_duplicates(inplace=True)
        data_withnp_1.to_csv(data_withnp_1_path, header=True, encoding="utf-8")
        data_withnp_2.to_csv(data_withnp_2_path, header=True, encoding="utf-8")

    return 1

def select_data(threshold):
    """
    根据阈值，选取样本数在threshold之上的setting作为数据集来使用

    :return:
    """

    with open(data_withnp_1_path, mode='r', encoding="utf-8") as f:
        data_withnp = pd.read_csv(f)
        data_withnp.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

        # 1. 统计每个unique setting[product_id,bidincrement,bidfee]下的样本数: data_ordered['size']
        # 统计的在某个setting下(n,p)的数量。和`basicinfo_datawithnp.py`里的代码一样
        data_withnp_grouped = data_withnp.groupby(by=['product_id', 'bidincrement', 'bidfee'], as_index=False)
        tmp = data_withnp_grouped.size()
        data_ordered = tmp.sort_values(by=['size'], ascending=True)
        data_ordered.reset_index(inplace=True)

        # 2. 根据阈值，提取并保存selected data的setting, 可以当做key使用: data_selected_key
        # 样本数:(data_ordered['size'])在threshold之上的settings才会被考虑进data_selected
        data_selected = data_ordered[data_ordered['size'] >= threshold][:]
        data_selected_key = data_selected[['product_id','bidincrement','bidfee']]
        data_selected_key.to_csv(data_selected_key_1_path, header=True, encoding="utf-8")
        total_amount = data_ordered.shape[0]
        data_selected_size = data_selected.shape[0]
        print("\n在当前threshold设置下，dataset包括了{}个setting\n".format(data_selected_size))
        print("当前threshold为{0}，相当于取了{1}%的settings\n".format(threshold, round(data_selected_size/total_amount*100, 3)))

    return 1

if __name__ == '__main__':

    flag = method_1()
    if (flag):
        print("\n method_1 which calculates the 'n' is SUCCESSFUL \n")

    flag = method_2()
    if(flag):
        print("\n method_2 which counts the 'n' is SUCCESSFUL \n")

    # 2个方法统计的`n`的比例相差多吗？
    # 其实没什么用，2个方法是差不多的
    percentage = diff()
    # There are 0.5883312933496532 'n' that are the same in method_1 and method_2
    print("\n There are {} 'n' that are the same in method_1 and method_2  \n".format(percentage))

    flag = cal_p()
    if(flag):
        print("\n cal_p() which calculated the 'p' is SUCCESSFUL \n")

    # threshold = 16
    flag = select_data(threshold)
    if(flag):
        print("\n select_data() which selects data according to the threshold is SUCCESSFUL \n")