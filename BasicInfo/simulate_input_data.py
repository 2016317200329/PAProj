#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 14:12
# @Author  : Wang Yujia
# @File    : simulate_input_data.py
# @Description : 从power law中采样，试图simulate出NN需要的data/ 试图simulate出GT model的output；
#                   并且根据selectedkey筛选出可以作为对照的data_withnp
# @Notation    : x: the num of rounds of an auction may last ('n' in the paper)
#                y: the num/frequence of auction-with-x-rounds ever appear (y/sum_of_y == 'p' in the paper )

import numpy.random
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

################# Global Params ###########################
# the size of sample data
N = 200
# power law param, between 0 to 1
a = 0.5
# params for Gaussian noise
mu = 0
sigma = 1
# size of data dropped in random
N_dropped = 40
# the file data written in
output_file_head = "../data/sim_data/data_sampled_"
output_file_tail = ".csv"
data_withnp_1_selected_path = "../data/data_withnp_1_selected.csv"
# data
data_withnp_1_selectedkeys_path = "../data/data_withnp_1_selectedkeys.csv"
data_withnp_1_path = "../data/data_withnp_1.csv"

################# Global Params ###########################
keys = pd.read_csv(data_withnp_1_selectedkeys_path)
n_of_files = keys.shape[0]
# generate one datafile for each key
for i in range(n_of_files):

    # 1. sample data
    # 1.1 sample x according to power-law distrb.
    # x = numpy.random.power(a,N)
    x = np.random.randint(1, 201, N)
    # 1.2 drop some data randomly
    # 注意这个方法可以完美delete！
    indices = np.arange(N)
    np.random.shuffle(indices)
    idx = indices[:(N-N_dropped)]
    x = np.sort(x[idx])
    # 如果x的人一个元素<0 返回True
    if any(x < 0):
        print("在循环{0}中，删除后的规模是{1} \n".format(i, x.size))

    # 1.3 add some noise to y
    eps = abs(random.gauss(mu, sigma))
    # 1.4 sample y and widen them(*80) and make them integer:
    sample_y_widen = (a*x**(a-1.)+eps+1)*100
    sample_y_int = np.round(sample_y_widen)
    # # 1.5 widen the range of x(+0.01*100) and make them integer
    # sample_x_widen = (x+0.01)
    # sample_x_int = np.round(sample_x_widen)

    ###### 因为要保证data的规模一致，所以决定delete掉相同比例的data[see 1.2]。这里是dele随机数量的data
    # # x should be more sparse as x grow larger
    # # make some random deletion (≈0.5)
    # num = sample_x_int.size
    # index_del = []
    # for i in range(num):
    #     tmp = np.random.randint(0, num)
    #     if(i >= tmp):
    #         index_del.append(int(i))
    # print("In total there are {} of data got deleted\n".format(len(index_del) / num))
    # sample_x_drop = np.delete(sample_x_int,index_del)
    # sample_y_drop = np.delete(sample_y_int,index_del)
    ###### 因为要保证data的规模一致，所以决定delete掉相同比例的data。这里是dele随机数量的data

    # 2.1 change 'y' into 'p': y/sum_of_y == p
    data = pd.DataFrame((x,sample_y_int),("x","y"))
    data = data.T
    sum_of_y = data['y'].sum()
    data['p'] = data['y'] / sum_of_y
    data.drop(labels='y',axis = 1,inplace=True)

    # plot
    sns.scatterplot(x= data.x, y = data.p)
    plt.show()

    # 3. add other attributes to the data: [product_id,bidincrement,bidfee]
    # 只有这样取出来的才是(1,3)格式的df！
    keys_1 = keys[i:(i+1)]
    # 注意加[]，这样append复制的是内容！
    # ignore_index=True一定要设置！
    keys_2 = keys_1.append([keys_1]*(data.shape[0]-1),ignore_index=True)
    data[['product_id','bidincrement','bidfee']] = keys_2

    # 4. output to the file: ../data/sim_data/data_sampled_str(i).csv
    output_file_path = output_file_head + str(i) + output_file_tail
    data.to_csv(output_file_path,index=False)

# 只运行一次就可以，平时注释掉
# # 5. select target data according to data_withnp_1_selectedkeys.csv
# # when [product_id,bidincrement,bidfee] are the same
# data_withnp_1 = pd.read_csv(data_withnp_1_path)
# data_withnp_1.drop(0,inplace=True)
# features = ['product_id','bidincrement','bidfee']
# # 用merge方法筛选数据！
# data_withnp_1_selected = pd.merge(data_withnp_1,keys[features],how="inner",on=features)
# data_withnp_1_selected.to_csv(data_withnp_1_selected_path,index=False)

# 只运行一次就可以，平时注释掉
# 6. Combine data_withnp_1_selected with ../data/sim_data/xxx respectively according to keys
# 根据key，在每一个training data的最后贴上target data，方便用dataloader读取，当然不用dataloader也可以
print("SUCCESS\n")