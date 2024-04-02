# 0.
1. 小数据集和大数据集都只是需要一个表！
2. 小数据集需要算N，大数据集不需要算N

# 1. select and clean
1. 先保留必要的columns
2. clean，根据规则
3. 洗完的要求是: 当小数据集计算完N后，所有data必须可以用同样的逻辑去处理
4. 文件：`small_data_cleaning.ipynb`，`large_data_cleaning.ipynb`
5. 明确setting feature：

```python
unique_setting_GT = ['bidincrement','bidfee','retail','flg_fixedprice']
unique_setting_NN = ['desc','bidincrement','bidfee','retail','flg_fixedprice']
```

# 2. 计算NP
1. `small_data_cal_np.ipynb`：
    - 计算N，P
    - 输出的data：`data/small_auctions_np.csv`
    - 输出setting: `data/small_settings_NN.csv`和`data/small_settings_GT.csv`
2. `large_data_cal_np.ipynb`: 
    - 不需要计算N，data里已经包括这个数据
    - 输出的data：`data/large_auctions_np.csv`
    - 输出setting：`data/large_settings_NN.csv`和`data/large_settings_GT.csv`
3. 注意`P`最后都删去了`P[0]=1`这个值，保证P的长度=LEN=300，因此现在`P[i]`表示duration=i+1的概率，但是在math上还是按照p[t]表...t..

# 3. GT-1
1. `GT_1_gen.ipynb`: 很好做，分别做
   - 输出两个表：`data/info_asymm/results/GT_1_large_LEN=300.csv`和`data/info_asymm/results/GT_1_small_LEN=300.csv`

# 4. GT-2：
## 4.1 infer
1. 读入两个data，concat，然后再infer，在`SA_for_PT_one_for_all.ipynb`中，对所有setting做一次infer：
    - 输出的params：`data/SA_PT/params_opitim_oneforall.csv`
2. 注意infer时，受制于T的影响，当实际duration>T时，ignore it，不纳入NLL的计算中
## 4.2 gen
1. 读入params，直接infer，在`PT_gen_oneforall.ipynb`中
   - 输出两个表：`data/SA_PT/results/GT_2_large_LEN=300.csv`和`data/SA_PT/results/GT_2_small_LEN=300.csv`

# 5. DA
1. 见`data_augmentation.ipynb`，对于整理粒度和不整理粒度做了区分

