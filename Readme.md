<u>这里介绍了这个proj各个文件夹/ 文件/ data 的说明，以及对于进度的说明</u>
[TOC]

# 测试data的路径

outcome_path = 'D:\\Desktop\\PROJ\\PracticePA\\data\\outcomes20.tsv'

trace_path = 'D:\\Desktop\\PROJ\\PracticePA\\data\\traces20.tsv'

# test_on_smalldata

1. 主要用来测试代码逻辑，在2个小数据集上进行的
2. 0425已经把命名规范化了，（把20什么的都去掉了），可以直接改data地址从而用到大数据集上

# BasicInfo

1.输出了一些数据集的basic信息
2. notebook比较好不要写.py了

# MLP

1. `mlp`主要用来设计mlp，初步测试思想

# data_handeler
1. 一些对于data进行处理的代码

## data_extract_for_asc_symmetry.ipynb
1. 处理数据，为了增价拍卖+info不对称的paper但是用的是对称模型

## GT_model/GT_asc_symmetry_gen.ipynb
1. 根据同路径下的GT_asc_symmetry_gen_demo.ipynb，针对所有setting，利用GT model进行预测，生成所需数据
2. GT模型是对称模型，包括fixed和ascending两个情况
3. 生成的data在data/info_asymm/results下

# data

## *outcomes.tsv*

1. 一共有*10个* ['bidincrement','bidfee'] 组合，数量如下 

```angular2html
   bidincrement  bidfee
   1             60         1794
              75        11717
   2             60         3421
   5             60           24
   6             60         9010
   7             75            6
   12            60        25233
   15            60           25
              75        65971
   24            60         4218
```

## *traces.tsv*

1. traces中的auction是否都出现在了`outcomes.tsv`中呢？**YES**

2. 一共记录了7353场auction
   
## *outcomes20.tsv*
- 是为了便于测试，进行设计的大概30行的data
  
## *traces20.tsv*

- 是为了便于测试，进行设计的大概30行的data
  
## *common_auction_id.csv*

- 记录了2个dataset共有的auction_id，产生于`BasicInfo/calculate_n.py`

- 一共有7353个共有的auction_id

## *data_withn.csv*

1. 产生于`BasicInfo/calculate_n.py`，记录了2个dataset共有的auction_id，用2个方法计算出来的n值，以及training可以用到的属性

2. 统计结果如下，7353场auction中一共有**5个** ['bidincrement','bidfee'] 组合，数量如下 
   
   ```angular2html
   bidincrement  bidfee
   1             60         143
   2             60         112
   6             60        1478
   12            60        5220
   24            60         400
   ```

`outcomes.tsv`是**10个**！！data_withn只包括bidfee为60的data！**which means `traces.tsv`只包括bidfee==60的auction**
3. 一共有unique setting **472**个

## *data_withnp_1.csv*

0. **需要的data都在这里，非常重要**
1. 产生于`calculate_n.py`的函数`cal_p()`,对第1个方法算出来的 ‘n’ 对应的 ‘p’ 进行计算
2. 有**6133** 条data，有**472**个unique setting
3. 列属性为：[product_id,bidincrement,bidfee, n_1,cnt_n_1,p_1]，释义是：对应于一个unique setting [product_id,bidincrement,bidfee], 有`cnt_n_1`场拍卖结束于`n_1`轮，这4个比例在当前setting下占比为`p_1`，
4. **'n'指的是`n_1`，'p'指的是`p_1`**
5. 根据`basicinfo_datawithnp.py`的分析，可以看到每个setting下样本不是很丰富, 75%分位数的样本数只有14（个）

```angular2html
            index    product_id  bidincrement  bidfee        size
count  472.000000  4.720000e+02    472.000000   472.0  472.000000
mean   235.500000  1.001327e+07     12.427966    60.0   12.993644
std    136.398925  1.538711e+03      6.152615     0.0   33.396511
min      0.000000  1.000522e+07      1.000000    60.0    1.000000
25%    117.750000  1.001236e+07     12.000000    60.0    2.000000
50%    235.500000  1.001373e+07     12.000000    60.0    4.000000
75%    353.250000  1.001412e+07     12.000000    60.0   14.000000
max    471.000000  1.001548e+07     24.000000    60.0  419.000000
```

## *data_withnp_2.csv*

0. **需要的data都在这里，非常重要**

1. 产生于`calculate_n.py`的函数`cal_p()`,对第2个方法算出来的‘n’对应的‘p’进行计算

2. 有**6219** 条data,具体数据释义如上

3. 根据`basicinfo_datawithnp.py`的分析，可以看到每个setting下样本不是很丰富,75%分位数的样本数**同样只有14（个）**
   
   ```angular2html
            index    product_id  bidincrement  bidfee        size
   count  472.000000  4.720000e+02    472.000000   472.0  472.000000
   mean   235.500000  1.001327e+07     12.427966    60.0   13.154661
   std    136.398925  1.538711e+03      6.152615     0.0   34.337890
   min      0.000000  1.000522e+07      1.000000    60.0    1.000000
   25%    117.750000  1.001236e+07     12.000000    60.0    2.000000
   50%    235.500000  1.001373e+07     12.000000    60.0    4.000000
   75%    353.250000  1.001412e+07     12.000000    60.0   14.000000
   max    471.000000  1.001548e+07     24.000000    60.0  431.000000
   ```

## *data_withnp_1_selectedkeys.csv*

1. 产生于`calculate_n.py`的函数`select_data()`,目的是根据阈值，select样本数在threshold之上的setting作为数据集来使用，这个csv保存了这些setting可以作为**key**使用
2. 一些threshold设置：（在method_1统计方法下）
   - threshold = 16时：包括了103个setting，相当于取了21.822%的settings
   - threshold = 17时：包括了99个setting，
3. 因此使用`data_withnp_1.csv`时最好根据这个文件筛选出来。

## *data_withnp_1_selected.csv*

1. 这个就是由上述key文件，从`data_withnp_1.csv`中select的data，作为target data使用

2. `simulate_input_data.py`最后一段有代码，可以随意挪用

## *./sim_data/xxx*

1. 产生于`simulate_input_data.py`，只需要run代码的part 1-4就可以产生data。下策，依照power-law采样出一些data来当做真实的GT模型的估计数据，注意加上列属性[product_id,bidincrement,bidfee]

2. 生成多个组的data呢？依照*data_withnp_1_selectedkeys.csv*来定的。一个文件对应了一个setting，对应了*data_withnp_1_selectedkeys.csv*的一行

## info_asymm
### *datawithnp_asc_symmetry.csv*
1. 产生于../data_handler/data_extract_for_asc_symmetry.ipynb
2. 基于`calculate_n.py`,重新对数据进行整理。其实主要目的是把“price”和“fixed-price”什么的筛选出来
3. 注意其实traces.tsv里并没有记录任何fixed-price auction
4. unique_setting = ['product_id', 'bidincrement', 'bidfee','retail']，多了个'retail'
5. 未经过threshold筛选

### *datawithnp_asc_symmetry_selected.csv*
1. 同上，但是经过threshold筛选. 
2. 可以发现，当threshold=16时，`unique_setting`增加一项'retail'会让筛选后的data从4614行变成3838

### *datawithnp_asc_symmetry_2.csv*
1. 产生于../data_handler/data_extract_for_asc_symmetry_2.ipynb
2. 用method-2算出来的，不需要traces.tsv
3. 记录了fixed-price auction
4. 未经过threshold筛选

### *datawithnp_asc_symmetry_2_selected.csv*
1. 同上，但是经过threshold筛选. 

### *datawithnp_fixed_symmetry.csv*
1. 产生于../data_handler/data_extract_for_fixed_symmetry.ipynb
2. 使用方法一，对于fixed-price的计算

### *datawithnp_fixed_symmetry.csv*
1. 同上，但是经过threshold筛选. 

## SA_PT
### *data_key.csv*
0. 不咋用这个了
1. 来自SA_for_PT_model.ipynb和SA_for_PT_model_delta_eq1.ipynb，两个文件输出的是一样的
2. 把PT模型中所有要估计SA参数的key：[product_id,bidincrement,bidfee,retail]的都输出出来
### *data_key_PT.csv*
1. 来自SA_for_PT_model_delta_eq1.ipynb
2. “_PT“表示是对应的是PT模型
### *data_key_PT_vbd.csv*
1. 来自PT_gen_oneforall.ipynb
2. "oneforall"表示是common params得出的结果
3. **实际上和common params与uniq params，他们的“data_key”是相同的**

### *params_opitim_delta.csv*
1. 来自SA_for_PT_model_delta_eq1.ipynb，当delta=1时对于另外两个参数的infer，
2. 中间文件罢了
### *params_opitim_delta_wset.csv*
1. 上表+data_key.csv组成的，来自SA_for_PT_model_delta_eq1.ipynb
2. 当delta=1时，对于另外两个参数的infer

### *params_opitim_delta_T.csv*
1. drop掉duration>T的samples
### *params_opitim_delta_T_wset.csv*
1. 同上

### *params_opitim_oneforall.csv*
1. 来自SA_for_PT_one_for_all.ipynb
2. 对所有settings infer出来的common params

### results
#### *PT_all1288_P.csv*
1. 来自`PT_gen.ipynb`保存了1288个auction setting的P结果

#### *PT_all1303_oneforall_P.csv*
1. “oneforall”表示infer一个common params，用这个params算P
2. “1303”表示只筛下去了duration < T的samples，剩下1303个settings

#### *PT_all1303_P.csv*
1. 对所有settings分别infer一组参数
2. “1303”表示只筛下去了duration < T的samples，剩下1303个settings

# GT_model
## *GT_asc_symmetry_gen.ipynb*
1. 对应的是小数据的那篇paper，计算的是增加拍卖的情况
## *GT_asc_symmetry_gen_demo.ipynb*
1. 上述实验，在某个setting上进行的小demo

## *GT_fixed_symmetry_gen.ipynb*
1. 对应的是小数据的那篇paper，计算的是定价拍卖的情况
## *GT_fixed_symmetry_gen_demo.ipynb*
1. 上述实验，在某个setting上进行的小demo

## *PT_demo.ipynb*
1. 对应的是PT模型的那篇paper，在某个setting上进行的小demo
2. 实际上是在通过SA做infer
## *PT_demo_table1.ipynb*
1. 对应的是PT模型的那篇paper
2. 是在复现table 1的结果

## *PT_gen_oneforall.ipynb*
1. 用所有settings infer得到的common params，去生成
2. 

## *SA_for_PT_model.ipynb*
1. 对应的是PT模型的那篇paper
2. 在*PT_demo.ipynb*的基础上，通过SA求解所有setting的参数
3. 但是慢，基本敲定用↓这个版本
## *SA_for_PT_model_delta_eq1.ipynb*
1. 对应的是PT模型的那篇paper
2. 在*PT_demo.ipynb*的基础上，通过SA求解所有setting的参数
3. 设置所有`delta=1`，只需要infer 2个参数，最终结果在*params_opitim_delta_wset.csv*
4. 注意有的结果infer不出来 QAQ

## *SA_for_PT_model_unsolved.ipynb*
1. 探究了一下30个infer不出来的setting是为什么
2. 探究完毕后infer出来了15个，剩30个无能为力
## *SA_for_PT_model_select_initialAlpha.ipynb*
1. 对15个infer不出来的settings，进行了(-0.3,0.3)上粒度为0.01的，600次对alpha值的查找，
2. But fail

## *SA_for_PT_one_for_all.ipynb*
1. 为所有settings，infer出 common params

## *SA_modified.py*
1. 改了下`sko.SA`。把不需要的部分注释掉
3. 在加噪音的时候，对于两个参数加了不同的噪音。（原code加的是一个噪音）

## *PT_gen.ipynb*
1. 用infer好的参数，求P并且保存P到`data/pt/results/PT_all1288_P.csv`
2. 一共1288个auction settings，把这些settings（P的keys）保存在`../data/SA_PT/data_key_15.csv`

# 规范命名 [never use them]

1. 类名： 单词首字母大写,单词无分隔
   参考：`class Logrecord(object):`
2. 普通变量、普通函数： 小写字母，单词用_分割
   参考：`exc_info = read_csv(file_path) `
3. 模块名(文件夹): 小写字母，单词用_分割
   参考: `test_on_smalldata`
4. 包名: 小写字母，单词之间用_分割
   参考: `logging`