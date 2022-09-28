<u>这里介绍了这个proj各个文件夹/ 文件/ data 的说明，以及对于进度的说明</u>
[TOC]

# 测试data的路径

outcome_path = 'D:\\Desktop\\PROJ\\PracticePA\\data\\outcomes20.tsv'

trace_path = 'D:\\Desktop\\PROJ\\PracticePA\\data\\traces20.tsv'

# test_on_smalldata

1. 主要用来测试代码逻辑，在2个小数据集上进行的
2. 0425已经把命名规范化了，（把20什么的都去掉了），可以直接改data地址从而用到大数据集上

# BasicInfo

1.输出了一些数据集的信息

# MLP

1. `mlp`主要用来设计mlp，初步测试思想

# data_handeler
1. 一些对于data进行处理的代码

## data_extract_for_asc_symmetry.ipynb
1. 处理数据，为了增价拍卖+info不对称的paper但是用的是对称模型

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
### *info_asymm/datawithnp_asc_symmetry.csv*
1. 产生于../data_handler/data_extract_for_asc_symmetry.ipynb
2. 基于`calculate_n.py`,重新对数据进行整理。其实主要目的是把“price”和“fixed-price”什么的筛选出来
3. 注意其实traces.tsv里并没有记录任何fixed-price auction
4. unique_setting = ['product_id', 'bidincrement', 'bidfee','retail']，多了个'retail'
5. 未经过threshold筛选

### *info_asymm/datawithnp_asc_symmetry_selected.csv*
1. 产生于../data_handler/data_extract_for_asc_symmetry.ipynb
2. 经过threshold筛选
3. 可以发现，当threshold=16时，`unique_setting`增加一项'retail'会让筛选后的data从4614行变成3838

# 规范命名[never use them]

1. 类名： 单词首字母大写,单词无分隔
   参考：`class Logrecord(object):`
2. 普通变量、普通函数： 小写字母，单词用_分割
   参考：`exc_info = read_csv(file_path) `
3. 模块名(文件夹): 小写字母，单词用_分割
   参考: `test_on_smalldata`
4. 包名: 小写字母，单词之间用_分割
   参考: `logging`