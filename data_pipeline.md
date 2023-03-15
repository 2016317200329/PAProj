# 0.
1. 小数据集和大数据集都只是需要一个表！
2. 小数据集需要算N，大数据集不需要算N
# 1. select and clean
1. 先保留必要的columns
2. clean，根据规则
3. 洗完的要求是: 当小数据集计算完N后，所有data必须可以用同样的逻辑去处理
4. 文件：`small_data_cleaning.ipynb`，`large_data_cleaning.ipynb`
# 2. 小数据集计算N
1. `small_data_cleaning.ipynb`：根据规则清洗+保留必要的columns+给desc添加id

# 3. 大数据集不需要计算N