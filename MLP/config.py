class DefaultConfig(object):

    N_gaussians = 3  # nums of Gaussian kernels

    # dataset划分
    batch_size = 40
    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    # train and optim.
    learning_rate = 5e-1
    lr_for_mu = 5e-1  # 给mu单独设置learning rate

    ####### 一般是5e-1，如果容易出现NaN，改成1e-2
    # learning_rate = 1e-2
    # lr_for_mu = 1e-2   # 给mu单独设置learning rate

    #### For W loss
    # learning_rate = 1e-3
    # lr_for_mu = 1e-3   # 给mu单独设置learning rate

    MIN_LOSS = 1e-4
    SAFETY = 1e-20

    # Training data的粒度，画图会使用到
    SCALE = 1
    # Target data是target_5时
    TARGET = 1

    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数


    ###################DATA PATH######################
    # Training data
    # train_path = r"../data/train_100"
    # train_path = r"../data/train_60"
    train_path = r"../data/train_300_v1"

    # Target data
    target_path = r"../data/targets"
    # target_path = r"../data/targets_5"
    # target_path = r"../data/targets_5_cdf"
    # data keys
    data_key_path = "../data/target_datakey.csv"

    # NLL metric

    NLL_metric_path = "../data/NLL_metric_GT_Tgt=1_e4.csv"
    # NLL_metric_path = "../data/NLL_metric_GT_Tgt=1_e6.csv"
    # NLL_metric_path = "../data/NLL_metric_GT_Tgt=1_e8.csv"
    # NLL_metric_path = "../data/NLL_metric_GT_Tgt=1_e10.csv"
    # NLL_metric_path  ="../data/NLL_metric_GT_Tgt=1_e20.csv"

    # Net path
    net_root_path = "../net_saved/"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# 原文链接：https://blog.csdn.net/lx_ros/article/details/122811361
