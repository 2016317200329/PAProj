class DefaultConfig(object):
    GT_CHOSEN = 1

    N_gaussians = 2  # nums of Gaussian kernels

    # dataset划分
    batch_size = 40     # 40比较好了
    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    # train and optim                  .
    learning_rate = 5e-3                            # 1e-3, 5e-3
    lr_for_pi = 1e-3   # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2
    lr_for_mu = 1e-2   # 给mu单独设置learning rate   # 1e-2和5e-2都可以
    lr_for_sigma = 5e-3                            # 影响比较大。5e-3>1e-3


    ####
    lr_for_shape = 1e-4
    lr_for_scale = 1e-3
    ############## tensorboard
    logs_str = f"padding-8"
    tag_str = f""

    #### For W loss
    # learning_rate = 1e-3
    # lr_for_mu = 1e-2   # 给mu单独设置learning rate
    # lr_for_sigma = 1e-4

    lr_decay = 0.95      # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-2  # for optimizer (regularization)   # 1e-3比较好

    # For scheduler
    StepLR_step_size = 10
    StepLR_gamma = 0.8

    SAFETY = 1e-30

    # Training data的粒度，画mdn时会使用到，
    SCALE = 1

    ################### DATA PATH ######################

    # Training data
    # train_path = r"../data/train_100"
    # train_path = r"../data/train_60"
    train_path = r"../data/train_300_v1"

    # Target data for metric calculation
    target_path_metric = r"../data/targets"

    # Target data for loss calculation
    TARGET = 5
    arr_flag = False        # whether drop uniform data
    # target_path_loss = r"../data/targets_5_DA_P=0.5_N_c=3"
    # arr_path = r"../data/arr_selected/arr_targets_5_DA_P=0.5_N_c=3_K=3.npy"
    arr_path = r"shuffled_indices.npy"

    # arr_path = r"../data/arr_selected/arr_targets_5_DA_P=0.5_N_c=3_K=2.npy"

    # target_path_loss = r"../data/targets_5_DA_P=1_N_c=3"        # 表现存疑
    target_path_loss = r"../data/targets_5_DA_P=0.5_N_c=2"      # 可以考虑，优先这个
    # target_path_loss = r"../data/targets_5_DA_P=0.5_N_c=3"        # 和上面差不多
    # target_path_loss = r"../data/targets_5"
    # target_path_loss = r"../data/targets"
    # target_path = r"../data/targets_5_cdf"

    # data keys
    data_key_path = "../data/target_datakey.csv"

    # NLL metric
    MIN_LOSS = 1e-30
    NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30.csv"
    # NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e40.csv"
    # NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e50.csv"

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
