class DefaultConfig(object):
    # tensorboard
    logs_str = r"GT_2_trial=1"
    tag_str = r""

    ######*********CHECK CAREFULLY***********########
    seed = 3             # seeds = [3,31,204,223,407]

    ARTIFICIAL = False      # If True, use the syns dataset; If False, use the real dataset.
    SET_VAL = False         # If False, 70% for training, 20% for testing; If True: the 20% is for vali, the 10% left is for testing
    DRAW_VAL = False        # Draw of not.

    USE_DA = False           # Use data augementation or not. This only work for REAL data

    noise_pct = 0.05        # 噪音占比:我们希望生成的data总体上最多浮动的百分比noise_pct

    ######*********CHECK CAREFULLY***********########

    # dataset划分
    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    # train and optim
    if ARTIFICIAL:
        # For artificial dataset
        batch_size = 40  # 48 is great

        learning_rate = 5e-2                           # 1e-3, 5e-3
        lr_for_alpha = 5e-2   # 给mu单独设置learning rate   # 1e-2和5e-3都可以

    else:
        # For real dataset
        batch_size = 64  # 48 is great

        learning_rate = 5e-2                                # 1e-3, 5e-3
        lr_for_alpha = 5e-2    # 给mu单独设置learning rate   # 1e-2和5e-2都可以

    lr_decay = 0.95      # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-2  # for optimizer (regularization)   # 1e-3比较好

    # For scheduler
    StepLR_step_size = 5
    StepLR_gamma = 0.9

    Exp_gamma = 0.98

    SAFETY = 1e-30

    # Training data的粒度，画mdn时会使用到，
    SCALE = 1

    ################### DATA PATH ######################

    # Training data
    train_path = r"../data/train_8_all"

    if ARTIFICIAL:
        target_path_metric = "../data/artificial_targets_v2_" + "noise=" + str(noise_pct)
        target_path_loss = "../data/artificial_targets_v2_" + "noise=" + str(noise_pct) + "_ls_T"
    else:
        target_path_metric = "../data/targets_all"
        target_path_loss = "../data/targets_all_ls_T"

    # Target data for metric calculation
    if ARTIFICIAL:
        params_opitim_path = r"../data/auction_assign.csv"
    else:
        params_opitim_path = r"../data/SA_PT/params_opitim_delta_T.csv"

    # Target data for loss calculation
    TARGET = 1
    arr_flag = False        # whether drop uniform data

    # data keys
    data_key_path = "../data/target_datakey_all.csv"

    # NLL metric
    MIN_LOSS = 1e-30

    if ARTIFICIAL:
        NLL_metric_path = r"../data/GT_metric/NLL_metric_GT_Tgt=1_e30_artificial_v_2_noise="+str(noise_pct)+".csv"

    else:
        NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30.csv"

    # Net path
    net_root_path = "net_saved/"



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
