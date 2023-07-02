class DefaultConfig(object):
    # Input Choice
    GT_1 = 0
    GT_2 = 1
    EMD = 2

    N_gaussians = 2  # nums of Gaussian kernels

    ########################################
    ARTIFICIAL = False         # If False, use the real dataset.
    SET_VAL = False         #If False, 70% for training, 20% for testing （True: the 20% is for vali, the 10% left is for testing）
    DRAW_VAL = False       # 是否画图
    # seed for real-data = [3,31,204,223,407]
    seed = 407
    noise_pct = 0.05  # 噪音占比:我们希望生成的data总体上最多浮动的百分比noise_pct
    ########################################

    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    # dataset划分 & lr
    if ARTIFICIAL:
        # for artificial data
        EPOCH_NUM = 30
        batch_size = 40

        learning_rate = 1e-3

        lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2
        lr_for_mu = 1e-2     # 给mu单独设置learning rate   # 1e-2和5e-2都可以
        lr_for_sigma = 1e-3  # 影响比较大。5e-3>1e-3

        lr_for_shape = 1e-3
        lr_for_scale = 5e-3  #1e-2或者1e-3
    else:
        # for real data.
        EPOCH_NUM = 30

        batch_size = 40  # 40比较好了
        learning_rate = 1e-3                            # 1e-3, 5e-3

        lr_for_pi = 1e-3    # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

        lr_for_mu = 1e-2    # 给mu单独设置learning rate   # 1e-2和5e-2都可以
        # lr_for_mu = 1e-3    # 给mu单独设置learning rate   # 1e-2和5e-2都可以
        lr_for_sigma = 5e-3  # 影响比较大。5e-3>1e-3
        # lr_for_sigma = 1e-3  # 影响比较大。5e-3>1e-3

        #### Weibull
        lr_for_shape = 1e-3
        lr_for_scale = 5e-3
        # lr_for_scale = 1e-3

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
    StepLR_step_size = 8
    StepLR_gamma = 0.9

    SAFETY = 1e-30

    # Training data的粒度，画mdn时会使用到，
    SCALE = 1

    ################### DATA PATH ######################

    # Data path
    if ARTIFICIAL:
        train_path = "../data/artificial_train_v2_noise=" + str(noise_pct) + "_seed=" + str(seed)

        target_path_metric = "../data/artificial_targets_v2_" + "noise=" + str(noise_pct)
        target_path_loss = "../data/artificial_targets_v2_" + "noise=" + str(noise_pct) + "_ls_T"

        NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30_all_" + "artificial_targets_v2_" + "noise=" + str(
        noise_pct) + "_seed=" + str(seed) + ".csv"
        # output: Net saving path
        net_root_path = "net_saved/MDN_artificial_seed=" + str(seed) + ".pth"

    else:
        # 使用SA预测的结果
        # train_path = "../data/train_300_uniq_all"
        # 使用NN infer的结果
        train_path = "../data/train_300_uniq_all_seed=" + str(seed)
        target_path_metric = "../data/targets_all"
        # target_path_loss = "../data/targets_all_5_DA_P=0.8_N_c=2"
        target_path_loss = "../data/targets_all_5_DA_P=0.5_N_c=2"

        NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30_seed="+str(seed)+".csv"

        # output: Net saving path
        net_root_path = "net_saved/MDN_seed=" + str(seed) + ".pth"

    # Target data for loss calculation
    TARGET = 5
    arr_flag = False        # whether drop uniform data
    # arr_path = r"../data/arr_selected/arr_targets_5_DA_P=0.5_N_c=3_K=3.npy"
    # arr_path = r"shuffled_indices.npy"
    arr_path = r"../data_handler/idx_GT2_better.pickle"
    # arr_path = r"../data/arr_selected/arr_targets_5_DA_P=0.5_N_c=3_K=2.npy"

    # data keys
    # data_key_path = "../data/target_datakey.csv"
    data_key_path = "../data/target_datakey_all.csv"

    # NLL metric
    MIN_LOSS = 1e-30



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
