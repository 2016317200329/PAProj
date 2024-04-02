class DefaultConfig(object):
    # Input Choice
    GT_1 = 0
    GT_2 = 1
    EMD = 2

    N_gaussians = 2  # nums of Gaussian kernels

    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    ######*********CHECK CAREFULLY***********########
    # For synth [15]
    seed = 407                # Five-fold seeds = [3,31,204,223,407]

    # Which Model to be Chosen
    VALI_DETAIL = True            # If true: output details of validation for plot.
    Conv_1_4 = False               # If True: MB-MDN; If False MLP-MDN

    ARTIFICIAL = False      # If True, use the Syns dataset; If False, use the Real dataset.
    SET_VAL = False         # If False, 70% for training, 20% for testing; If True: the 20% is for vali, the 10% left is for testing
    DRAW_VAL = True        # Draw of not.

    USE_DA = False           # Use data augementation or not. This only work for REAL data
    REVENUE = False          # If True: compare the revenue.

    noise_pct = 0.05        # 噪音占比:我们希望生成的data总体上最多浮动的百分比noise_pct

    q = 1
    ######*********CHECK CAREFULLY***********########

    # For Synth data + MLP_MDN.
    if ARTIFICIAL and not Conv_1_4:
        EPOCH_NUM = 20
        batch_size = 24

        learning_rate = 1e-3
        lr_for_pi = 1e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

        lr_for_mu = 1e-2     # 给mu单独设置learning rate   # 1e-2和5e-2都可以
        lr_for_sigma = 1e-3  # 影响比较大。5e-3>1e-3

        lr_for_shape = 1e-3
        lr_for_scale = 1e-3  #1e-2或者1e-3

    # for Real data + MLP_MDN.
    if not ARTIFICIAL and not Conv_1_4:
        EPOCH_NUM = 30
        batch_size = 24

        learning_rate = 1e-3
        lr_for_pi = 1e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

        lr_for_mu = 1e-2     # 给mu单独设置learning rate   # 1e-2和5e-2都可以
        lr_for_sigma = 1e-3  # 影响比较大。5e-3>1e-3

        lr_for_shape = 1e-3
        lr_for_scale = 5e-3  #1e-2或者1e-3

    #### For W loss
    # learning_rate = 1e-3
    # lr_for_mu = 1e-2   # 给mu单独设置learning rate
    # lr_for_sigma = 1e-4

    lr_decay = 0.95      # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-2  # for optimizer (regularization)   # 1e-3比较好

    # For scheduler
    StepLR_step_size = 5
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

        # Visdom env
        env_str = "synthetic_seed="+str(seed)

        # Tensorboard
        logs_str = f"synthetic_seed=" + str(seed)
        vali_main_tag_str = "vali_metric/synthetic_seed=" + str(seed)

    else:
        # 使用SA预测的结果
        # train_path = "../data/train_300_uniq_all"
        # 使用NN infer的结果
        train_path = "../data/train_300_uniq_all_seed=" + str(seed)
        target_path_metric = "../data/targets_all"

        if USE_DA:
            target_path_loss = "../data/targets_all_5_DA_P=0.5_N_c=2"
        else:
            target_path_loss = "../data/targets_all"

        NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30_seed="+str(seed)+".csv"

        # output: Net saving path
        net_root_path = "net_saved/MDN_seed=" + str(seed) + ".pth"

        # Visdom env
        env_str = "real_seed="+str(seed)

        # Tensorboard
        logs_str = f"real_seed=" + str(seed)
        vali_main_tag_str = "vali_metric/real_seed=" + str(seed)

    # Target data for loss calculation
    TARGET = 1
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
