class DefaultConfig(object):
    # tensorboard
    logs_str = r"GT_2_trial=1"
    tag_str = r""

    N_gaussians = 2
    seed = 407 # seed = [3,31,204,223,407]
    noise_pct = 0.05  # 噪音占比:我们希望生成的data总体上最多浮动的百分比noise_pct

    ARTIFICIAL = True
    # Input NN path
    if ARTIFICIAL:
        model_params_MLP = r"../../MLP/net_saved/NN_params_infer_artificial_v2_noise=" + str(noise_pct) + "_seed=" + str(
            seed) + ".pth"
    else:
        model_params_MLP = r"../../MLP/net_saved/NN_params_infer_seed=" + str(seed) + ".pth"

    # Trained-well NN
    # output path
    if ARTIFICIAL:
        params_gen_path = r"../../data/SA_PT/params_artificial_v2_noise=" + str(noise_pct)+"_seed=" + str(seed) + ".csv"
    else:
        params_gen_path = r"../../data/SA_PT/params_seed=" + str(seed) + ".csv"

    # dataset划分
    batch_size = 40    # 40比较好了
    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    SAFETY = 1e-30

    # Training data的粒度，画mdn时会使用到，
    SCALE = 1

    ################### DATA PATH ######################

    # Training data
    train_path = r"../../data/train_8_all"

    #######################从这里往下其实都不需要
    if ARTIFICIAL:
        target_path_metric = "../../data/artificial_targets_v2_" + "noise=" + str(noise_pct)
        target_path_loss = "../../data/artificial_targets_v2_" + "noise=" + str(noise_pct) + "_ls_T"

    else:
        target_path_metric = "../../data/artificial_targets_v2_" + "noise=" + str(noise_pct)
        target_path_loss = "../../data/artificial_targets_v2_" + "noise=" + str(noise_pct) + "_ls_T"


    # Target data for loss calculation
    TARGET = 5
    arr_flag = False        # whether drop uniform data

    # data keys
    data_key_path = "../../data/target_datakey_all.csv"

    # NLL metric
    MIN_LOSS = 1e-30
    # # NLL_metric_path = r"../../data/GT_metric/NLL_metric_GT_Tgt=1_e30_all_compare.csv"
    NLL_metric_path = r"../../data/GT_metric/NLL_metric_GT_Tgt=1_e30_artificial_v_2.csv"
    #
    # params_opitim_path = r"../../data/params_and_K_sampled.csv"
    params_opitim_path = r"../../data/auction_assign.csv"

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
