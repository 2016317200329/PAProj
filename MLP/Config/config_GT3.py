import os.path
from MLP.Config.config_base import BaseConfig  # 必须写full-path

class DefaultConfig(BaseConfig):
    def __init__(self):
        super().__init__()  # 调用父类的__init__方法以继承基础配置

        # tensorboard
        self.logs_str = r"GT_2_trial=1"
        self.tag_str = r""

        ######*********CHECK CAREFULLY***********########
        self.seed = 407  # seeds = [3,31,204,223,407]

        self.SET_VAL = False  # If False, 70% for training, 20% for testing; If True: the 20% is for vali, the 10% left is for testing
        self.DRAW_VAL = False  # Draw of not.

        self.USE_DA = False  # Use data augementation or not. This only work for REAL data

        self.ARTIFICIAL = True      # If True, use the syns dataset; If False, use the real dataset.
        self.EPOCH_NUM= 17
        self.noise_pct = 0.05  # 噪音占比:我们希望生成的data总体上最多浮动的百分比noise_pct

        ######*********CHECK CAREFULLY***********########


        # Training data
        self.train_path = os.path.join(self.data_root, "data/train_8_all")

        if self.ARTIFICIAL:
            self.target_path_metric = os.path.join(self.data_root, "data/artificial_targets_v2_" + "noise=" + str(self.noise_pct))
            self.target_path_loss = os.path.join(self.data_root, "data/artificial_targets_v2_" + "noise=" + str(self.noise_pct) + "_ls_T")
        else:
            self.target_path_metric = os.path.join(self.data_root, "data/targets_all")
            self.target_path_loss = os.path.join(self.data_root, "data/targets_all_ls_T")

        # Target data for metric calculation
        if self.ARTIFICIAL:
            self.params_opitim_path = os.path.join(self.data_root, "data/auction_assign.csv")
        else:
            # self.params_opitim_path = r"../data/SA_PT/params_opitim_delta_T.csv"
            self.params_opitim_path = os.path.join(self.data_root, "data/SA_PT/params_opitim_delta_T.csv")

        # train and optim
        if self.ARTIFICIAL:
            # For artificial dataset
            self.batch_size = 56  # 48 is great
            self.EPOCH_NUM= 15

            self.learning_rate = 5e-2                           # 1e-3, 5e-3
            self.lr_for_alpha = 5e-2   # 给mu单独设置learning rate   # 1e-2和5e-3都可以

        else:
            # For real dataset
            self.batch_size = 56  # 48 is great
            self.EPOCH_NUM = 30

            self.learning_rate = 5e-2                               # 1e-3, 5e-3
            self.lr_for_alpha = 1e-2     # 给mu单独设置learning rate   # 1e-2和5e-2  都可以 [1014: 5e-2]

        self.lr_decay = 0.95      # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = 5e-2  # for optimizer (regularization)   # 1e-3比较好

        # For scheduler
        self.StepLR_step_size = 5
        self.StepLR_gamma = 0.9

        self.Exp_gamma = 0.98

        self.SAFETY = 1e-30

        # Training data的粒度，画mdn时会使用到，
        self.SCALE = 1

        ################### DATA PATH ######################

        # Target data for loss calculation
        self.TARGET = 1
        self.arr_flag = False        # whether drop uniform data

        if self.ARTIFICIAL:
            self.NLL_metric_path = os.path.join(self.data_root, "data/GT_metric/NLL_metric_GT_Tgt=1_e30_artificial_v_2_noise="+str(self.noise_pct)+".csv")

        else:
            self.NLL_metric_path = os.path.join(self.data_root, "data/GT_metric/NLL_metric_GT_Tgt=1_e30.csv")


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