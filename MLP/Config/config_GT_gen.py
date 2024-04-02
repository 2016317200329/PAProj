import os.path
from .config_base import BaseConfig  # 注意路径写法

class DefaultConfig(BaseConfig):
    def __init__(self):
        super().__init__()  # 调用父类的__init__方法以继承基础配置

        self.N_gaussians = 2

        self.GT_w_Params = 2  # GT2 or GT3. If GT1, this will be ignored

        self.seed = 62        # seed = [3,31,62,204,223,407,508,626], [4,31,35,204,407,66,508]
        self.noise_pct = 0.05  # 噪音占比:我们希望生成的data总体上最多浮动的百分比noise_pct

        self.ARTIFICIAL = False

        # Input NN path with old seed
        if self.ARTIFICIAL:
            self.model_params_MLP = os.path.join(self.data_root, "MLP/net_saved/NN_params_infer_GT"+str(self.GT_w_Params)+"_artificial_v2_noise=" + str(self.noise_pct) + "_seed=" + str(self.seed) + ".pth")
        else:
            self.model_params_MLP = os.path.join(self.data_root, "MLP/net_saved/NN_params_infer_GT"+str(self.GT_w_Params)+ "_seed=" + str(self.seed) + ".pth")

        # output path
        if self.ARTIFICIAL:
            self.params_gen_path = os.path.join(self.data_root, "data/SA_PT/params_artificial_GT"+str(self.GT_w_Params)+"_noise=" + str(self.noise_pct) + "_seed=" + str(self.seed) + ".csv")
        else:
            self.params_gen_path = os.path.join(self.data_root, "data/SA_PT/params_GT"+str(self.GT_w_Params)+"_seed=" + str(self.seed) + ".csv")

        ################### DATA PATH ######################
        # Training data
        self.train_path = os.path.join(self.data_root, "data/train_8_all")

        ####################################从这里往下其实都不需要

        if self.ARTIFICIAL:
            self.target_path_metric = os.path.join(self.data_root, "data/artificial_targets_v2_" + "noise=" + str(self.noise_pct))
            self.target_path_loss = os.path.join(self.data_root, "data/artificial_targets_v2_" + "noise=" + str(self.noise_pct) + "_ls_T")

        else:
            self.target_path_metric = os.path.join(self.data_root, "data/artificial_targets_v2_" + "noise=" + str(self.noise_pct))
            self.target_path_loss = os.path.join(self.data_root, "data/artificial_targets_v2_" + "noise=" + str(self.noise_pct) + "_ls_T")

        # Target data for loss calculation
        self.TARGET = 1
        self.arr_flag = False        # whether drop uniform data

        # # NLL_metric_path = r"../../data/GT_metric/NLL_metric_GT_Tgt=1_e30_all_compare.csv"
        self.NLL_metric_path = os.path.join(self.data_root, "data/GT_metric/NLL_metric_GT_Tgt=1_e30_artificial_v_2_noise="+str(self.noise_pct)+".csv")
        #
        # params_opitim_path = r"../../data/params_and_K_sampled.csv"
        self.params_opitim_path = os.path.join(self.data_root, "data/auction_assign.csv")

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
