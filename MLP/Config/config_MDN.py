import os.path
from .config_base import BaseConfig  # 注意路径写法

class DefaultConfig(BaseConfig):
    def __init__(self,MODEL_NAME = None,seed=None):
        super().__init__()  # 调用父类的__init__方法以继承基础配置

        self.N_gaussians = 2         # nums of Gaussian kernels

        if seed:
            self.seed = seed
        else:
            self.seed = 99999                   # seed = [3,31,204,223,407,62,508,626], [4,31,35,204,407,66,508]

        self.W_GT3 = True            # Whether or not use GT3, it will decide which dataset we use. Default: True

        self.ARTIFICIAL = True      # If True, use the syns dataset; If False, use the real dataset.
        self.REVENUE = False        # If True: compare the revenue.

        self.SET_VAL = False        # If False, 70% for training, 20% for testing; If True: the 20% is for vali, the 10% left is for testing
        self.DRAW_VAL = False       # Draw Vali results not.
        self.PRINT_VAL = True       # Print or Not the vali result

        self.USE_DA = False  # Use data augementation or not. This only work for REAL data
        self.noise_pct = 0.05  # 噪音占比:我们希望生成的data总体上最多浮动的百分比noise_pct

        self.VALI_DETAIL = False    # If true: output details of validation for plot.
        self.Conv_1_4 = True        # If True: MB-MDN； If False MLP-MDN

        self.q = 1
        # For scheduler
        self.StepLR_step_size = 5
        self.StepLR_gamma = 0.92

        ######*********CHECK CAREFULLY***********########
        if MODEL_NAME == "GT1(MDN)":
            # For both synthetic and real data
            self.lr_for_pi = 1e-3
            self.lr_for_shape = 1e-3  # 除了508：1e-3
            self.lr_for_scale = 5e-4
            self.learning_rate = 1e-3  # 除了508：1e-3

            self.batch_size = 56
            self.EPOCH_NUM = 20
        elif MODEL_NAME == "GT2(MDN)":
            if self.ARTIFICIAL:
                # For both synthetic and real data
                self.lr_for_pi = 5e-3
                self.lr_for_shape = 5e-3 # 除了508：1e-3
                self.lr_for_scale = 5e-3
                self.learning_rate = 5e-3  # 除了508：1e-3

                self.batch_size = 48
                self.EPOCH_NUM = 25
            if not self.ARTIFICIAL:
                self.lr_for_pi = 1e-3
                self.lr_for_shape = 5e-3
                self.lr_for_scale = 1e-3
                self.learning_rate = 1e-3
                self.batch_size = 56
                self.EPOCH_NUM = 20
        elif MODEL_NAME == "GT3":
            if self.ARTIFICIAL:
                # For both synthetic and real data
                self.lr_for_pi = 5e-3
                self.lr_for_shape = 5e-3 # 除了508：1e-3
                self.lr_for_scale = 5e-3
                self.learning_rate = 5e-3  # 除了508：1e-3

                self.batch_size = 48
                self.EPOCH_NUM = 25
            if not self.ARTIFICIAL:
                self.lr_for_pi = 1e-3
                self.lr_for_shape = 5e-3
                self.lr_for_scale = 1e-3
                self.learning_rate = 1e-3
                self.batch_size = 56
                self.EPOCH_NUM = 20
        elif MODEL_NAME == "EMD":
            if self.ARTIFICIAL: # 文件里存的是real的params
                self.EPOCH_NUM = 25
                self.batch_size = 56

                self.learning_rate = 1e-3 # 测试中：5e-3
                self.lr_for_pi = 1e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 5e-3
                self.lr_for_scale = 5e-3  #1e-2或者1e-3
            else:
                self.EPOCH_NUM = 25
                self.batch_size = 56

                self.learning_rate = 1e-3  # 测试中：5e-3
                self.lr_for_pi = 1e-3  # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 1e-3
                self.lr_for_scale = 1e-3  # 1e-2或者1e-3
        elif MODEL_NAME == "GT1+EMD":
            # For both synthetic and real data
            self.EPOCH_NUM = 20
            self.batch_size = 64 # 测试中：48

            self.learning_rate = 1e-3 # 测试中：5e-3
            self.lr_for_pi = 1e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3  #1e-2或者1e-3
        elif MODEL_NAME == "GT2+EMD":
            # For both synthetic and real data

            self.EPOCH_NUM = 20
            self.batch_size = 40 # 测试中：48

            self.learning_rate = 1e-3 # 测试中：5e-3
            self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3  #1e-2或者1e-3
        elif MODEL_NAME == "GT3+EMD":
            # For both synthetic and real data
            self.EPOCH_NUM = 20
            self.batch_size = 40 # 测试中：48

            self.learning_rate = 1e-3 # 测试中：5e-3
            self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3  #1e-2或者1e-3
        elif MODEL_NAME == "GT1+GT2":
            # For both synthetic and real data
            if self.ARTIFICIAL:
                self.EPOCH_NUM = 20
            if not self.ARTIFICIAL:
                self.EPOCH_NUM = 25
            self.batch_size = 40 # 测试中：48

            self.learning_rate = 1e-3 # 测试中：5e-3
            self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3  #1e-2或者1e-3
        elif MODEL_NAME == "GT1+GT3":
            # For both synthetic and real data
            self.lr_for_pi = 5e-3
            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3
            self.learning_rate = 5e-3
            self.batch_size = 48
            self.EPOCH_NUM = 25
        elif MODEL_NAME == "GT2+GT3":
        # For both synthetic and real data
            self.lr_for_pi = 5e-3
            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3
            self.learning_rate = 1e-3
            self.batch_size = 40
            self.EPOCH_NUM = 25

        elif MODEL_NAME == "GT1_GT2_EMD":
            # Datasets split & lr
            if self.ARTIFICIAL:
                self.EPOCH_NUM = 20
                self.batch_size = 40

                self.learning_rate = 1e-3
                self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 5e-3
                self.lr_for_scale = 5e-3  #1e-2或者1e-3

            # for Real data + MB_MDN.
            if not self.ARTIFICIAL:

                self.EPOCH_NUM = 25

                self.batch_size = 40  # 40比较好了
                self.learning_rate = 1e-3                            # 1e-3, 5e-3

                self.lr_for_pi = 5e-3    # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 1e-2
                self.lr_for_scale = 5e-3
        elif MODEL_NAME == "GT1_GT3_EMD":
            if self.ARTIFICIAL:
                self.EPOCH_NUM = 25
                self.batch_size = 40

                self.learning_rate = 5e-3
                self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 5e-3
                self.lr_for_scale = 1e-2  #1e-2或者1e-3
            else:
                self.EPOCH_NUM = 25
                self.batch_size = 56

                self.learning_rate = 5e-3
                self.lr_for_pi = 5e-3  # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 1e-3
                self.lr_for_scale = 1e-3  # 1e-2或者1e-3

        elif MODEL_NAME == "GT1_GT2_GT3":
            self.lr_for_pi = 1e-3
            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3
            self.learning_rate = 1e-3
            self.batch_size = 64
            self.EPOCH_NUM = 20
        elif MODEL_NAME == "GT2_GT3_EMD":
            if self.ARTIFICIAL:
                self.EPOCH_NUM = 25
                self.batch_size = 40

                self.learning_rate = 5e-3
                self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 5e-3
                self.lr_for_scale = 1e-2  #1e-2或者1e-3
            else:
                self.EPOCH_NUM = 20
                self.batch_size = 64

                self.learning_rate = 5e-3
                self.lr_for_pi = 5e-3  # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 5e-3
                self.lr_for_scale = 1e-3  # 1e-2或者1e-3

        elif MODEL_NAME == "GT1_GT2_GT3_EMD":
            self.EPOCH_NUM = 25
            self.batch_size = 48

            self.learning_rate = 5e-3
            self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3  #1e-2或者1e-3

        else:
        ######*********CHECK CAREFULLY***********########

            # Datasets split & lr
            if self.ARTIFICIAL and self.Conv_1_4:
                self.EPOCH_NUM = 25
                self.batch_size = 48

                self.learning_rate = 5e-3
                self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_shape = 5e-3
                self.lr_for_scale = 5e-3  #1e-2或者1e-3

                # For scheduler
                self.StepLR_step_size = 5
                self.StepLR_gamma = 0.92

            # for Real data + MB_MDN.
            if not self.ARTIFICIAL and self.Conv_1_4:

                self.EPOCH_NUM = 25

                self.batch_size = 40  # 40比较好了
                self.learning_rate = 1e-3                            # 1e-3, 5e-3

                self.lr_for_pi = 5e-3    # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

                self.lr_for_mu = 1e-2    # 给mu单独设置learning rate   # 1e-2和5e-2都可以
                # lr_for_mu = 1e-3    # 给mu单独设置learning rate   # 1e-2和5e-2都可以
                self.lr_for_sigma = 5e-3  # 影响比较大。5e-3>1e-3
                # lr_for_sigma = 1e-3  # 影响比较大。5e-3>1e-3

                #### Weibull
                self.lr_for_shape = 5e-3
                self.lr_for_scale = 5e-3

                # For scheduler
                self.StepLR_step_size = 5
                self.StepLR_gamma = 0.92


        #### For W loss
        # learning_rate = 1e-3
        # lr_for_mu = 1e-2   # 给mu单独设置learning rate
        # lr_for_sigma = 1e-4

        self.lr_decay = 0.95      # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = 5e-2  # for optimizer (regularization)   # 1e-3比较好


        # Training data的粒度，画mdn时会使用到，
        self.SCALE = 1

        ################### DATA PATH ######################

        # Data path
        if self.ARTIFICIAL:
            if not self.W_GT3:
                self.train_path = os.path.join(self.data_root, "data/artificial_train_v2_noise=" + str(self.noise_pct) + "_seed=" + str(self.seed))
            else:
                self.train_path = os.path.join(self.data_root, "data/artificial_train_v3_noise=" + str(self.noise_pct) + "_seed=" + str(self.seed))

            self.target_path_metric = os.path.join(self.data_root, "data/artificial_targets_v2_" + "noise=" + str(self.noise_pct))
            self.target_path_loss = os.path.join(self.data_root, "data/artificial_targets_v2_" + "noise=" + str(self.noise_pct) + "_ls_T")

            # self.NLL_metric_path = os.path.join(self.data_root, "data/GT_metric/NLL_metric_GT_Tgt=1_e30_all_" + "artificial_targets_v2_" + "noise=" + str(
            #                         self.noise_pct) + "_seed=" + str(self.seed) + ".csv")

            # output: Details of validatiaon test
            self.vali_root = os.path.join(self.data_root, "data/figs/vali_syn/vali_noise=" + str(self.noise_pct) + "_seed=" + str(self.seed))

            # Visdom env
            self.env_str = "synthetic_seed="+str(self.seed)

            # Tensorboard
            self.logs_str = "synthetic_seed=" + str(self.seed)
            self.vali_main_tag_str = "vali_metric/synthetic_seed=" + str(self.seed)

        else:
            # 使用SA预测的结果
            # train_path = "../data/train_300_uniq_all"
            # 使用NN infer的结果
            if not self.W_GT3:
                self.train_path = os.path.join(self.data_root, "data/train_300_uniq_all_seed=" + str(self.seed))
            else:
                self.train_path = os.path.join(self.data_root, "data/train_300_uniq_all_v3_seed=" + str(self.seed))
            self.target_path_metric = os.path.join(self.data_root, "data/targets_all")

            if self.USE_DA:
                self.target_path_loss = os.path.join(self.data_root, "data/targets_all_5_DA_P=0.5_N_c=2")
            else:
                self.target_path_loss = os.path.join(self.data_root, "data/targets_all")

            # target_path_loss = "../data/targets_all_5_DA_P=0.5_N_c=2"

            # self.NLL_metric_path = os.path.join(self.data_root, "data/GT_metric/NLL_metric_GT_Tgt=1_e30_seed="+str(self.seed)+".csv")

            # output: Details of validatiaon test
            self.vali_root = os.path.join(self.data_root, "data/figs/vali_real/vali_seed="+ str(self.seed))

            # Visdom env
            self.env_str = "real_seed="+str(self.seed)

            # Tensorboard
            self.logs_str = f"real_seed=" + str(self.seed)
            self.vali_main_tag_str = "vali_metric/real_seed=" + str(self.seed)

        # Plot
        # params of GT2 or GT3
        self.params_path_2 = os.path.join(self.data_root, "data/SA_PT/params_seed=" + str(self.seed) + ".csv")
        self.params_path_3 = os.path.join(self.data_root, "data/SA_PT/params_GT3_seed=" + str(self.seed) + ".csv")

        # Target data for loss calculation
        self.TARGET = 1
        self.arr_flag = False        # whether drop uniform data
        # arr_path = r"../data/arr_selected/arr_targets_5_DA_P=0.5_N_c=3_K=3.npy"
        # arr_path = r"shuffled_indices.npy"
        self.arr_path = os.path.join(self.data_root, "data_handler/idx_GT2_better.pickle")
        # arr_path = r"../data/arr_selected/arr_targets_5_DA_P=0.5_N_c=3_K=2.npy"

        # cluster assignment
        self.N_CLUSTER = 7
        self.CHOSEN_CLUSTER = 2
        self.cluster_assign_path = os.path.join(self.data_root, "data/cluster_assignment_N="+str(self.N_CLUSTER) + "CHOSEN="+str(self.CHOSEN_CLUSTER)+".csv")

        # target_revenue
        self.target_revenue_path = os.path.join(self.data_root, "data/revenue/target_ER.csv")
