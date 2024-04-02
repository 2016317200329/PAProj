import os
class BaseConfig(object):
    def __init__(self):
        # Input Choice
        self.GT_1 = 0
        self.GT_2 = 1
        self.EMD = 2

        self.train_pct = 0.7
        self.vali_pct = 0.1
        self.test_pct = 0.2

        # NLL metric
        self.MIN_LOSS = 1e-30

        self.SAFETY = 1e-30

        self.data_root = "D:/Desktop/PROJ/PAProj"

        # data keys
        self.data_key_path = os.path.join(self.data_root,"data/target_datakey_all.csv")
        # Net path
        self.net_root_path = os.path.join(self.data_root, "MLP/net_saved/")

        self.fig_path = os.path.join(self.data_root, "data/figs")

        self.metric_list = ['NLL','KL-D']
