import config_MDN

class DefaultConfig(config_MDN.DefaultConfig):
    def __init__(self):
        super().__init__()  # 调用父类的__init__方法以继承基础配置

        self.ARTIFICIAL = False       # If True, use the syns dataset; If False, use the real dataset.
        # Datasets split & lr
        if self.ARTIFICIAL:
            self.EPOCH_NUM = 25
            self.batch_size = 48

            self.learning_rate = 5e-3
            self.lr_for_pi = 5e-3     # 给pi单独设置learning rate   # 影响其次,1e-3>5e-2

            self.lr_for_mu = 1e-2     # 给mu单独设置learning rate   # 1e-2和5e-2都可以
            self.lr_for_sigma = 1e-3  # 影响比较大。5e-3>1e-3

            self.lr_for_shape = 5e-3
            self.lr_for_scale = 5e-3  #1e-2或者1e-3

            # For scheduler
            self.StepLR_step_size = 5
            self.StepLR_gamma = 0.92

        # for Real data + MB_MDN.
        if not self.ARTIFICIAL:

            self.EPOCH_NUM = 25

            self.batch_size = 0  # 40比较好了
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

if __name__ == '__main__':
    config = DefaultConfig()
    print(config.batch_size)
