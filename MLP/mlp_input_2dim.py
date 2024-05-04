from common import *
from utils import *
from mydataset import *

import my_collate_fn
from my_collate_fn import my_collate_fn_3

reload(loss)    # 必须reload！！
reload(plot)
reload(my_collate_fn)

from Config.config_MDN import DefaultConfig

######*********CHECK CAREFULLY***********########
# foreach ($i in 3,31,62,204,223,407,508,626) { D:\Anaconda\python.exe "mlp_wGT3_wEMD.py" --seed $i --MODEL_NAME "GT3+EMD" }
# foreach ($i in 4,31,35,204,407,66,508,512) { D:\Anaconda\python.exe "mlp_wGT3_wEMD.py" --seed $i --MODEL_NAME "GT3+EMD" }

# 创建一个解析器对象
parser = argparse.ArgumentParser(description="Run the script with a specific seed.")
# 添加seed参数
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility.")
parser.add_argument("--MODEL_NAME", type=str, required=True, help="Model name.")
# 解析命令行参数
args = parser.parse_args()

seed = args.seed
MODEL_NAME = args.MODEL_NAME

# seed = 62
# MODEL_NAME = "GT3+EMD"
if MODEL_NAME == "GT1+GT2" or "GT1_GT2_sft_bidfee" or "GT1_GT2_sft_bidinc":
    INPUT_LIST = [1,2]
elif MODEL_NAME == "GT1+GT3":
    INPUT_LIST=[1,3]
elif MODEL_NAME == "GT1+EMD":
    INPUT_LIST=[1,4]
elif MODEL_NAME == "GT2+GT3":
    INPUT_LIST=[2,3]
elif MODEL_NAME == "GT2+EMD":
    INPUT_LIST=[2,4]
elif MODEL_NAME == "GT3+EMD":
    INPUT_LIST=[3,4]
else:
    assert "Wrong Model Name!"

opt = DefaultConfig(MODEL_NAME=MODEL_NAME,seed=seed)

lr_for_pi = opt.lr_for_pi # opt.lr_for_pi
lr_for_shape = opt.lr_for_shape
lr_for_scale = opt.lr_for_scale
learning_rate = opt.learning_rate

MB_MDN_Flag = opt.Conv_1_4

batch_size = opt.batch_size
EPOCH_NUM = opt.EPOCH_NUM
q=opt.q

######*********CHECK CAREFULLY***********########
class Conv_block_4(nn.Module):

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, ch_out=1,kernel_size=9, stride=3, init_weight=False) -> None:
        super().__init__()

        self.kernel_size = (2,kernel_size)
        self.stride = (2,stride)
        self.ln_in = int((300-kernel_size)/stride+1)
        # self.ln_in = 84

        self.ac_func = nn.Softplus()

        self.conv = nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=self.kernel_size, stride=self.stride, padding=0, dilation=(1,1))

        self.BN_aff2 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)      # works better

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        # Conv=>BN=>AC
        x = self.conv(x)
        # print(x.shape)
        # 方法一：
        # x = torch.squeeze(x,dim=2)
        # x = self.ac_func(self.BN_aff1(x))

        # 方法二：works better
        x = torch.flatten(x,start_dim=1)
        x = self.ac_func(self.BN_aff2(x))
        # x = self.ac_func(self.BN_aff1(x))
        return x

class Conv_1_4(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=9, stride=3) -> None:
        super().__init__()

        self.kernel_size = (2,kernel_size)
        self.stride = (2,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.BN1 = nn.BatchNorm1d(num_features=2,affine=True)
        # self.IN1 = nn.InstanceNorm1d(num_features=3,affine=True)
        self.layer_pi = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_scale = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_shape = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)

        self.ac_func = nn.Softplus()

        self.z_pi = nn.Sequential(
            nn.Linear(self.ln_in, n_gaussians),
            nn.Softmax(dim=1)           # dim=0是B, dim=1才是feature
        )
        self.z_scale = nn.Linear(self.ln_in, n_gaussians)
        self.z_shape = nn.Linear(self.ln_in, n_gaussians)

    def forward(self, x):

        x = self.BN1(x)
        # x = self.IN1(x)
        x = torch.unsqueeze(x,dim=1)                     # torch.Size([B, 1, 3, 300])

        x_pi = self.layer_pi(x)
        x_scale = self.layer_scale(x)
        x_shape = self.layer_shape(x)

        pi = self.z_pi(x_pi)
        scale = torch.exp(self.z_scale(x_scale))
        scale = torch.clamp(scale,1e-4)
        shape = torch.exp(self.z_shape(x_shape))
        shape = torch.clamp(shape,1e-4)

        return pi,scale,shape

def get_params(mlp,opt):
    """
    Set learning rates for different layers and return params for training
    Args:
        mlp:
        opt:

    Returns: params

    """
    shape_params = list(map(id, mlp.z_shape.parameters()))
    scale_params = list(map(id, mlp.z_scale.parameters()))
    pi_params = list(map(id, mlp.z_pi.parameters()))

    params_id = shape_params + scale_params + pi_params

    base_params = filter(lambda p: id(p) not in params_id, mlp.parameters())
    params = [{'params': base_params},  # 如果对某个参数不指定学习率，就使用最外层的默认学习率
            {'params': mlp.z_pi.parameters(), 'lr': lr_for_pi},
            {'params': mlp.z_shape.parameters(), 'lr': lr_for_shape},
            {'params': mlp.z_scale.parameters(), 'lr': lr_for_scale}]

    return params

def trainer(train_loader, val_loader, test_loader, mlp, opt, device):
    """
    Main body of a training process. Called by objective function
    Args:
        train_loader:
        val_loader:
        mlp:
        opt:    params to be held
        device:

    Returns: performance(avg NLL in last 5 epoch) in validation set

    """
    params = get_params(mlp,opt)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.StepLR_step_size, gamma=opt.StepLR_gamma)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    total_train_step = 0
    min_loss = np.inf

    for epoch in range(EPOCH_NUM):
        mlp.train()
        epoch_train_loss = 0

        for batch_id, data in enumerate(train_loader):
            input_data, _, target_loss, setting = data
            # Do the inference
            input_data = input_data.to(device)
            target_loss = target_loss.to(device)

            pi, mu, sigma = mlp(input_data)

            loss = loss_fn_wei(pi, mu, sigma, target_loss, opt.N_gaussians, opt.TARGET, opt.SAFETY, device)

            epoch_train_loss += loss.detach().cpu()
            optimizer.zero_grad()
            loss.backward()   # Compute gradient(backpropagation).
            clip_grad_norm_(mlp.parameters(), max_norm=20, norm_type=2)  # 使用第二种裁剪方式
            optimizer.step()  # Update parameters.

            total_train_step += 1
        scheduler.step()
        # print(f"========== IN EPOCH {epoch} the total loss is {epoch_train_loss}, ==========")

        ########### Vali
        mlp.eval()
        with torch.no_grad():
            total_vali_metric = validate(model, val_loader, opt.N_gaussians, opt.MIN_LOSS, device, detail=True)

            # if opt.PRINT_VAL:
            #     print(f"========== IN EPOCH {epoch} the vali metric is {total_vali_metric} | min_loss = {min_loss} ==========")

        # 记录最小的total_vali_metric，并且保存此时的模型
        if total_vali_metric < min_loss:
            min_loss = total_vali_metric
            model_path = get_MDN_save_path(opt.ARTIFICIAL, seed, opt.net_root_path, opt.noise_pct, MODEL_NAME)
            hyperparameters = {
                'model_name': MODEL_NAME,
                'N_gaussians':opt.N_gaussians,
                'learning_rate': opt.learning_rate,
                'lr_for_shape': opt.lr_for_shape,
                'lr_for_scale': opt.lr_for_scale,
                'lr_for_pi': opt.lr_for_pi,
                'batch_size': opt.batch_size,
                'epoch_when_saved': epoch
            }
            save_checkpoint(mlp, hyperparameters, model_path)

    ########### Do Test
    mlp.eval()
    with torch.no_grad():
        # The validation
        if q == 1:
            model_path = get_MDN_save_path(opt.ARTIFICIAL, seed, opt.net_root_path, opt.noise_pct, MODEL_NAME)
            mlp, _ = load_checkpoint(model_path, mlp)
            total_test_metric_NLL = validate(mlp, test_loader, opt.N_gaussians, opt.MIN_LOSS, device, detail=True)
            total_test_metric_KL = validate_KL(mlp, test_loader, opt.N_gaussians, opt.MIN_LOSS, device)

        mlp.train()

        print(f"NN prediction in NLL = {total_test_metric_NLL}")
        print(f"NN prediction in KL = {total_test_metric_KL}")

        return total_test_metric_NLL, total_test_metric_KL

pred_metric_NLL = []
pred_metric_KL = []
metric_pd = pd.DataFrame()

if __name__ == '__main__':

    setup_seed(seed)
    print(f"========== MODEL_NAME = {MODEL_NAME} | INPUT_LIST = {INPUT_LIST} ==========")
    print(f"========== seed = {seed} ==========")
    print(f"========== seed = {seed} ==========")
    print(f"========== seed = {seed} ==========")

    # # 生成当前时间的时间戳, 作为画图的区分
    # timestamp = int(time.time())
    # time_str = str("_")+time.strftime('%y%m%d%H%M%S', time.localtime(timestamp))
    # print(f"time_str = {time_str}")
    # env_str = MODEL_NAME + "_seed=" + str(seed) + time_str

    # viz = visdom.Visdom(env=env_str)
    # writer = SummaryWriter(log_dir="logs-MLP/" + opt.logs_str, flush_secs=60)   # opt.logs_str是log的文件夹名字

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = myDataset(opt.train_path, opt.target_path_metric, opt.target_path_loss, opt.data_key_path)

    train_idx, val_idx, test_idx = get_data_idx_from_MODEL_NAME(dataset, MODEL_NAME, opt)

    my_collate_fn = functools.partial(my_collate_fn_3, INPUT_LIST=INPUT_LIST)
    train_loader,val_loader,test_loader = get_data_loader(dataset, opt.batch_size, train_idx, val_idx, test_idx, my_collate_fn)

    model = Conv_1_4(opt.N_gaussians).to(device)  # put your model and data on the same computation device.
    if not opt.ARTIFICIAL:
        if (seed == 626 and MODEL_NAME == "GT1+GT2")\
            or (seed == 204 and MODEL_NAME == "GT3+EMD")\
            or (seed == 508 and MODEL_NAME == "GT2+EMD") \
            or (seed == 626 and MODEL_NAME == "GT1+GT3")\
            or (seed == 626 and MODEL_NAME == "GT2+GT3") :
            model_path = get_MDN_save_path(opt.ARTIFICIAL, seed, opt.net_root_path, opt.noise_pct, MODEL_NAME + "_init")
            hyperparameters = {
                'model_name': MODEL_NAME+"_init_",
                'N_gaussians': opt.N_gaussians,
                'learning_rate': learning_rate,
                'lr_for_shape': lr_for_shape,
                'lr_for_scale': lr_for_scale,
                'lr_for_pi': lr_for_pi,
                'batch_size': batch_size,
            }
            model, _ = load_checkpoint(model_path, model)
            # save_checkpoint(model, hyperparameters, model_path)
    if opt.ARTIFICIAL:
        if (seed == 66 and MODEL_NAME == "GT1+GT3") or (seed == 35 and MODEL_NAME == "GT1+GT3")\
            or (seed == 204 and MODEL_NAME == "GT2+GT3"):
            model_path = get_MDN_save_path(opt.ARTIFICIAL, seed, opt.net_root_path, opt.noise_pct, MODEL_NAME + "_init")
            hyperparameters = {
                'model_name': MODEL_NAME + "_init_",
                'N_gaussians': opt.N_gaussians,
                'learning_rate': learning_rate,
                'lr_for_shape': lr_for_shape,
                'lr_for_scale': lr_for_scale,
                'lr_for_pi': lr_for_pi,
                'batch_size': batch_size,
            }
            model, _ = load_checkpoint(model_path, model)
            # save_checkpoint(model, hyperparameters, model_path)
            # print(f"INIT DONE")
            # print(f"{model_path}")

    # if opt.DRAW_VAL:
    #     model.eval()
    #     with torch.no_grad():
    #         total_vali_metric = validate(model, val_loader, opt.N_gaussians, opt.MIN_LOSS, device, detail=True)

    model.train()

    # For training , cal the exp revenue
    performance = trainer(train_loader, val_loader, test_loader, model, opt, device)

    # Save metric into a file
    save_performance(opt.ARTIFICIAL, seed, MODEL_NAME, performance, opt.metric_list)

    print(f"========== seed = {seed} ==========")
    print(f"========== MODEL_NAME = {MODEL_NAME} ==========")
    print(f"========== METRIC: {opt.metric_list}==========")
    print(f"{performance}")

