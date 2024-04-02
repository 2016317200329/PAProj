import random
import torch.utils.data
from mydataset import *
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
# from tensorboardX import SummaryWriter
import optuna
from importlib import reload

######### Ray Tune

from MLP.Config import config
import loss
import plot
import my_collate_fn

reload(config)  # 必须reload！！
reload(loss)    # 必须reload！！
reload(plot)
reload(my_collate_fn)

from my_collate_fn import my_collate_fn_3

from MLP.Config.config import DefaultConfig
# from models import MLP_1_1
from loss import loss_fn_wei
from loss import validate

opt = DefaultConfig()

def setup_seed(seed):
    """
    Set seed
    Args:
        seed:
    """
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

# # trainable params
# parameters = dict(
#     lr=[.01,.001],
#     batch_size = [100,1000],
#     shuffle = [True,False]
# )
# #创建可传递给product函数的可迭代列表
# param_values = [v for v in parameters.values()]
# #把各个列表进行组合，得到一系列组合的参数
# #*号是告诉乘积函数把列表中每个值作为参数，而不是把列表本身当做参数来对待
# for lr,batch_size,shuffle in product(*param_values):
#     comment  = f'batch_size={batch_size}lr={lr}shuffle={shuffle}'
#     #这里写你调陈的主程序即可
#     print(comment)

def save_data_idx(dataset,opt):
    """
    因为objective function会被执行很多次，所以这里先保存一下idx，使得所有objective在同一组dataset上进行。
    然后在tuning时，会在shuffle_time组不同的dataset split上进行，用以取平均
    Args:
        dataset:
        opt:
        shuffle_time: The num of list of index to be generated.
    """
    shuffled_indices = []
    # 使用全部的data
    if not opt.arr_flag:
        DATA_len = dataset.__len__()
        shuffled_indices = np.random.permutation(DATA_len)

    # 使用指定的data
    if opt.arr_flag:
        shuffled_indices = np.load(opt.arr_path)
        # DATA_len = len(shuffled_indices)
        np.random.shuffle(shuffled_indices)

    return shuffled_indices

def get_data_idx(shuffled_indices,opt):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        opt:

    Returns:
    """

    DATA_len = len(shuffled_indices)

    train_idx = shuffled_indices[:int(opt.train_pct * DATA_len)]
    tmp = int((opt.train_pct + opt.vali_pct) * DATA_len)
    val_idx = shuffled_indices[int(opt.train_pct * DATA_len):tmp]
    test_idx = shuffled_indices[tmp:]

    return train_idx,val_idx,test_idx

def get_data_loader(dataset, shuffled_indices, opt):
    """
    To get dataloader according to shuffled 'shuffled_indices'
    Args:
        dataset:
        shuffled_indices:
        opt:

    Returns:

    """
    train_idx,val_idx,test_idx = get_data_idx(shuffled_indices,opt)

    train_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(train_idx), collate_fn=my_collate_fn_3)
    val_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx), collate_fn=my_collate_fn_3)
    # 注意test_loader的batch size
    test_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(test_idx), collate_fn=my_collate_fn_3)

    return train_loader,val_loader,test_loader

def get_params(mlp,config,opt):
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
            {'params': mlp.z_pi.parameters(), 'lr': opt.lr_for_pi},
            {'params': mlp.z_shape.parameters(), 'lr': opt.lr_for_shape},
            {'params': mlp.z_scale.parameters(), 'lr': opt.lr_for_scale}]

    return params

# Not Sequential

class Conv_block_9(nn.Module):

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, ch_out=1,kernel_size=9, stride=3, dilation=1,init_weight=True) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-kernel_size-(kernel_size-1)*(dilation-1))/stride+1)

        self.ac_func = nn.Softplus()

        self.conv = nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=self.kernel_size, stride=self.stride, padding=0, dilation=(1,dilation))

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
        return x

class Conv_1_9(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=9, stride=3, dilation = 1) -> None:
        super().__init__()

        self.dilation = dilation
        self.ln_in = int((300-9)/3+1)
        self.ln_scale = int((300-9)/3+1)
        self.ln_shape = int((300-kernel_size-(dilation-1)*(kernel_size-1))/stride+1)

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)
        self.BN_aff0 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)
        self.BN_aff1 = nn.BatchNorm1d(num_features=self.ln_scale,affine=True)
        self.BN_aff2 = nn.BatchNorm1d(num_features=self.ln_shape,affine=True)      # works better

        self.layer_pi = Conv_block_9(ch_out=1,kernel_size=9,stride=3)
        self.layer_scale = Conv_block_9(ch_out=1,kernel_size=9,stride=3)
        self.layer_shape = Conv_block_9(ch_out=1,kernel_size=kernel_size,stride=stride,dilation=self.dilation)

        self.ac_func = nn.Softplus()

        self.z_pi = nn.Sequential(
            nn.Linear(self.ln_in, n_gaussians),
            nn.Softmax(dim=1)           # dim=0是B, dim=1才是feature
        )
        self.z_scale = nn.Linear(self.ln_scale, n_gaussians)
        self.z_shape = nn.Linear(self.ln_shape, n_gaussians)

    def forward(self, x):

        x = self.BN1(x)
        x = torch.unsqueeze(x,dim=1)                     # torch.Size([B, 1, 3, 300])

        x_pi = self.ac_func(self.BN_aff0(self.layer_pi(x)))
        x_scale = self.ac_func(self.BN_aff1(self.layer_scale(x)))

        x_shape = self.ac_func(self.BN_aff2(self.layer_shape(x)))

        pi = self.z_pi(x_pi)
        scale = torch.exp(self.z_scale(x_scale))
        scale = torch.clamp(scale,1e-4)
        shape = torch.exp(self.z_shape(x_shape))
        shape = torch.clamp(shape,1e-4)

        return pi,scale,shape


def trainer(train_loader, val_loader, test_loader, mlp, config, opt, device):
    """
    Main body of a training process. Called by objective function
    Args:
        train_loader:
        val_loader:
        mlp:
        config: params to be tuned
        opt:    params to be held
        device:

    Returns: performance(avg NLL in last 5 epoch) in validation set

    """
    params = get_params(mlp,config,opt)
    optimizer = torch.optim.Adam(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.StepLR_step_size, gamma=opt.StepLR_gamma)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)

    total_train_step = 0
    EPOCH_NUM = 60

    for epoch in range(EPOCH_NUM):
        mlp.train()
        epoch_train_loss = 0

        for batch_id, data in enumerate(train_loader):
            input_data, _, target_loss, setting, _ = data
            # Do the inference
            input_data = input_data.to(device)
            target_loss = target_loss.to(device)
            pi, mu, sigma = mlp(input_data)

            # loss = loss_fn_v2(pi, mu, sigma, target_loss, opt.N_gaussians, opt.TARGET, opt.SAFETY, device)
            loss = loss_fn_wei(pi, mu, sigma, target_loss, opt.N_gaussians, opt.TARGET, opt.SAFETY, device)
            epoch_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()   # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.

            total_train_step += 1
        scheduler.step()

        ########### Do validation
        mlp.eval()
        with torch.no_grad():
            total_test_metric, GT_metric = validate(mlp, test_loader, opt.N_gaussians, opt.MIN_LOSS, device)
            GT_metric_1 = GT_metric[0, 0]
            metric_diff = GT_metric_1 - total_test_metric

    # 和GT-1的差值：越大越好
    return metric_diff.cpu().numpy()


class Objective:
    def __init__(self, dataset, opt):
        # Hold this implementation specific arguments as the fields of the class.
        self.dataset = dataset
        self.opt = opt
        # Hold the data split!!
        self.shuffled_indices = save_data_idx(dataset,opt)

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        config = {
            'kernel_size': trial.suggest_categorical('kernel_size',[15,20,30]),
            'stride': trial.suggest_categorical('stride',[3,6,9]),
            'dilation': trial.suggest_categorical('dilation',[3,6])
        }

        train_loader, val_loader, test_loader = get_data_loader(self.dataset, self.shuffled_indices, self.opt)

        TRIALS = 3  # 跑3次取平均表现，减小init方式的影响
        performance = []
        for k in range(TRIALS):
            model = Conv_1_9(self.opt.N_gaussians,kernel_size=config['kernel_size'],stride=config['stride'],dilation=config['dilation']).to(device)  # put your model and data on the same computation device.
            performance.append(trainer(train_loader, val_loader, test_loader, model, config, self.opt, device))

        return np.mean(performance)

######## main #########

# 固定seed
setup_seed(6)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### optuna-dashboard sqlite:///study_name
study_name = 'dila'  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

# 里面所有的组合被cover之后会自动stop
sampler = optuna.samplers.GridSampler(search_space={
            'kernel_size':[15,20,30],
            'stride':[3,6,9],
            'dilation':[3,6]
        })

# sampler = optuna.samplers.QMCSampler()
pruner = optuna.pruners.NopPruner()
study = optuna.create_study(study_name=study_name,direction='maximize',storage=storage_name,load_if_exists=True,sampler=sampler,pruner=pruner)

dataset = myDataset(opt.train_path, opt.target_path_metric, opt.target_path_loss, opt.data_key_path,
                    opt.NLL_metric_path)

# n_trials表示会尝试n_trials次params的组合方式
# objective会被调用这么多次
study.optimize(Objective(dataset,opt), n_trials=50,show_progress_bar=True)

results = study.trials_dataframe(attrs=("number", "value", "params", "state"))

# 输出最优的超参数组合和性能指标
print('Best hyperparameters: {}'.format(study.best_params))
print('Best performance: {:.6f}'.format(study.best_value))
