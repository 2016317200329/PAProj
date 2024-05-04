import sys
sys.path.insert(0, 'D:\\Desktop\\PROJ\\PAProj')
from common import *
from utils import *
from mydataset import *

print(os.getcwd())

import mydataset_GT2
from functorch import vmap
from mydataset_GT2 import myDataset

from GT_model.GT_2 import SA_for_PT_funcs_delta_eq1 as SA_for_PT_funcs_delta_eq1_GT2
from GT_model.GT_3 import SA_for_PT_funcs_delta_eq1 as SA_for_PT_funcs_delta_eq1_GT3

from torch.cuda.amp import GradScaler,autocast
from models import MLP_2_1
from models import MLP_GT3_2

scaler = GradScaler(backoff_factor = 0.1)

def get_model_opt(MODEL_NAME):
    if MODEL_NAME == 'InferNet_GT2':
        return {
            'model' : MLP_2_1(),
            'opt': config_GT2.DefaultConfig()
        }
    elif MODEL_NAME == 'InferNet_GT3':
        return {
            'model': MLP_GT3_2,
            'opt': config_GT3.DefaultConfig()
        }
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

# 创建一个解析器对象
parser = argparse.ArgumentParser(description="Run the script with a specific seed.")
# 添加seed参数
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility.")
parser.add_argument("--MODEL_NAME", type=str, required=True, help="Model name.")

# 解析命令行参数
args = parser.parse_args()
seed = args.seed
MODEL_NAME = args.MODEL_NAME

MODEL_LIST = ["InferNet_GT2","InferNet_GT3"]

if MODEL_NAME not in MODEL_NAME:
    print(f"WRONG MODEL NAME of {MODEL_NAME}")
    exit(0)

######################### Carefully Check ####################
from Config import config_GT2
from Config import config_GT3

opt = get_model_opt(MODEL_NAME)['opt']

## 设置：sys.path.insert(0, 'D:\\Desktop\\PROJ\\PAProj')
# conda activate
# D:
# cd D:\Desktop\PROJ\PAProj
# foreach ($i in 3,31,62,204,223,407,508,626) { D:\Anaconda\python.exe "MLP\NN_params_infer_GT2.py" --seed $i --MODEL_NAME "InferNet_GT2" }

EPOCH_NUM = opt.EPOCH_NUM

######################### Carefully Check ####################

import my_collate_fn
from my_collate_fn import my_collate_fn_GT2

reload(config_GT2)  # 必须reload！！
reload(config_GT3)  # 必须reload！！
reload(loss)    # 必须reload！！
reload(plot)
reload(mydataset_GT2)
reload(SA_for_PT_funcs_delta_eq1_GT3)
reload(SA_for_PT_funcs_delta_eq1_GT2)
reload(my_collate_fn)

total_train_step = 0
total_test_step = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

bound_alpha = torch.tensor([-0.3,0.3],device=device)
bound_labda = torch.tensor([0.01,18],device=device)

def get_params_GT2(mlp,opt):
    """
    Set learning rates for different layers and return params for training
    Args:
        mlp:
        opt:

    Returns: params

    """
    alpha_params = list(map(id, mlp.block_alpha.parameters()))
    labda_params = list(map(id, mlp.block_labda.parameters()))

    params_id = alpha_params + labda_params
    other_params = filter(lambda p: id(p) not in params_id, mlp.parameters())
    params = [{'params': other_params},
                {'params': mlp.block_alpha.parameters(), 'lr': opt.lr_for_alpha},
                {'params': mlp.block_labda.parameters(), 'lr': opt.lr_for_labda}]

    return params

def get_params_GT3(mlp,opt):
    """
    Set learning rates for different layers and return params for training
    Args:
        mlp:
        opt:

    Returns: params

    """
    alpha_params = list(map(id, mlp.block_alpha.parameters()))

    params_id = alpha_params
    other_params = filter(lambda p: id(p) not in params_id, mlp.parameters())
    params = [{'params': other_params},
                {'params': mlp.block_alpha.parameters(), 'lr': opt.lr_for_alpha}]

    return params


############# Tensor #############:
def C(t,b):
    return 0.2*t*b

def f_ts_GT2(x, alpha, device):
    # -alpha*x不能超过85，否则overflow
    # y1 = torch.clamp(-alpha*x,max=torch.tensor(85.,device=device))#.to(device)       # torch.exp(x), x<85
    y2 = (1-torch.exp(torch.clamp(-alpha*x,max=torch.tensor(85.,device=device))))  #.to(device)

    return y2
def f_ts_GT3(x, alpha, device):
    """

    Args:
        x:
        alpha:
        device:

    Returns: y2= 1-log(-alpha*x), and restrict the return value within a range.

    """
    # -alpha*x不能超过85，否则overflow
    # y1 = torch.clamp(-alpha*x,max=torch.tensor(85.,device=device))#.to(device)       # torch.exp(x), x<85
    y2 = (1-torch.exp(torch.clamp( -alpha*x, max=torch.tensor(85.,device=device))))  #.to(device)

    return y2


def f_2_ts_GT2(x, alpha,device):
    # -alpha*x不能超过85，否则overflow
    # y1 = torch.min(-alpha*x,torch.tensor(85.)).to(device)       # torch.exp(x), x<85
    y2 = (torch.exp(torch.min(-alpha*x,torch.tensor(85.)).to(device)))#.to(device)

    return y2

# root = U_{t-1}
def f_Equi_ts_GT2(t, T, v, d, b, alpha, labda, device):

    tmp = (v-d*(t-1)-C((t-2),b) - b).to(device)

    if (tmp>0):

        root = (labda*f_ts_GT2(C(t-2,b),alpha,device) + f_ts_GT2(((v-d*(t-1)-C((t-2),b) - b)),alpha,device)) / (labda*f_ts_GT2(C(t-2,b)+b,alpha,device) + f_ts_GT2(((v-d*(t-1)-C((t-2),b) - b)),alpha,device))

    else:

        root = (1- f_2_ts_GT2(-(v-(t-1)*d-b),alpha,device)) / (f_2_ts_GT2(b,alpha,device) - f_2_ts_GT2(-(v-(t-1)*d-b),alpha,device))

    assert not torch.isnan(root.detach()),f"root = {root.detach()} and t, T, v, d, b, alpha, labda = {t, T, v, d, b, alpha.detach(), labda.detach()}"
    return root

# 返回log(root)
def f_Equi_ts_log_GT2(t, T, v, d, b, alpha, labda, device):

    tmp = (v-d*(t-1)-C((t-2),b) - b).to(device)

    if (tmp>0):

        root = torch.log(torch.abs((labda*f_ts_GT2(C(t-2,b),alpha,device) + f_ts_GT2(((v-d*(t-1)-C((t-2),b) - b)),alpha,device)))) \
            - torch.log(torch.abs((labda*f_ts_GT2(C(t-2,b)+b,alpha,device) + f_ts_GT2(((v-d*(t-1)-C((t-2),b) - b)),alpha,device))))

    else:
        # 分母!=0
        # root = torch.log(torch.abs((1- f_2_ts(-(v-(t-1)*d-b),alpha,device)))) \
        #     - (-alpha*b)- torch.log(torch.abs((1 - f_2_ts(-(v-(t-1)*d),alpha,device))))

        root = torch.log(torch.abs((1- f_2_ts_GT2(-(v-(t-1)*d-b),alpha,device)))) \
            - torch.log(torch.abs((f_2_ts_GT2(b,alpha,device) - f_2_ts_GT2(-(v-(t-1)*d-b),alpha,device))))

    assert not torch.isnan(root.detach()),f"root = {root.detach()}, tmp = {tmp}, and t, T, v, d, b, alpha, labda = {t, T, v, d, b, alpha.detach(), labda.detach()}"
    assert not torch.isinf(root.detach()),f"root = {root.detach()}, tmp = {tmp}, and t, T, v, d, b, alpha, labda = {t, T, v, d, b, alpha.detach(), labda.detach()}"

    return root


def f_Equi_ts_log_vmap_GT2(t, v, d, b, alpha, labda):       # , device
    root = torch.where((v-d*(t-1)-C((t-2),b) - b)>0,
                        torch.log(torch.abs(
                                (labda*f_ts_GT2(C(t-2,b),alpha,device) + f_ts_GT2(((v-d*(t-1)-C((t-2),b) - b)),alpha,device)) )) \
                        - torch.log(torch.abs(
                                (labda*f_ts_GT2(C(t-2,b)+b,alpha,device) + f_ts_GT2(((v-d*(t-1)-C((t-2),b) - b)),alpha,device)) )),
                        torch.log(torch.abs(
                            (1- f_2_ts_GT2(-(v-(t-1)*d-b),alpha,device)) )) \
                        - torch.log(torch.abs((f_2_ts_GT2(b,alpha,device) - f_2_ts_GT2(-(v-(t-1)*d-b),alpha,device)) ))
                        )
    return root
############### Method 2 ################
def f_Equi_ts_log_vmap_GT3(t, v, d, b, alpha):
    root = torch.log( torch.abs( (f_ts_GT3(((v-d*(t-1) - b)),alpha,device)) )) - torch.log(
        torch.abs((f_ts_GT3(((v-d*(t-1) - b)),alpha,device)) - f_ts_GT3(-b,alpha,device)))

    return root


def get_LEN_T_ts(v,b,d, max_T):
    """

    Args:
        v:
        b:
        d:
        max_T:

    Returns:
        LEN: Length of U vector
        T: max possible duration: max_T or theoratic T

    """

    if d == 0:          # fixed-price
        # T = np.inf
        T = max_T       # 最多计算target data需要的duration
    else:               # asc-price
        T = torch.floor((v - b) / d)

    LEN = max_T
    return LEN,T

def get_U_GT2_ts_log_vmap(LEN, T, v, d, b, alpha, labda, eps = 1e-30, device = device):

    # U: the prob. that someone offers a bid in t_th round
    # log(1) = 0.
    U_head = torch.tensor([0.,0.],requires_grad=True,device=device)  # u[0]用不到,u[1]=1保证auction至少1轮

    idx = torch.arange(2,LEN+1).to(device)

    # 方法二：使用vmap：
    v_extend = torch.expand_copy(v,size=idx.shape).to(device)    # 必须扩张到和idx同一个size否则报错
    b_extend = torch.expand_copy(b,size=idx.shape).to(device)
    d_extend = torch.expand_copy(d,size=idx.shape).to(device)
    alpha_extend = torch.expand_copy(alpha,size=idx.shape).to(device)
    labda_extend = torch.expand_copy(labda,size=idx.shape).to(device)

    f_Equi_vec = vmap(f_Equi_ts_log_vmap_GT2)  # [N, D], [N, D] -> [N]
    U_root = f_Equi_vec(idx, v_extend, d_extend, b_extend, alpha_extend, labda_extend)

    U_T_1 = torch.log(torch.tensor([eps],requires_grad=True,device=device))  # u[T+1]表示最后拍卖在T+1轮发生的概率为0
    U_i_log = torch.concat([U_head,U_root,U_T_1])                            # 减少原地修改，使用concat拼接

    U_i_log = U_i_log.to(device)

    # eps < U < 1-eps
    # 实际上在pytorch中 torch.tensor(1-eps)=1.
    # U = torch.clip(U_i,torch.tensor(eps,device=device),torch.tensor(1-eps,device=device))

    del idx
    del v_extend
    del b_extend
    del d_extend
    # torch.cuda.empty_cache()  # 会变慢不少
    return U_i_log

def get_U_GT3_ts_log_vmap(LEN, T, v, d, b, alpha, eps = 1e-30, device = device):

    # U: the prob. that someone offers a bid in t_th round
    # log(1) = 0.
    U_head = torch.tensor([0.,0.],requires_grad=True,device=device)  # u[0]用不到,u[1]=1保证auction至少1轮

    idx = torch.arange(2,LEN+1).to(device)

    # 方法二：使用vmap：
    v_extend = torch.expand_copy(v,size=idx.shape).to(device)    # 必须扩张到和idx同一个size否则报错
    b_extend = torch.expand_copy(b,size=idx.shape).to(device)
    d_extend = torch.expand_copy(d,size=idx.shape).to(device)
    alpha_extend = torch.expand_copy(alpha,size=idx.shape).to(device)

    # vmap 接受一个vector
    f_Equi_vec = vmap(f_Equi_ts_log_vmap_GT3)  # [N, D], [N, D] -> [N]
    U_root = f_Equi_vec(idx, v_extend, d_extend, b_extend, alpha_extend)    # U_root is log-value

    U_T_1 = torch.log(torch.tensor([eps],requires_grad=True,device=device))  # u[T+1]表示最后拍卖在T+1轮发生的概率为0
    U_i_log = torch.concat([U_head,U_root,U_T_1])                            # 减少原地修改，使用concat拼接

    U_i_log = U_i_log.to(device)

    # eps < U < 1-eps
    # 实际上在pytorch中 torch.tensor(1-eps)=1.
    # U = torch.clip(U_i,torch.tensor(eps,device=device),torch.tensor(1-eps,device=device))

    del idx
    del v_extend
    del b_extend
    del d_extend
    # torch.cuda.empty_cache()  # 会变慢不少
    return U_i_log



def loss_fn_InferNet_GT2(input_data, Alpha, Labda, Target_data, eps, device):
    # 注意：P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)
    for i in range(len(input_data)):

        # Get target data
        target = Target_data[i, :].long()

        # Solve for U from Equi. condt.
        # Get settings
        setting = input_data[i, :, 0:3]
        d = setting[0, 0]
        b = setting[0, 1]
        v = setting[0, 2]

        # 1.首先计算LEN,T
        LEN,T = get_LEN_T_ts(v,b,d,max(target))
        # 2.然后筛选target data：大于T的没必要，算不了

        idx = torch.nonzero(target)
        target_nonzero = target[idx]  #.reshape(1,-1).squeeze().long()      # int

        # U_i最大必须可以访问到U_i[LEN+1],因为计算P[LEN]需要用到U[LEN+1]
        # 方法二：直接求U_log，并且使用vmap进行并行运算
        U_log = get_U_GT2_ts_log_vmap(LEN, T, v, d, b, Alpha[i,:], Labda[i,:], eps = eps, device=device)
        # assert not (torch.any(torch.isnan(U_log.detach()))), f"U_log has NaN and U_log = {U_log.detach()}"

        U_log_cumsum = torch.cumsum(U_log,dim=0)#.to(device)
        # assert U_log_cumsum.shape == U.shape, "U_log_cumsum.shape != U.shape"

        # U_log_cumsum_extend.shape: [len(target_nonzero), LEN]

        # 减少显存的方法一：
        # 无非是求和from 1 to i in target_nonzero
        # U_sum_1 = get_sum_1_vmap(U_log_cumsum,target_nonzero)

        # 通过复制的方法二：
        U_log_cumsum_extend = torch.repeat_interleave(U_log_cumsum[None,:], repeats=len(target_nonzero),dim=0)#.to(device)
        del U_log_cumsum
        # print("U_log_cumsum_extend.shape shoule be: [len(target_nonzero),len(U)]")
        # U_log_cumsum.register_hook(save_grad('U_log_cumsum'))

        U_sum_1_idx = target_nonzero
        U_sum_2_idx = target_nonzero.squeeze()+1
        U_sum_1 = torch.gather(U_log_cumsum_extend,dim=1,index=U_sum_1_idx)#.to(device)
        del U_log_cumsum_extend
        del U_sum_1_idx
        # U_sum_2 = torch.log(torch.max(1-torch.gather(U,0,index=U_sum_2_idx),torch.tensor(eps,device=device)))#.to(device)
        # U_sum_2 = 1-torch.gather(U_log,0,index=U_sum_2_idx)#.to(device)
        # 或许可以：用log(U[t+1])表示log(1-U[t+1])
        # 潜在的问题：gather之后的值太小，因此exp之后非常接近1，用1-去之后是一个接近0的值，log=inf
        U_sum_2 = torch.log(torch.max(1-torch.exp(torch.gather(U_log,0,index=U_sum_2_idx)),torch.tensor(opt.MIN_LOSS)))#.to(device)
        # print(f"torch.gather(U_log,0,index=U_sum_2_idx)={torch.gather(U_log,0,index=U_sum_2_idx)}")
        del U_log
        del U_sum_2_idx
        loss_sum = U_sum_1.sum() + U_sum_2.sum() + loss_sum
        del U_sum_1
        # print(f"old method: U_sum_1.sum() ={U_sum_1.sum().detach().cpu()}, U_sum_2.sum()={U_sum_2.sum().detach().cpu()}")

        # assert not torch.isnan(U_sum_1.detach().sum()),f"U_sum_1={U_sum_1}"
        # assert not torch.isinf(U_sum_1.detach().sum()),f"U_sum_1={U_sum_1}"
        # assert not torch.isnan(U_sum_2.detach().sum()),f"U_sum_2={U_sum_2}"
        # assert not torch.isinf(U_sum_2.detach().sum()),f"U_sum_2={U_sum_2}"
        # loss_sum.register_hook(save_grad('loss_sum'))

        # torch.cuda.empty_cache()  # 会变慢不少
    # 除以batchsize
    # loss_sum.register_hook(save_grad('loss_sum'))
    # print("ALPHA:register_hook")
    # handler = Alpha.register_hook(save_grad('Alpha'))

    return -loss_sum/len(input_data)

def loss_fn_InferNet_GT3(input_data, Alpha, Target_data, eps, device):
    # 注意：P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)
    for i in range(len(input_data)):

        # Get target data
        target = Target_data[i, :].long()

        # Solve for U from Equi. condt.
        # Get settings
        setting = input_data[i, :, 0:3]
        d = setting[0, 0]
        b = setting[0, 1]
        v = setting[0, 2]

        # 1.首先计算LEN,T
        LEN,T = get_LEN_T_ts(v,b,d,max(target))
        # 2.然后筛选target data：大于T的没必要，算不了

        idx = torch.nonzero(target)
        target_nonzero = target[idx]  #.reshape(1,-1).squeeze().long()      # int

        # U_i最大必须可以访问到U_i[LEN+1],因为计算P[LEN]需要用到U[LEN+1]
        # 方法二：直接求U_log，并且使用vmap进行并行运算
        U_log = get_U_GT3_ts_log_vmap(LEN, T, v, d, b, Alpha[i,:],eps = eps, device=device)
        # assert not (torch.any(torch.isnan(U_log.detach()))), f"U_log has NaN and U_log = {U_log.detach()}"

        U_log_cumsum = torch.cumsum(U_log,dim=0)   #.to(device)
        # assert U_log_cumsum.shape == U.shape, "U_log_cumsum.shape != U.shape"

        # U_log_cumsum_extend.shape: [len(target_nonzero), LEN]

        # 减少显存的方法一：
        # 无非是求和from 1 to i in target_nonzero
        # U_sum_1 = get_sum_1_vmap(U_log_cumsum,target_nonzero)

        # 通过复制的方法二：
        U_log_cumsum_extend = torch.repeat_interleave(U_log_cumsum[None,:], repeats=len(target_nonzero),dim=0)#.to(device)
        del U_log_cumsum
        # print("U_log_cumsum_extend.shape shoule be: [len(target_nonzero),len(U)]")
        # U_log_cumsum.register_hook(save_grad('U_log_cumsum'))

        U_sum_1_idx = target_nonzero
        U_sum_2_idx = target_nonzero.squeeze()+1
        U_sum_1 = torch.gather(U_log_cumsum_extend,dim=1,index=U_sum_1_idx)#.to(device)
        del U_log_cumsum_extend
        del U_sum_1_idx
        # U_sum_2 = torch.log(torch.max(1-torch.gather(U,0,index=U_sum_2_idx),torch.tensor(eps,device=device)))#.to(device)
        # U_sum_2 = 1-torch.gather(U_log,0,index=U_sum_2_idx)#.to(device)
        # 或许可以：用log(U[t+1])表示log(1-U[t+1])
        # 潜在的问题：gather之后的值太小，因此exp之后非常接近1，用1-去之后是一个接近0的值，log=inf
        U_sum_2 = torch.log(torch.max(1-torch.exp(torch.gather(U_log,0,index=U_sum_2_idx)),torch.tensor(opt.MIN_LOSS)))#.to(device)
        # print(f"torch.gather(U_log,0,index=U_sum_2_idx)={torch.gather(U_log,0,index=U_sum_2_idx)}")
        del U_log
        del U_sum_2_idx
        loss_sum = U_sum_1.sum() + U_sum_2.sum() + loss_sum
        del U_sum_1
        # print(f"old method: U_sum_1.sum() ={U_sum_1.sum().detach().cpu()}, U_sum_2.sum()={U_sum_2.sum().detach().cpu()}")

        # assert not torch.isnan(U_sum_1.detach().sum()),f"U_sum_1={U_sum_1}"
        # assert not torch.isinf(U_sum_1.detach().sum()),f"U_sum_1={U_sum_1}"
        # assert not torch.isnan(U_sum_2.detach().sum()),f"U_sum_2={U_sum_2}"
        # assert not torch.isinf(U_sum_2.detach().sum()),f"U_sum_2={U_sum_2}"
        # loss_sum.register_hook(save_grad('loss_sum'))

        # torch.cuda.empty_cache()  # 会变慢不少
    # 除以batchsize
    # loss_sum.register_hook(save_grad('loss_sum'))
    # print("ALPHA:register_hook")
    # handler = Alpha.register_hook(save_grad('Alpha'))

    return -loss_sum/len(input_data)


def validate_params(mlp, data_loader, eps, device, MODEL_NAME):

    NLL_sum = torch.tensor(0., device=device, requires_grad=False)
    # KL_sum = torch.tensor(0., device=device, requires_grad=False)
    # GT_metric = torch.tensor([0.,0.,0.]).reshape(1,-1)

    cnt = 0
    for batch_id, data in enumerate(data_loader):

        input_data, target_metric,_, setting= data

        # print(f"---- {batch_id} batch----")
        # Do the inference
        input_data = input_data.to(device)
        target_data = target_metric.to(device)
        # target_params_data = target_params_data.to(device)      # params inferred by SA

        # Alpha, Labda = mlp(input_data)

        if MODEL_NAME == "InferNet_GT2":
            Alpha, Labda = mlp(input_data)
            Alpha = Alpha.detach().cpu().numpy()
            Labda = Labda.detach().cpu().numpy()
        elif MODEL_NAME == "InferNet_GT3":
            Alpha = mlp(input_data)
            Alpha = Alpha.detach().cpu().numpy()

        # GT_metric += torch.sum(metric_data,dim=0)
        cnt += len(input_data)  # 累加起来data_loader中所有的样本数

        for i in range(len(Alpha)):

            # Get target data
            target = target_data[i, :]
            idx = torch.nonzero(target)
            target_nonzero = target[idx].detach().cpu().squeeze().numpy()
            target_ls = [int(x) for x in target_nonzero]        # 统一一下，target在这里是int list

            # Solve for U from Equi. condt.
            # Get settings
            setting = input_data[i, :, 0:3]
            d = setting[0, 0].detach().cpu().numpy().item()
            b = setting[0, 1].detach().cpu().numpy().item()
            v = setting[0, 2].detach().cpu().numpy().item()

            #### metric计算参考自 GT_2_1
            # GT2和GT3对于这一部分的计算逻辑是一样的
            LEN, T = SA_for_PT_funcs_delta_eq1_GT2.get_LEN_T(v, b, d, max(target_ls))

            if MODEL_NAME == "InferNet_GT2":
                # Solve for U
                U = SA_for_PT_funcs_delta_eq1_GT2.get_U_GT2(LEN, v, d, b, Alpha[i].item(), Labda[i].item(), eps=0.)

            elif MODEL_NAME == "InferNet_GT3":
                # Solve for U
                U = SA_for_PT_funcs_delta_eq1_GT3.get_U_GT3(LEN, v, d, b, Alpha[i].item(), eps=0.)

            # 返回值是正值
            # GT2和GT3对于这一部分的计算逻辑是一样的
            nll_metric = SA_for_PT_funcs_delta_eq1_GT2.get_nll_meric(target_ls, U, LEN,TARGET = 1)
            # kl_metric = SA_for_PT_funcs_delta_eq1_GT2.get_KL_meric(target, target_p, U, LEN, TARGET = 1)

            NLL_sum = nll_metric + NLL_sum
            # KL_sum = kl_metric + KL_sum

    return NLL_sum.detach().cpu().item() / cnt

def trainer(MODEL_NAME, train_loader, val_loader, test_loader, mlp, opt, device):
    """
    Main body of a training process. Called by objective function
    Args:
        MODEL_NAME:
        train_loader:
        val_loader:
        mlp:
        opt:    params to be held
        device:

    Returns

    """
    if MODEL_NAME == "InferNet_GT2":
        params = get_params_GT2(mlp,opt)
    elif MODEL_NAME == "InferNet_GT3":
        params = get_params_GT3(mlp,opt)

    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.Exp_gamma, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.StepLR_step_size, gamma=opt.StepLR_gamma,last_epoch=-1)

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

            # Cal the MLE loss and draw the distrb.
            if MODEL_NAME == "InferNet_GT2":
                alpha, labda = mlp(input_data)
                loss = loss_fn_InferNet_GT2(input_data, alpha, labda, target_loss, opt.MIN_LOSS, device)
            elif MODEL_NAME == "InferNet_GT3":
                alpha = mlp(input_data)
                loss = loss_fn_InferNet_GT3(input_data, alpha, target_loss, opt.MIN_LOSS, device)

            epoch_train_loss += loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # draw_loss(viz,total_train_step, loss.detach().cpu(), plot.win_train_loss_str)

            total_train_step += 1

            # torch.cuda.empty_cache()  # 会变慢不少

        scheduler.step()

        # print(f"========== IN EPOCH {epoch} the total loss is {epoch_train_loss} ==========")

        # Vali
        mlp.eval()
        with torch.no_grad():
            total_vali_metric, GT_metric = validate_params(mlp, val_loader, opt.MIN_LOSS, device, MODEL_NAME)
            # draw_metric_wo_GT(viz, epoch, total_vali_metric)
            # print(f"========== IN EPOCH {epoch} the vali metric is {total_vali_metric}, ==========")

            # 记录最小的total_vali_metric，并且保存此时的模型
            if total_vali_metric < min_loss:
                min_loss = total_vali_metric
                model_params_MLP = get_InferNet_save_path(opt.ARTIFICIAL, seed, opt.net_root_path, opt.noise_pct , MODEL_NAME)
                hyperparameters = {
                    'model_name': MODEL_NAME,
                    'learning_rate': opt.learning_rate,
                    'lr_for_alpha': opt.lr_for_alpha,
                    'lr_for_labda': opt.lr_for_labda,
                    'batch_size': opt.batch_size,
                    'epoch_when_saved': epoch
                }
                save_checkpoint(mlp,hyperparameters,model_params_MLP)

        # draw_loss(viz,epoch, epoch_train_loss, plot.win_train_epoch_loss_str)

    ########### Load .pth and do testing
    mlp.eval()
    with torch.no_grad():
        # Read-in
        model_path = get_InferNet_save_path(opt.ARTIFICIAL, seed, opt.net_root_path, opt.noise_pct , MODEL_NAME)
        mlp,_ = load_checkpoint(model_path, mlp)
        total_test_metric, GT_metric = validate_params(mlp,test_loader, opt.MIN_LOSS, device, MODEL_NAME)

    mlp.train()

    return total_test_metric,GT_metric


if __name__ == '__main__':

    setup_seed(seed)
    print(f"========== MODEL_NAME = {MODEL_NAME} ==========")

    print(f"========== seed = {seed} ==========")
    print(f"========== seed = {seed} ==========")
    print(f"========== seed = {seed} ==========")
    # 画图设置
    # timestamp = int(time.time())
    # time_str = str("_") + time.strftime('%y%m%d%H%M%S', time.localtime(timestamp))
    # print(f"time_str = {time_str}")
    # env_str = MODEL_NAME + "_seed=" + str(seed) + time_str
    # viz = Visdom(env=env_str)
    #
    # viz.line(X=[0.], Y=[0.], env=env_str, win=plot.win_train_loss_str, opts=dict(title=plot.win_train_loss_str))
    # viz.line(X=[0.], Y=[0.], env=env_str, win=plot.win_train_epoch_loss_str, opts=dict(title=plot.win_train_epoch_loss_str))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 数据设置
    dataset = myDataset(opt.train_path, opt.target_path_metric, opt.target_path_loss, opt.data_key_path)
    shuffled_indices = save_data_idx(dataset, opt.arr_flag)
    train_idx, val_idx, test_idx = get_data_idx(shuffled_indices,opt.train_pct, opt.vali_pct)
    train_loader,val_loader,test_loader = get_data_loader(dataset, opt.batch_size, train_idx, val_idx, test_idx, my_collate_fn_GT2)

    model = get_model_opt(MODEL_NAME)['model'].to(device)

    performance = trainer(MODEL_NAME,train_loader, val_loader, test_loader, model, opt, device)

    print(f"========== seed = {seed} ==========")
    print(f"========== MODEL_NAME = {MODEL_NAME} ==========")
    print(f"{performance}")
