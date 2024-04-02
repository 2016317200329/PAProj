from common import *
from utils import *
from mydataset import *

import my_collate_fn
from my_collate_fn import my_collate_fn_3

reload(loss)    # 必须reload！！
reload(plot)
reload(my_collate_fn)
# reload(models)

from Config.config_MDN import DefaultConfig
opt = DefaultConfig()

######*********CHECK CAREFULLY***********########

# MODEL_NAME = "GT1_GT2_EMD"  # 不能用+
MODEL_NAME = "GT1_GT2_GT3_EMD"  # 不能用+
if MODEL_NAME == "GT1_GT2_EMD":
    INPUT_LIST=[1,2,4]  # 注意和MODEL_NAME要对应起来！
elif MODEL_NAME == "GT1_GT2_GT3_EMD":
    INPUT_LIST=[1,2,3,4]  # 注意和MODEL_NAME要对应起来！


lr_for_pi = opt.lr_for_pi # opt.lr_for_pi
lr_for_shape = opt.lr_for_shape
lr_for_scale = opt.lr_for_scale
learning_rate = opt.learning_rate

MB_MDN_Flag = opt.Conv_1_4

batch_size = opt.batch_size
EPOCH_NUM = opt.EPOCH_NUM
q=opt.q

# 创建一个解析器对象
parser = argparse.ArgumentParser(description="Run the script with a specific seed.")
# 添加seed参数
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility.")
# 解析命令行参数
args = parser.parse_args()
seed = args.seed
# seed = opt.seed

######*********CHECK CAREFULLY***********########

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

def trainer(train_loader, val_loader, test_loader, mlp, opt, viz, device):
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
            input_data, _, target_loss, setting, _ = data
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

            draw_loss(viz,total_train_step, loss.detach().cpu(), plot.win_train_loss_str)

            total_train_step += 1
        scheduler.step()
        # print(f"========== IN EPOCH {epoch} the total loss is {epoch_train_loss}, ==========")

        draw_loss(viz,epoch, epoch_train_loss, plot.win_train_epoch_loss_str)

        ########### Vali
        mlp.eval()
        with torch.no_grad():
            total_vali_metric, GT_metric, odds = validate(mlp, val_loader, opt.N_gaussians, opt.MIN_LOSS, device, detail=True)
            if opt.DRAW_VAL:

                draw_metric_w_GT(viz, epoch+1, total_vali_metric, GT_metric, plot.win_vali_metric_str, MODEL_NAME)
                # writer.add_scalars(opt.vali_main_tag_str, {MODEL_NAME: total_vali_metric,
                #                                         "GT-1": GT_metric[0, 0],
                #                                         "GT-2": GT_metric[0, -1]}, epoch+1)
            if opt.PRINT_VAL:
                print(f"========== IN EPOCH {epoch} the vali metric is {total_vali_metric} | min_loss = {min_loss} ==========")

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
            total_test_metric, GT_metric, odds = validate(mlp, test_loader, opt.N_gaussians, opt.MIN_LOSS, device,detail=True)
        else:
            pass
            # total_test_metric, GT_metric, odds = validate_q(mlp, test_loader, opt.N_gaussians, opt.MIN_LOSS, device,detail=True)

        GT_metric_1 = GT_metric[0, 0]
        # if opt.REVENUE:
        #     # The expected revenue
        #     rmse_er = rmse_revenue(mlp, test_loader, test_loader_er, opt.N_gaussians, device)

    mlp.train()

    # 和GT-1的差值：越大越好
    metric_diff = GT_metric_1 - total_test_metric
    print(f"GT_metric = {GT_metric}, NN prediction = {total_test_metric}")
    print(f"the GTs: GT1,GT2(common),GT2(SA), GT2(NN)")

    print(f"The odds:  {odds}")
    # if opt.REVENUE:
    #     print(f"RMSE of Exp Revenue: ", rmse_er)

    # 保存模型
    mlp_all.append(mlp.state_dict())

    return metric_diff,total_test_metric,GT_metric

ans = []
pred_metric = []    # 记录了所有metric
mlp_all = []        # 记录了所有MDN model

if __name__ == '__main__':

    running_times=3
    setup_seed(seed)
    print(f"========== MODEL_NAME = {MODEL_NAME} | INPUT_LIST = {INPUT_LIST} ==========")
    print(f"========== seed = {seed} ==========")
    print(f"========== seed = {seed} ==========")
    print(f"========== seed = {seed} ==========")

    # 生成当前时间的时间戳, 作为画图的区分
    timestamp = int(time.time())
    time_str = str("_")+time.strftime('%y%m%d%H%M%S', time.localtime(timestamp))
    print(f"time_str = {time_str}")
    env_str = MODEL_NAME + "_seed=" + str(seed) + time_str

    viz = visdom.Visdom(env=env_str)
    # writer = SummaryWriter(log_dir="logs-MLP/" + opt.logs_str, flush_secs=60)   # opt.logs_str是log的文件夹名字

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = myDataset(opt.train_path, opt.target_path_metric, opt.target_path_loss, opt.data_key_path, opt.NLL_metric_path)
    # dataset_revenue = myDataset_revenue(opt.target_path_metric, opt.target_revenue_path)

    shuffled_indices = save_data_idx(dataset, opt.arr_flag)
    train_idx, val_idx, test_idx = get_data_idx(shuffled_indices,opt.train_pct, opt.vali_pct,opt.SET_VAL)
    my_collate_fn = functools.partial(my_collate_fn_3, INPUT_LIST=INPUT_LIST)
    train_loader,val_loader,test_loader = get_data_loader(dataset, opt.batch_size, train_idx, val_idx, test_idx, my_collate_fn)

    for i in range(running_times):

        if MB_MDN_Flag:
            model = Conv_1_4(opt.N_gaussians,ch_in=len(INPUT_LIST)).to(device)  # put your model and data on the same computation device.

        if not MB_MDN_Flag:
            model = MDN_MLP_2_Wei(opt.N_gaussians).to(device)

        if opt.DRAW_VAL:
            model.eval()
            with torch.no_grad():
                total_vali_metric, GT_metric, odds = validate(model, val_loader, opt.N_gaussians, opt.MIN_LOSS, device, detail=True)
                draw_metric_w_GT(viz, 0, total_vali_metric, GT_metric, plot.win_vali_metric_str, MODEL_NAME)

            # writer.add_scalars(opt.vali_main_tag_str, {MODEL_NAME: total_vali_metric.detach().cpu(),
            #                                                 "GT-1": GT_metric[0, 0],
            #                                                 "GT-2": GT_metric[0, -1]}, 0)
            model.train()

        # For training , cal the exp revenue
        performance = trainer(train_loader, val_loader, test_loader, model, opt, viz, device)

        ans.append(performance[0])
        pred_metric.append(performance[1])

        print(f"running time = {i}, performance = {performance[0]}")
        # if opt.REVENUE:
        #     print(f"running time = {i}, Expected Revenue = {performance[-1]}")

    print(f"========== seed = {seed} ==========")
    print(f"========== MODEL_NAME = {MODEL_NAME} ==========")

    print(f"========== mean metric: {np.mean(pred_metric)} ==========")
    print(f"========== all metric: {pred_metric} ==========")
    print(f"========== best performance: {min(pred_metric)} ==========")

