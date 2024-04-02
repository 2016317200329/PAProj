def plot_net(writer,model,input):
    """
    在tensorboard中可视化model structure
    Args:
        writer:
        model:
        input:
    """
    writer.add_graph(model, input_to_model=input, verbose=False)

def plot_conv(writer,model,epoch):
    """
    对conv层的weight进行可视化（display as a pic）
    Args:
        writer:
        model:
        epoch:
    """
    for name,param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            in_channels = param.size()[1]	# 输入通道
            out_channels = param.size()[0]   # 输出通道
            k_w, k_h = param.size()[3], param.size()[2]   # 卷积核的尺寸
            kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
            kernel_grid = torchvision.utils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
            # print(kernel_grid)
            win_str = f'{name}_all'+str(epoch)
            # viz.images(kernel_all, win=win_str,env="001",opts=dict(title = win_str))
            # writer.add_image(f'{name}_all', kernel_grid, global_step=epoch)


def plot_conv_weight(writer,mlp,epoch,tag_str):
    """
    对conv层的weight在tensorbord中输出（display in hist）
    Args:
        mlp:
        epoch:
    """
    for name,param in mlp.named_parameters():
        if 'conv' in name and 'weight' in name:
            writer.add_histogram(tag = name+tag_str, values=param.data.clone().cpu().numpy(),global_step=epoch)


def plot_mu_weight(writer,mlp,epoch,tag_str):
    """
    对z_mu层的weight在tensorbord中输出（display in hist）
    Args:
        mlp:
        epoch:
    """
    for name, param in mlp.named_parameters():
        if 'z_mu' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


def plot_pi_weight(writer,mlp,epoch,tag_str):
    """
    对z_pi层的weight在tensorbord中输出（display in hist）
    Args:
        mlp:
        epoch:
    """
    for name, param in mlp.named_parameters():
        if 'z_pi' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


def plot_sigma_weight(writer,mlp,epoch,tag_str):
    """
    对z_sigma层的weight在tensorbord中输出（display in hist）
    Args:
        mlp:
        epoch:
    """
    for name, param in mlp.named_parameters():
        if 'z_sigma' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


def plot_alpha(writer,mlp,epoch,tag_str):
    """
    plot weight of 'alpha' in tensorboard
    Args:
        writer:  tensorboard writer
        mlp:     the NN model
        epoch:   epoch
        tag_str: the name of the plot window
    """
    for name, param in mlp.named_parameters():
        # 简单写法：想画什么就大概记录下这一层的名字‘block_alpha’，想画权重就是‘weight’，偏移就是'bias'
        if 'block_alpha' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)

def plot_labda(writer,mlp,epoch,tag_str):
    for name, param in mlp.named_parameters():
        if 'block_labda' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


win_train_loss_str = "The Loss of BATCH in the Training Data"
win_vali_loss_str = "The Loss in the Vali Data"
win_train_epoch_loss_str = "The Loss of EPOCH in the Training Data"
win_vali_metric_str = "The NLL of ALL Vali Data"

def draw_loss(viz, X_step, loss, win_str):
    viz.line(X = [X_step], Y = [loss],win=win_str, update="append",
        opts= dict(title=win_str))

def draw_metric_wo_GT(viz, X_step, total_vali_metric):
    '''
    Only plot metric, ignore the metric of the GTs
    Args:
        viz:
        X_step:
        total_vali_metric:

    Returns:

    '''
    viz.line(X = [X_step], Y = [total_vali_metric],win=win_vali_metric_str, update="append", opts= dict(title=win_vali_metric_str, legend=['pred','GT-1','GT-2'], showlegend=True,xlabel="epoch", ylabel="NLL"))


def draw_metric_w_GT(viz, X_step, total_vali_metric, GT_metric, win_str = win_vali_metric_str, MODEL_NAME='pred'):
    '''
    Plot metric, including GTs'.
    Args:
        viz:
        X_step:
        total_vali_metric:
        GT_metric:
        win_str:
        MODEL_NAME:

    Returns:

    '''
    legend_str = [MODEL_NAME] + ['GT1', 'GT2(InferNet)']
    viz.line(X=[X_step], Y=[[total_vali_metric, GT_metric[0, 0], GT_metric[0, -1]]], win=win_str, update="append",
            opts=dict(title=win_vali_metric_str, legend=legend_str, showlegend=True, xlabel="epoch", ylabel="NLL"))
