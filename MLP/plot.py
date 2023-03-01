

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