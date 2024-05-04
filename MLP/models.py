# 记录了所有model

import torch.nn as nn
import torch
import torch.nn.functional as F
from thop import profile

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bound_alpha = torch.tensor([-0.3,0.3],device=device)
bound_labda = torch.tensor([0.01,18],device=device)


# Basic MDN model
class Conv_block_4(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, ch_in = 3, len_in = 300, ch_out=1,kernel_size=9, stride=3, init_weight=True) -> None:
        super().__init__()

        self.kernel_size = (ch_in,kernel_size)
        self.stride = (1,stride)
        self.ln_in = int((len_in-self.kernel_size[1])/self.stride[1]+1)
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
    '''
        According to `ch_in` to decide the channel num.
    '''
    def __init__(self, n_gaussians, ch_in = 3, len_in = 300, ch_out=1, kernel_size=9, stride=3) -> None:
        super().__init__()

        self.kernel_size = (ch_in,kernel_size)
        self.stride = (1,stride)
        self.ln_in = int((len_in-self.kernel_size[1])/self.stride[1]+1)

        self.BN1 = nn.BatchNorm1d(num_features=ch_in,affine=True)
        self.layer_pi = Conv_block_4(ch_in=ch_in,len_in=len_in,ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_scale = Conv_block_4(ch_in=ch_in,len_in=len_in,ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_shape = Conv_block_4(ch_in=ch_in,len_in=len_in,ch_out=1,kernel_size=kernel_size,stride=stride)

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


# Basic MDN model + dilated conv
class Conv_block_4_dilated(nn.Module):

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, ch_out=1,kernel_size=9, stride=3, dilation = 1, init_weight=False) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.dilation = (1,dilation)
        self.ln_in = int((300-kernel_size - (kernel_size-1)*(dilation-1))/stride+1)

        self.ac_func = nn.Softplus()

        self.conv = nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=self.kernel_size, stride=self.stride, padding=0, dilation=self.dilation)
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
        return x

class Conv_1_4_dilated(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=[9,9,12],stride=[3,3,3], dilation = [1,1,2]) -> None:
        super().__init__()

        self.ln_in_1 = int((300-kernel_size[0] - (kernel_size[0]-1)*(dilation[0]-1))/stride[0]+1)
        self.ln_in_2 = int((300-kernel_size[1] - (kernel_size[1]-1)*(dilation[1]-1))/stride[1]+1)
        self.ln_in_3 = int((300-kernel_size[2] - (kernel_size[2]-1)*(dilation[2]-1))/stride[2]+1)

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)

        self.layer_pi = Conv_block_4_dilated(ch_out=1,kernel_size=kernel_size[0],dilation = dilation[0],stride=stride[0])
        self.layer_scale = Conv_block_4_dilated(ch_out=1,kernel_size=kernel_size[1],dilation = dilation[1],stride=stride[1])
        self.layer_shape = Conv_block_4_dilated(ch_out=1,kernel_size=kernel_size[2],dilation = dilation[2],stride=stride[2])

        self.ac_func = nn.Softplus()

        self.z_pi = nn.Sequential(
            nn.Linear(self.ln_in_1, n_gaussians),
            nn.Softmax(dim=1)           # dim=0是B, dim=1才是feature
        )

        self.z_scale = nn.Linear(self.ln_in_2, n_gaussians)
        self.z_shape = nn.Linear(self.ln_in_3, n_gaussians)

    def forward(self, x):

        x = self.BN1(x)
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



# class MLP_1_1(nn.Module):
#     '''
#     This is a 2-layer net for GT2 to infer parameters.
#
#     '''
#     # code->generate->override methods
#     def __init__(self) -> None:
#         super().__init__()
#         self.BN1 = nn.BatchNorm1d(num_features=8, affine=True)
#
#         self.block_alpha = nn.Sequential(
#             nn.Linear(8, 1),
#             nn.Tanh()
#         )
#         self.block_labda = nn.Sequential(
#             nn.Linear(8, 1),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = torch.squeeze(x, dim=1)  # [B,C,N]->[B,N]，因为C=1
#         x = self.BN1(x)
#
#         alpha = self.block_alpha(x)
#         labda = self.block_labda(x)
#
#         # Clamp
#         alpha = torch.clamp(alpha, min=bound_alpha[0], max=bound_alpha[1])
#         labda = torch.clamp(labda, min=bound_labda[0], max=bound_labda[1])
#
#         return alpha, labda


class MLP_2_1(nn.Module):
    '''
    This is a 3-layer net for GT2 to infer parameters.
    '''
    # code->generate->override methods
    def __init__(self) -> None:
        super().__init__()
        self.BN1 = nn.BatchNorm1d(num_features=8,affine=True)
        self.LN1 = nn.Linear(8,4)

        self.block_alpha = nn.Sequential(
            nn.Linear(4, 1),
            nn.Tanh()
        )
        self.block_labda = nn.Sequential(
            nn.Linear(4, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.squeeze(x,dim=1)
        x = self.BN1(x)

        x = self.LN1(x) # a common layer
        x = F.relu(x)

        alpha = self.block_alpha(x)
        labda = self.block_labda(x)

        # Clamp
        alpha = torch.clamp(alpha,min=bound_alpha[0],max=bound_alpha[1])
        labda = torch.clamp(labda,min=bound_labda[0],max=bound_labda[1])

        return alpha,labda

# class MLP_GT3_1(nn.Module):
#     '''
#     This is a 2-layer net for GT3 to infer parameters.
#     '''
#     # code->generate->override methods
#     def __init__(self) -> None:
#         super().__init__()
#         self.BN1 = nn.BatchNorm1d(num_features=8, affine=True)
#
#         self.block_alpha = nn.Sequential(
#             nn.Linear(8, 1),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = torch.squeeze(x, dim=1)
#         x = self.BN1(x)
#
#         alpha = self.block_alpha(x)
#
#         # Clamp
#         alpha = torch.clamp(alpha, min=bound_alpha[0], max=bound_alpha[1])
#
#         return alpha

class MLP_GT3_2(nn.Module):
    '''
    This is a 3-layer net for GT3 to infer parameters.
    '''
    # code->generate->override methods
    def __init__(self) -> None:
        super().__init__()
        self.BN1 = nn.BatchNorm1d(num_features=8, affine=True)
        self.LN1 = nn.Linear(8, 4)

        self.block_alpha = nn.Sequential(
            nn.Linear(4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.BN1(x)

        x = self.LN1(x)
        x = F.relu(x)

        alpha = self.block_alpha(x)

        # Clamp
        alpha = torch.clamp(alpha, min=bound_alpha[0], max=bound_alpha[1])

        return alpha


# 参数量91,000+
# 2层MLP+MDN+weibull
class MDN_MLP_2_Wei(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians) -> None:
        super().__init__()
        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)
        self.BN2 = nn.BatchNorm1d(num_features=100,affine=True)
        self.BN3 = nn.BatchNorm1d(num_features=12,affine=True)
        self.drop = nn.Dropout(0.3)

        self.linear1 = nn.Linear(900, 100)
        self.linear2 = nn.Linear(100, 12)

        self.ac_func = nn.Softplus()
        self.ac_func2 = nn.SELU()
        self.ac_func3 = nn.LeakyReLU()

        self.flatten = nn.Flatten()

        self.z_pi = nn.Sequential(
            nn.Linear(12, n_gaussians),  # 30个params要learn
            nn.Softmax(dim=1)
        )

        self.z_scale = nn.Linear(12, n_gaussians)
        self.z_shape = nn.Linear(12, n_gaussians)


    def forward(self, x):
        # print("x.shape: ",x.shape)  # torch.Size([40, 3, 300])
        x = self.BN1(x)
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.BN2(x)
        x = self.ac_func3(self.drop(x))

        x = self.linear2(x)
        x = self.BN3(x)
        x = self.ac_func3(self.drop(x))

        pi = self.z_pi(x)

        scale = torch.exp(self.z_scale(x))
        scale = torch.clamp(scale,1e-4)
        shape = torch.exp(self.z_shape(x))
        shape = torch.clamp(shape,1e-4)

        return pi, scale, shape


# 1569
class Conv_block_9(nn.Module):

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, ch_out=1,kernel_size=9, stride=3, dilation=1,init_weight=False) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-kernel_size-(kernel_size-1)*(dilation-1))/stride+1)
        # self.ln_in = 84

        self.ac_func = nn.Softplus()

        self.conv = nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=self.kernel_size, stride=self.stride, padding=0, dilation=(1,dilation))
        self.BN_aff1 = nn.BatchNorm1d(num_features=1,affine=True)
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
        return x

class Conv_1_9(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=9, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)
        self.ln_scale = int((300-self.kernel_size[1])/self.stride[1]+1)
        self.ln_shape = 36

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)

        self.layer_pi = Conv_block_9(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_scale = Conv_block_9(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_shape = Conv_block_9(ch_out=1,kernel_size=30,stride=6,dilation=3)

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

        x_pi = self.layer_pi(x)
        x_scale = self.layer_scale(x)
        x_shape = self.layer_shape(x)

        pi = self.z_pi(x_pi)
        scale = torch.exp(self.z_scale(x_scale))
        scale = torch.clamp(scale,1e-4)
        shape = torch.exp(self.z_shape(x_shape))
        shape = torch.clamp(shape,1e-4)

        return pi,scale,shape


# Basic MDN model
# Extend the elements number in the linear layer
class Conv_block_5(nn.Module):

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, ch_out=1,kernel_size=9, stride=3, init_weight=True) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
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

class Conv_1_5(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=6, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)
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


# if __name__ == '__main__':
#
#     model_conv = Conv_1_4(2)
#     model_mlp = MDN_MLP_2_Wei(2)
#     input = torch.randn(1, 3,300)
#     macs_conv, params_conv = profile(model_conv, inputs=(input,))
#     macs_mlp, params_mlp = profile(model_mlp, inputs=(input,))
#     print("conv:",macs_conv,params_conv)
#     print("mlp:",macs_mlp,params_mlp)
#
#     # 当模型的FLOPS（每秒浮点运算次数）较低时，通常意味着模型在计算方面的需求较小或计算效率较高。以下是一些可能的含义
#     # conv: 13307.0 1272.0
#     # mlp: 95325.0 91620.0