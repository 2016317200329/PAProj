

# Not Sequential
# 573, 588, 1581
class MLP_1_1(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians,kernel_size, stride) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=0,bias=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=0,bias=True)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=0,bias=True)

        self.BN_aff1 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)
        self.BN_aff2 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)
        self.BN_aff3 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)

        self.ac_func = nn.Softplus()

        self.z_pi = nn.Sequential(
            nn.Linear(self.ln_in, n_gaussians),
            nn.Softmax(dim=1)
        )

        self.z_mu = nn.Linear(self.ln_in, n_gaussians)
        self.z_sigma = nn.Linear(self.ln_in, n_gaussians)

    def forward(self, x):

        x = self.BN1(x)
        x = torch.unsqueeze(x,dim=1)                     # torch.Size([B, 1, 3, 300])

        x1 = F.softplus(self.conv1(x))                   # [B, 1, 1, ln_in]
        x2 = F.softplus(self.conv2(x))                   # [B, 1, 1, ln_in]
        x3 = F.softplus(self.conv3(x))                   # [B, 1, 1, ln_in]

        x1_1 = torch.squeeze(x1)
        x2_1 = torch.squeeze(x2)
        x3_1 = torch.squeeze(x3)

        x1_2 = self.BN_aff1(x1_1)
        x2_2 = self.BN_aff2(x2_1)
        x3_2 = self.BN_aff3(x3_1)

        x1_3 = F.softplus(x1_2)
        x2_3 = F.softplus(x2_2)
        x3_3 = F.softplus(x3_2)

        pi = self.z_pi(x1_3)
        mu = self.z_mu(x2_3)

        sigma = torch.exp(self.z_sigma(x3_3))
        sigma = torch.clamp(sigma,1e-4)

        # return x1_3
        return pi, mu, sigma


# 1005
class Conv_block_1(nn.Module):
    def __init__(self, ch_out,kernel_size=12, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.ac_func = nn.Softplus()

        self.conv = nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=self.kernel_size, stride=self.stride, padding=0,bias=True)
        self.BN_aff2 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)
        self.BN_aff1 = nn.BatchNorm1d(num_features=1,affine=True)

    def forward(self, x):
        # Conv=>BN=>AC
        x = self.conv(x)
        x = torch.squeeze(x,dim=1)
        x = self.ac_func(self.BN_aff1(x))

        return x

class Conv_1_1(nn.Module):
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=12, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)

        self.layer_pi = Conv_block_1(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_mu = Conv_block_1(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_sigma = Conv_block_1(ch_out=1,kernel_size=kernel_size,stride=stride)

        self.ac_func = nn.Softplus()

        self.z_pi = nn.Sequential(
            nn.Linear(self.ln_in, n_gaussians),
            nn.Softmax(dim=1)
        )

        self.z_mu = nn.Linear(self.ln_in, n_gaussians)
        self.z_sigma = nn.Linear(self.ln_in, n_gaussians)

    def forward(self, x):

        x = self.BN1(x)
        x = torch.unsqueeze(x,dim=1)                     # torch.Size([B, 1, 3, 300])

        x_pi = torch.squeeze(self.layer_pi(x))          # 不加squeeze不行
        x_mu = torch.squeeze(self.layer_pi(x))
        x_sigma = torch.squeeze(self.layer_pi(x))

        # 有必要吗ac：Definitely
        # 哪一层起到了作用？mu
        # pi = self.ac_func(self.z_pi(x_pi))
        pi = self.z_pi(x_pi)

        # mu = self.z_mu(x_mu)
        mu = self.ac_func(self.z_mu(x_mu))
        # 不要给sigma加ac
        # sigma = self.ac_func(self.z_sigma(x_sigma))
        sigma = self.z_sigma(x_sigma)
        # print("sigma shape:",sigma.shape)
        # print("sigma :",sigma)

        sigma = torch.exp(sigma)
        sigma = torch.clamp(sigma,1e-4)

        # return x1_3
        return pi, mu, sigma