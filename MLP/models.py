

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
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # nn.init.xavier_uniform_(m.weight,gain=1)
        #         nn.init.uniform_(m.weight,a=-0.5,b=0.5)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.uniform_(m.weight,a=-1,b=1)
        #         # m.weight.data.normal_(0, 0.02)
        #         # nn.init.xavier_normal_(m.weight,gain=1)
        #         # nn.init.xavier_uniform_(m.weight,gain=1)
        #         # nn.init.orthogonal_(m.weight)
        #         # nn.init.zeros_(m.bias)

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