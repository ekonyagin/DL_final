import torch.nn as nn
import torch.functional as F


class ResBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(ResBlock2d, self).__init__()
        ks = kernel_size
        d = dilation
        stride = 1
        pad_size = int((ks + (ks - 1) * (d - 1) / stride - 1) / 2)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1,
                               padding=pad_size, dilation=dilation)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride=1,
                               padding=pad_size, dilation=dilation)
        self.bn = nn.InstanceNorm2d(in_ch)
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm2d(out_ch)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        bypass = []
        if in_ch != out_ch:
            bypass.append(nn.Conv2d(in_ch, out_ch, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)


class ResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock3d, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, padding=1)
        self.bn = nn.InstanceNorm3d(in_ch)
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm3d(out_ch)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        bypass = []
        if in_ch != out_ch:
            bypass.append(nn.Conv3d(in_ch, out_ch, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)


class HoloEncoder(nn.Module):
    def __init__(self, nf=32):
        super().__init__()
        self.nf = nf

        self.res_conv1 = ResBlock2d(3, 128, kernel_size=3, dilation=2)
        self.res_conv2 = ResBlock2d(128, self.nf * 128, kernel_size=3,
                                    dilation=2)

        self.postproc = nn.Sequential(
            nn.Conv3d(nf, nf // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(nf // 2, affine=True),
            nn.ReLU(),
            nn.Conv3d(nf // 2, nf // 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(nf // 4, affine=True),
            nn.ReLU()
        )

        pnf = (nf // 4) * 128  # 1024  #3d_channels * depth

        # TODO: should be 1x1
        self.proj = nn.Sequential(
            nn.Conv2d(pnf, pnf // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(pnf // 2, affine=True),
            nn.ReLU()
        )

        self.res_conv3 = ResBlock2d(pnf // 2, 512)
        self.res_conv4 = ResBlock2d(512, 512)
        self.res_conv5 = ResBlock2d(512, 512)

        self.fc = nn.Sequential(
            nn.Linear(512 * 16 * 16, 7 * 512)
        )

    def stn(self, x, theta):
        # theta must be (Bs, 3, 4) = [R|t]
        # theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        out = F.grid_sample(x, grid, padding_mode='zeros')
        return out

    def forward(self, x, thetas):
        '''
            x: batch of images of shape (Bs, c, h, w)
            thetas : Transform matricies, must be (Bs, 3, 4) = [R|t]
        '''
        bs = x.size(0)
        x = self.res_conv1(x)  # -> (bs, 128, 128, 128)
        x = self.res_conv2(x)  # -> (bs, 128*nf, 128, 128)

        # reshape to (bs, 3d_channels, d, h, w)
        x = x.reshape(bs, self.nf, x.size(1) // self.nf, x.size(2), x.size(3))

        # Perform rotation
        x = self.stn(x, thetas)

        # Postprocess before projection
        x = self.postproc(x)  # -> (bs, nf//4, 128, 128, 128)

        # Projection unit. Concat depth and channels
        x = x.reshape(bs, x.size(1) * x.size(2), x.size(3), x.size(4))
        x = self.proj(x)  # -> (bs, 1024, 128, 128)

        # Downsaple to (bs, 7 * 512)
        x = self.res_conv3(x)  # -> (bs, 512, 128, 128) NO DILATIONS, KS=3
        x = F.avg_pool2d(x, 2)  # -> (bs, 512, 64, 64)

        x = self.res_conv4(x)  # -> (bs, 512, 64, 64) NO DILATIONS, KS=3
        x = F.avg_pool2d(x, 2)  # -> (bs, 512, 32, 32)

        x = self.res_conv5(x)  # -> (bs, 512, 32, 32) NO DILATIONS, KS=3
        x = F.avg_pool2d(x, 2)  # -> (bs, 512, 16, 16)
        x = x.reshape(bs, -1)

        x = self.fc(x)  # -> (bs, 7 * 512)
        latent_code = x.reshape(bs, 7, 512)

        return latent_code
