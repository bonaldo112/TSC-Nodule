'''
Description:
Date: 2023-07-21 14:36:27
LastEditTime: 2023-07-27 18:41:47
FilePath: /chengdongzhou/ScConv.py
'''
import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupBatchnorm3d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm3d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W, D = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W, D)
        return x * self.weight + self.bias


class SRU_3D(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm3d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        # print("x:", x.shape)
        gn_x = self.gn(x)
        # print("gn_x:", gn_x.shape)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        # print("w_gamma:", w_gamma.shape)
        w_gamma = w_gamma.view(1, -1, 1, 1, 1)
        # print("w_gamma:", w_gamma.shape)
        reweigts = self.sigomid(gn_x * w_gamma)
        # print("reweights:", reweigts.shape)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        # print("w1:", w1.shape)
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        # print("w2:", w2.shape)
        x_1 = w1 * x
        # print("x_1:", x_1.shape)
        x_2 = w2 * x
        # print("x_2:", x_2.shape)
        y = self.reconstruct(x_1, x_2)
        # print("y:", y.shape)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        # print("x_11:", x_11.shape)
        # print("x_12:", x_12.shape)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        # print("x_21:", x_21.shape)
        # print("x_22:", x_22.shape)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU_3D(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv3d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv3d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv3d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv3d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv3d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # print("x:", x.shape)
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        # print("up:", up.shape)
        # print("low:", low.shape)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # print("up:", up.shape)
        # print("low:", low.shape)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # print("Y1:", Y1.shape)
        # print("Y2:", Y2.shape)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        # print("out:", out.shape)
        out = F.softmax(self.advavg(out), dim=1) * out
        # print("out:", out.shape)
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        # print("out1:", out1.shape)
        # print("out2:", out2.shape)
        return out1 + out2


class ScConv_3D(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU_3D(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU_3D(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16, 16)
    model = ScConv_3D(32)
    # model = SRU_3D(32)
    # model = CRU_3D(32)
    print(model(x).shape)
