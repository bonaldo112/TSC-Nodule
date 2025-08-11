import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedBlockConv3d(nn.Module):
    '''Tied Block Conv2d'''
    def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=True, \
                B=1, args=None, dropout_tbc=0.0, groups=1):
        super(TiedBlockConv3d, self).__init__()
        assert planes % B == 0
        assert in_planes % B == 0
        self.B = B
        self.stride = stride
        self.padding = padding
        self.out_planes = planes
        self.kernel_size = kernel_size
        self.dropout_tbc = dropout_tbc

        self.conv = nn.Conv3d(in_planes//self.B, planes//self.B, kernel_size=kernel_size, stride=stride, \
                    padding=padding, bias=bias, groups=groups)
        if self.dropout_tbc > 0.0:
            self.drop_out = nn.Dropout(self.dropout_tbc)

    def forward(self, x):
        # print("x:", x.shape)
        n, c, h, w, d = x.size()
        x = x.contiguous().view(n*self.B, c//self.B, h, w, d)
        # print("x:", x.shape)
        h_o = (h - self.kernel_size + 2*self.padding) // self.stride + 1
        w_o = (w - self.kernel_size + 2*self.padding) // self.stride + 1
        d_o = (d - self.kernel_size + 2 * self.padding) // self.stride + 1
        x = self.conv(x)
        # print("x:", x.shape)
        x = x.view(n, self.out_planes, h_o, w_o, d_o)
        # print("x:", x.shape)
        if self.dropout_tbc > 0:
            x = self.drop_out(x)
            # print("x:", x.shape)
        # print("x:", x.shape)
        return x



if __name__ == '__main__':
    x = torch.randn(1,32,16,16,16)
    model = TiedBlockConv3d(32,64,1)
    print(model(x).shape)
