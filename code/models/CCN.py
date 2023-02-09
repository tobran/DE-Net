import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from collections import OrderedDict

class _cConvNd(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, padding_mode, cond_dim):
        super(_cConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))

        self.scales = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(cond_dim, in_channels*out_channels)),
            ]))
        self.shifts = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(cond_dim, in_channels*out_channels)),
            ]))

        self._initialize()
        self.reset_parameters()
        
    def _initialize(self):
        nn.init.zeros_(self.scales.linear2.weight.data)
        nn.init.ones_(self.scales.linear2.bias.data)
        nn.init.zeros_(self.shifts.linear2.weight.data)
        nn.init.zeros_(self.shifts.linear2.bias.data)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn. init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_cConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class cConv2d(_cConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros', cond_dim=256):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        kernel_size = [kernel_size, kernel_size]
        stride = [stride, stride]
        padding = [padding, padding]
        dilation = [dilation, dilation]
        super(cConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode, cond_dim)

    def conv2d_forward(self, input, weight, c):
        b_size, c_size, height, width = input.shape

        scale = self.scales(c).view(b_size, self.out_channels, self.in_channels, 1, 1)
        shift = self.shifts(c).view(b_size, self.out_channels, self.in_channels, 1, 1)

        weight = (weight * scale + shift)

        return F.conv2d(input.view(1, b_size*c_size, height, width),
                        weight.view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
                        None, self.stride, self.padding,
                        self.dilation, b_size).view(-1, self.out_channels, height, width)# + self.bias

    def forward(self, x, c):
        return self.conv2d_forward(x, self.weight, c)

