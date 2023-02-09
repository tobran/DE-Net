import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from .CCN import cConv2d


class DCBlock(nn.Module):
    def __init__(self, in_ch, cond_dim=256):
        super(DCBlock, self).__init__()
        self.conv = cConv2d(in_ch, 8, 3, 1, 1, cond_dim=cond_dim)
        self.conv_weight = nn.Conv2d(8, 1, 3, 1, 1, bias=False)
        self.conv_bias = nn.Conv2d(8, 1, 3, 1, 1, bias=False)

    def forward(self, img, mask, c):
        h = self.conv(mask, c)
        weight = self.conv_weight(h)
        bias = self.conv_bias(h)
        return weight * img + bias


class CAFF(nn.Module):
    def __init__(self, num_features, cond_dim=256):
        super(CAFF, self).__init__()

        self.num_features = num_features
        
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(cond_dim, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(cond_dim, num_features)),
            ]))

        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, c):
        gamma = self.fc_gamma(c).view(-1, self.num_features, 1, 1)
        beta = self.fc_beta(c).view(-1, self.num_features, 1, 1)
        return gamma * x + beta


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetDown, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias = False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, h):
        m = h
        h = self.conv(h)
        h = self.bn(h)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return h, m


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 4, 1, 0)

    def forward(self, h):
        h = self.conv2(self.act(self.conv1(h)))
        return h


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.main = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(in_dim, in_dim)),
            ('relu1',nn.LeakyReLU(0.2, inplace=True)),
            ('linear2',nn.Linear(in_dim, out_dim)),
            ]))

    def forward(self, h):
        h = self.main(h)
        return h


class UNetYp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetYp, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, h, skip_input):
        h = F.interpolate(h, scale_factor=2)
        h = self.conv(h)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = h + skip_input
        return h


class DEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.CAFF = CAFF(out_ch)
        self.DCBlk = DCBlock(out_ch)
        self.fc = nn.Sequential(OrderedDict([
            ('linear2',nn.Linear(256*4, out_ch)),
            ('sigmoid',nn.Sigmoid()),
            ]))

        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc.linear2.weight.data)
        nn.init.zeros_(self.fc.linear2.bias.data)

    def forward(self, h, skip, mask, y, ti):
        h = F.interpolate(h, scale_factor=2)
        h = self.conv(h)
        h_m = nn.LeakyReLU(0.2, inplace=True)(self.CAFF(h, y))
        h_s = nn.LeakyReLU(0.2, inplace=True)(self.DCBlk(h, mask, y))
        weights = self.fc(ti).unsqueeze(-1).unsqueeze(-1)
        h = weights*h_m + (1-weights)*h_s
        h = h + skip
        return h


class CWP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CWP, self).__init__()
        self.img_enc = ConvBlock(in_ch, out_ch)
        self.txt_enc = MLP(out_ch, out_ch)
        self.fuse = MLP(256*2, 256*2)
        self.fc = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256*4, 256*4)),
            ('relu1',nn.LeakyReLU(0.2,inplace=True)),
        ]))

    def forward(self, v, t):
        img_emb = self.img_enc(v).view(v.size(0),-1)
        txt_emb = self.txt_enc(t)
        ti = self.fuse(torch.cat((img_emb, txt_emb), dim=1))
        tid = ti[:,:256] - ti[:,256:]
        tim = ti[:,:256] * ti[:,256:]
        ti = self.fc(torch.cat((tid, tim, img_emb, txt_emb), dim=1))
        return ti


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.ext = nn.Conv2d(3, ngf * 1, 3, 1, 1)#256

        self.down1 = UNetDown(ngf * 1,  ngf * 2)#128
        self.down2 = UNetDown(ngf * 2,  ngf * 4)#64
        self.down3 = UNetDown(ngf * 4,  ngf * 8)#32
        self.down4 = UNetDown(ngf * 8,  ngf * 8)#16
        self.down5 = UNetDown(ngf * 8,  ngf * 8)#8
        self.down6 = UNetDown(ngf * 8,  ngf * 8)#4

        self.yp1 = UNetYp(ngf * 8, ngf * 8)#8
        self.yp2 = UNetYp(ngf * 8, ngf * 8)#16
        self.yp3 = UNetYp(ngf * 8, ngf * 8)#32
        self.yp4 = UNetYp(ngf * 8, ngf * 4)#64
        self.yp5 = UNetYp(ngf * 4, ngf * 2)#128
        self.yp6 = UNetYp(ngf * 2, ngf * 1)#256

        self.up1 = DEBlock(ngf * 8, ngf * 8)#8
        self.up2 = DEBlock(ngf * 8, ngf * 8)#16
        self.up3 = DEBlock(ngf * 8, ngf * 8)#32
        self.up4 = DEBlock(ngf * 8, ngf * 4)#64
        self.up5 = DEBlock(ngf * 4, ngf * 2)#128
        self.up6 = DEBlock(ngf * 2, ngf * 1)#256

        self.cwp = CWP(ngf * 8, 256)

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf * 1, 3, 3, 1, 1),
            nn.Tanh(),
        )
        
    def forward(self, img, c):
        h = self.ext(img)

        d1, s1 = self.down1(h)
        d2, s2 = self.down2(d1)
        d3, s3 = self.down3(d2)
        d4, s4 = self.down4(d3)
        d5, s5 = self.down5(d4)
        d6, s6 = self.down6(d5)


        m1 = self.yp1(d6, s6)#8
        m2 = self.yp2(m1, s5)#16
        m3 = self.yp3(m2, s4)#32
        m4 = self.yp4(m3, s3)#64
        m5 = self.yp5(m4, s2)#128
        m6 = self.yp6(m5, s1)#256

        ti = self.cwp(d6, c)

        u1 = self.up1(d6, s6, m1, c, ti)#8
        u2 = self.up2(u1, s5, m2, c, ti)#16
        u3 = self.up3(u2, s4, m3, c, ti)#32
        u4 = self.up4(u3, s3, m4, c, ti)#64
        u5 = self.up5(u4, s2, m5, c, ti)#128
        u6 = self.up6(u5, s1, m6, c, ti)#256

        fake = self.conv_img(u6)
        return fake


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.block0 = D_Block(3, ndf, 4, 2, 1, bn=False)#128
        self.block1 = D_Block(ndf * 1, ndf * 2, 4, 2, 1)#64
        self.block2 = D_Block(ndf * 2, ndf * 4, 4, 2, 1)#32
        self.block3 = D_Block(ndf * 4, ndf * 8, 4, 2, 1)#16
        self.block4 = D_Block(ndf * 8, ndf * 8, 4, 2, 1)#8
        self.block5 = D_Block(ndf * 8, ndf * 8, 4, 2, 1)#4

    def forward(self,out):

        h = self.block0(out)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        return h


class NetC(nn.Module):
    def __init__(self, ndf):
        super(NetC, self).__init__()
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 8+256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):
        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out 


class D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bn=True):
        super(D_Block, self).__init__()
        self.bn = bn
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        if bn==True:
            self.batchnorm = nn.BatchNorm2d(out_ch)
        else:
            self.batchnorm = None

    def forward(self, h, y=None):
        h = self.conv(h)
        if self.bn==True:
            h = self.batchnorm(h)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return h