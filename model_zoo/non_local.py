import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np


def get_nonlocal_block(block_type):
    block_dict = {'nl': NonLocal, 'bat': BATBlock}
    if block_type in block_dict:
        return block_dict[block_type]
    else:
        raise ValueError("UNKOWN NONLOCAL BLOCK TYPE:", block_type)


class NonLocalModule(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(NonLocalModule, self).__init__()

    def init_modules(self):
        for name, m in self.named_modules():
            if len(m._modules) > 0:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif len(list(m.parameters())) > 0:
                raise ValueError("UNKOWN NONLOCAL LAYER TYPE:", name, m)


class NonLocal(NonLocalModule):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """

    def __init__(self, inplanes, use_scale=True, **kwargs):
        planes = inplanes // 2
        self.use_scale = use_scale

        super(NonLocal, self).__init__(inplanes)
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1,
                           stride=1, bias=True)
        self.bn = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            att = att.div(c**0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, h, w)

        x = self.z(x)
        x = self.bn(x) + residual

        return x


class BATransform(nn.Module):

    def __init__(self, in_channels, s, k):
        super(BATransform, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, k, 1),
                                   nn.BatchNorm2d(k),
                                   nn.ReLU(inplace=True))
        self.conv_p = nn.Conv2d(k, s * s * k, [s, 1])
        self.conv_q = nn.Conv2d(k, s * s * k, [1, s])
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))
        self.s = s
        self.k = k
        self.in_channels = in_channels

    def extra_repr(self):
        return 'BATransform({in_channels}, s={s}, k={k})'.format(**self.__dict__)

    def resize_mat(self, x, t):
        n, c, s, s1 = x.shape
        assert s == s1
        if t <= 1:
            return x
        x = x.view(n * c, -1, 1, 1)
        x = x * torch.eye(t, t, dtype=x.dtype, device=x.device)
        x = x.view(n * c, s, s, t, t)
        x = torch.cat(torch.split(x, 1, dim=1), dim=3)
        x = torch.cat(torch.split(x, 1, dim=2), dim=4)
        x = x.view(n, c, s * t, s * t)
        return x

    def forward(self, x):
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.s, 1))
        cp = F.adaptive_max_pool2d(out, (1, self.s))
        p = self.conv_p(rp).view(x.size(0), self.k, self.s, self.s)
        q = self.conv_q(cp).view(x.size(0), self.k, self.s, self.s)
        p = torch.sigmoid(p)
        q = torch.sigmoid(q)
        p = p / p.sum(dim=3, keepdim=True)
        q = q / q.sum(dim=2, keepdim=True)
        p = p.view(x.size(0), self.k, 1, self.s, self.s).expand(x.size(
            0), self.k, x.size(1) // self.k, self.s, self.s).contiguous()
        p = p.view(x.size(0), x.size(1), self.s, self.s)
        q = q.view(x.size(0), self.k, 1, self.s, self.s).expand(x.size(
            0), self.k, x.size(1) // self.k, self.s, self.s).contiguous()
        q = q.view(x.size(0), x.size(1), self.s, self.s)
        p = self.resize_mat(p, x.size(2) // self.s)
        q = self.resize_mat(q, x.size(2) // self.s)
        y = p.matmul(x)
        y = y.matmul(q)

        y = self.conv2(y)
        return y


class BATBlock(NonLocalModule):

    def __init__(self, in_channels, r=2, s=4, k=4, dropout=0.2, **kwargs):
        super().__init__(in_channels)

        inter_channels = in_channels // r
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(inplace=True))
        self.batransform = BATransform(inter_channels, s, k)
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        xl = self.conv1(x)
        y = self.batransform(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x

    def init_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
