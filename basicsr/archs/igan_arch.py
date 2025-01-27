import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

def make_model(args, parent=False):
    return IIAN(args)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                # m.append(conv(n_feat, 2 * n_feat, 3, bias))
                # m.append(conv(2 * n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class IIA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(IIA, self).__init__()
        # self.Capattention = nn.Sequential(
        #     nn.Conv2d(channel, channel*2, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(channel*2, channel, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(channel, 1, 1, stride=1, padding=0, bias=False),
        # )
        self.Capattention = nn.Sequential(
            nn.Conv2d(channel,  channel//2, 3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channel//2, channel//4, 3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channel//4, channel, 1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, prevousinput,x):
        b, c, w, h = x.size()
        ii_feature = x - prevousinput
        atten = self.Capattention(ii_feature)


        return x*atten



## Residual  Block (RB)
class RB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True),
                 res_scale=1, dilation=2):
        super(RB, self).__init__()

        self.n_feat = n_feat
        self.gamma1 = 1.0
        self.conv_first = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                        act,
                                        conv(n_feat, n_feat, kernel_size, bias=bias)
                                        )

        self.res_scale = res_scale

    def forward(self, x):
        b,c,h,w = x.size()
        y1 = self.conv_first(x)
        y = y1 + x

        return y


## Local-source Residual Attention Group (LSRARG)
class RAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(RAG, self).__init__()
        ##
        # body = [RB(conv, n_feat, kernel_size, reduction, \
        #                               bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) for _ in
        #                            range(n_resblocks)]
        # self.rbs = nn.Sequential(*body)

        self.rbs = nn.ModuleList([RB(conv, n_feat, kernel_size, reduction, \
                                      bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) for _ in
                                   range(n_resblocks)])

        self.iia = (IIA(n_feat, reduction=reduction))
        self.conv_last = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        previous = x

        for i, l in enumerate(self.rbs):
            # x = l(x) + self.gamma*residual
            x = l(x)

        x = self.iia(previous,x)

        x = self.conv_last(x)
        x = x + previous

        return x


## Information-increment
@ARCH_REGISTRY.register()
class IGAN(nn.Module):
    def __init__(self, n_resgroups, n_resblocks,n_feats,reduction,rgb_range):
        super(IGAN, self).__init__()
        conv = default_conv

        kernel_size = 3
        scale = 4
        act = nn.ReLU(inplace=True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        modules_head = [conv(3, n_feats, kernel_size)]

        self.gamma = nn.Parameter(torch.zeros(1))

        self.n_resgroups = n_resgroups
        body = [RAG(conv, n_feats, kernel_size, reduction, \
                                       act=act, res_scale=1, n_resblocks=n_resblocks) for _ in
                                 range(n_resgroups)]
        self.body = nn.Sequential(*body)

        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)

        self.tail = nn.Sequential(*modules_tail)

        self.iia = IIA(channel=n_feats)


    def forward(self, x):
        b,n,c,h,w = x.size()
        x = x[:,n//2,:,:,:]*255.0
        
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)

        x = x + res

        x = self.tail(x)
        x = self.add_mean(x)
        x = x/255.0
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


