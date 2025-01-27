import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet
@ARCH_REGISTRY.register()
class CVSR3(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)
        # for p in self.spynet.parameters():
        #     p.requires_grad = False

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.fusionR = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.fusionI = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        
        self.upconv1R = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv1I = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        
        self.upconv2R = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.upconv2I = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hrR = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hrI = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_lastR = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_lastI = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x, phase_rot):
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()
        
        attcos = torch.cos(phase_rot)
        attsin = torch.sin(phase_rot)
        
        # backward branch
        out_pre = []
        out_l = []
        phaselist = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)
            
            #out = torch.cat([out_l[i], feat_prop], dim=1)
            
            outR = self.lrelu(self.fusionR(feat_prop)-self.fusionI(out_l[i]))
            outI = self.lrelu(self.fusionR(out_l[i])+self.fusionI(feat_prop))
            
            
            outRup = self.lrelu(self.pixel_shuffle(self.upconv1R(outR)-self.upconv1I(outI)))
            outIup = self.lrelu(self.pixel_shuffle(self.upconv1R(outI)+self.upconv1I(outR)))
            
            outRup_2 = self.lrelu(self.pixel_shuffle(self.upconv2R(outRup)-self.upconv2I(outIup)))
            outIup_2 = self.lrelu(self.pixel_shuffle(self.upconv2R(outIup)+self.upconv2I(outRup)))
            
            outRup_5 = outRup_2*attcos - outIup_2*attsin
            outIup_5 = outRup_2*attsin + outIup_2*attcos
            
            outRup_3 = self.lrelu(self.conv_hrR(outRup_5)-self.conv_hrI(outIup_5))
            outIup_3 = self.lrelu(self.conv_hrR(outIup_5)+self.conv_hrI(outRup_5))
            
            outRup_4 = self.conv_lastR(outRup_3)-self.conv_lastI(outIup_3)
            outIup_4 = self.conv_lastR(outIup_3)+self.conv_lastI(outRup_3)

            #lastoutput = out
            
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            baseR = base*attcos-base*attsin
            baseI = base*attsin+base*attcos
            
            outR = outRup_4+baseR
            outI = outIup_4+baseI
            
            out_l[i] = torch.cat([outR,outI],dim=1)

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)

# class ComplexConvResidualBlocks(nn.Module):

#     def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
#         super().__init__()
#         self.preconv = nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True)
#         self.num_out_ch = num_out_ch
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.main = nn.Sequential(
#             make_layer(ComplexResidualBlockNoBNbase, num_block, num_feat=num_out_ch))

#     def forward(self, fea):
#         f = self.lrelu(self.preconv(fea))
#         f = self.main(f)
#         return f





if __name__=="__main__":
    network=CVSR3(64).cuda()
    print('# model parameters:', sum(param.numel() for param in generator.parameters()))
    inputdata=torch.rand(4,7,3,50,50).cuda()
    output,_=network(inputdata)
    print(output.shape)
