import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet
@ARCH_REGISTRY.register()
class CVSR2(nn.Module):
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

        # backward branch
        out_pre = []
        out_l = []
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
            
            outRup_3 = self.lrelu(self.conv_hrR(outRup_2)-self.conv_hrI(outIup_2))
            outIup_3 = self.lrelu(self.conv_hrR(outIup_2)+self.conv_hrI(outRup_2))
            
            outRup_4 = self.conv_lastR(outRup_3)-self.conv_lastI(outIup_3)
            outIup_4 = self.conv_lastR(outIup_3)+self.conv_lastI(outRup_3)

            #lastoutput = out
            attcos = torch.cos(phase_rot)
            attsin = torch.sin(phase_rot)
            
            outRup_5 = outRup_4*attcos - outIup_4*attsin
            outIup_5 = outRup_4*attsin + outIup_4*attcos
            
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            baseR = base*attcos-base*attsin
            baseI = base*attsin+base*attcos
            
            
            outR = outRup_5+baseR
            outI = outIup_5+baseI
            
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


@ARCH_REGISTRY.register()
class CIconVSR2(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)
        
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

    def pad_spatial(self, x):
        """ Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def forward(self, x,phase_rot):
        b, n, _, h_input, w_input = x.size()

        x = self.pad_spatial(x)
        h, w = x.shape[3:]

        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
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
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)
            
            
            outR = self.lrelu(self.fusionR(feat_prop)-self.fusionI(out_l[i]))
            outI = self.lrelu(self.fusionR(out_l[i])+self.fusionI(feat_prop))
            
            outRup = self.lrelu(self.pixel_shuffle(self.upconv1R(outR)-self.upconv1I(outI)))
            outIup = self.lrelu(self.pixel_shuffle(self.upconv1R(outI)+self.upconv1I(outR)))
            
            outRup_2 = self.lrelu(self.pixel_shuffle(self.upconv2R(outRup)-self.upconv2I(outIup)))
            outIup_2 = self.lrelu(self.pixel_shuffle(self.upconv2R(outIup)+self.upconv2I(outRup)))
            
            outRup_3 = self.lrelu(self.conv_hrR(outRup_2)-self.conv_hrI(outIup_2))
            outIup_3 = self.lrelu(self.conv_hrR(outIup_2)+self.conv_hrI(outRup_2))
            
            outRup_4 = self.conv_lastR(outRup_3)-self.conv_lastI(outIup_3)
            outIup_4 = self.conv_lastR(outIup_3)+self.conv_lastI(outRup_3)

            #lastoutput = out
            attcos = torch.cos(phase_rot)
            attsin = torch.sin(phase_rot)
            
            outRup_5 = outRup_4*attcos - outIup_4*attsin
            outIup_5 = outRup_4*attsin + outIup_4*attcos
            
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            baseR = base*attcos-base*attsin
            baseI = base*attsin+base*attcos
            
            
            outR = outRup_5+baseR
            outI = outIup_5+baseI
            
            out_l[i] = torch.cat([outR,outI],dim=1)

        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]


class EDVRFeatureExtractor(nn.Module):

    def __init__(self, num_input_frame, num_feat, load_path):

        super(EDVRFeatureExtractor, self).__init__()

        self.center_frame_idx = num_input_frame // 2

        # extract pyramid features
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=64)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        # TSA fusion
        return self.fusion(aligned_feat)


if __name__=="__main__":
    network=CVSR2(64).cuda()
    print('# model parameters:', sum(param.numel() for param in generator.parameters()))
    inputdata=torch.rand(4,7,3,50,50).cuda()
    output,_=network(inputdata)
    print(output.shape)
