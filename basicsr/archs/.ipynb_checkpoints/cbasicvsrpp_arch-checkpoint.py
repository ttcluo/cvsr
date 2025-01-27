import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings

from basicsr.archs.arch_util import flow_warp,make_layer
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class CBasicVSRPlusPlus(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped. Besides, we adopt the official DCN
    implementation and the version of torch need to be higher than 1.9.

    ``Paper: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment``

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_path=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ConvResidualBlocks(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = CConvResidualBlocks(5 * mid_channels, mid_channels, 3)

        self.upconv1R = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv1I = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2R = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)
        self.upconv2I = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hrR = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hrI = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_lastR = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_lastI = nn.Conv2d(64, 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the flows used for forward-time propagation \
                (current to previous). 'flows_backward' corresponds to the flows used for backward-time \
                propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated \
                features. Each key in the dictionary corresponds to a \
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats,phase_rot):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hrR = torch.cat((hr[0],hr[1],hr[3]), dim=1)
            hrI = torch.cat((hr[0],hr[2],hr[4]), dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            outR,outI = self.reconstruction(hrR,hrI)
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
            
            base = self.img_upsample(lqs[:, i, :, :, :])
            baseR = base*attcos-base*attsin
            baseI = base*attsin+base*attcos
            
            outR = outRup_5+baseR
            outI = outIup_5+baseI
            
            hr = torch.cat([outR,outI],dim=1)

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs,phase_rot):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats,phase_rot)


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)

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
    
class CConvResidualBlocks(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.preconvR = nn.Conv2d(192, 64, 3, 1, 1, bias=True)
        self.preconvI = nn.Conv2d(192, 64, 3, 1, 1, bias=True)
        self.num_out_ch = num_out_ch
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.main1 = ComplexResidualBlockNoBNbase()
        self.main2 = ComplexResidualBlockNoBNbase()
        #self.main3 = ComplexResidualBlockNoBNbase()

    def forward(self, feaR,feaI):
        feaRew = self.preconvR(feaR) - self.preconvI(feaI)
        feaINew = self.preconvR(feaI) + self.preconvI(feaR)
        fR = self.lrelu(feaRew)
        fI = self.lrelu(feaINew)
        fR,fI = self.main1(fR,fI)
        fR,fI = self.main2(fR,fI)
        #fR,fI = self.main3(fR,fI)
        return fR,fI

    
class ComplexResidualBlockNoBNbase(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv1R = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv1I = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2R = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2I = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xR, xI):
        identityR = xR
        identityI = xI
        x1R = self.conv1R(xR)-self.conv1I(xI)
        x1I = self.conv1R(xI)+self.conv1I(xR)
        x1R = self.relu(x1R)
        x1I = self.relu(x1I)
        outR = self.conv2R(x1R)-self.conv2I(x1I)
        outI =  self.conv2R(x1I)+self.conv2I(x1R)
        return identityR + outR * self.res_scale, identityI+outI* self.res_scale
# if __name__ == '__main__':
#     spynet_path = 'experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
#     model = BasicVSRPlusPlus(spynet_path=spynet_path).cuda()
#     input = torch.rand(1, 2, 3, 64, 64).cuda()
#     output = model(input)
#     print('===================')
#     print(output.shape)