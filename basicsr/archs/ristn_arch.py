from .RISTNModule.RRIN import SRRIN
from .RISTNModule.RDBCLSTM import CLSTM
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as  np
from torch.utils.checkpoint import checkpoint
from .RISTNModule.VSRRecon import Reconsturcture
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class RISTN(nn.Module):
    def __init__(self,temporal_depth,growth_rate,spatial_path):
        super(RISTN, self).__init__()
        self.realtionlen = 5

        netG = SRRIN()
        netG = torch.nn.DataParallel(netG,device_ids=[0])
        #netG.load_state_dict(torch.load('./RISTNsp.pth'))
        #netG = netG.cuda()
        self.sptioCNN = netG
        self.sptioCNN.load_state_dict(torch.load(spatial_path))
        #self.sptioCNN = torch.nn.DataParallel(self.sptioCNN, device_ids=[0])
        self.temporalRNN = CLSTM(320, 3, 16, 10)
        # self.recon = Reconsturcture(256)
        self.trainMode = True
        self.FW = nn.Conv2d(512,256,1,1,0)

        self.convertsTot = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=320, kernel_size=1, stride=1, padding=0, bias=False),
        )


        self.eaualization = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )


        self.convertTtos = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )


    def calc_sp(self, x):
        x = x.transpose(0, 1)  # now is seq_len,B,C,H,W
        t = None
        t2 = None
        for i in range(self.realtionlen):
            ax = x[i]
            ax_s = self.sptioCNN(ax)
            ax = self.convertsTot(ax_s)
            ax = torch.unsqueeze(ax, 0)
            ax_s = torch.unsqueeze(ax_s, 0)

            if i == 0:
                t = ax
                t2 = ax_s
            else:
                t = torch.cat((t, ax), 0)
                t2 = torch.cat((t2, ax_s), 0)
        return t, t2

    def forward(self, x):
        b,n,c,h,w = x.size()
        orispatial = []
        spatialf = []
        for i in range(n):
            a = self.sptioCNN(x[:,i,:,:,:])
            orispatial.append(a)
            a = self.convertsTot(a)
            spatialf.append(a)
        spatialf = torch.stack(spatialf,dim=0)
        tempf,_ = self.temporalRNN(spatialf)

        out = []
        for i in range(n):
            xori = orispatial[i]
            x = self.convertTtos(tempf[i])
            newf = torch.cat([x,xori],dim=1)
            x = self.FW(newf)
            x = self.eaualization(x)
            sr_frame = self.sptioCNN.module.reconstructure(x)
            out.append(sr_frame)

        final = torch.stack(out, dim=1)
        return final[:,n//2,:,:,:]

