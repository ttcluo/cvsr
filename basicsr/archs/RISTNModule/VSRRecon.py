import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .model_utils import split, merge, injective_pad, psi

class Reconsturcture(nn.Module):
    def __init__(self, channels):
        super(Reconsturcture, self).__init__()

        self.eaualization = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU()
        )

        self.reconstruct = nn.Sequential(
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        )


    def forward(self, x):
        x = self.eaualization(x)
        # x = self.psi.inverse(x)
        # x = self.psi.inverse(x)
        x = self.deconv(x)
        x = self.reconstruct(x)

        return x
