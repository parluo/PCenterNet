"""
from MMDetection
"""

import torch.nn as nn
import torch.nn.functional as F
from ..module.conv import ConvModule
from ..module.init_weights import xavier_init
from .ffn import DeformConv


class CFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None
                 ):
        super(CFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        ## 以下是针对对shufflenetv2的改动 + line64
        self.up1 =  ConvModule(self.in_channels[0], self.in_channels[1]//2,3,padding=1)
        self.in_channels[0] = self.in_channels[1]//2

        self.leftright = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False)
            self.leftright.append(l_conv)

        self.updown = nn.ModuleList()
        for i in range(1, len(in_channels)):
            up = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                2, # kerner size: 2
                stride=2,
                padding=0, 
                output_padding=0,
                groups=out_channels, 
                bias=False)
            self.updown.append(up)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs[0] = self.up1(inputs[0])  # 针对sv2

        # build laterals
        pyrimaid = [
            toright(inputs[i])
            for i, toright in enumerate(self.leftright)
        ]

        # build top-down path
        nlevel = len(pyrimaid)-1
        for i in range(nlevel, 0, -1):
            pyrimaid[i-1] = self.updown[i-1](pyrimaid[i]) + pyrimaid[i-1]

        # build outputs
        outs = pyrimaid[0] # 最大size层
        return outs


# if __name__ == '__main__':
