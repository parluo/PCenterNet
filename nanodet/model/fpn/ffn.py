from torch import nn
import torch
import math
# from DCN.dcn_v2 import DCN
from torch.nn import Conv2d as DCN
import numpy as np
from torch.utils import model_zoo
from os.path import join


def file_up_weight(up):
    w = up.weight.data
    f = math.ceil(w.size(2)/2) # 向上取整
    c = (2 * f -1-f%2)/(2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0,0,i,j] = (1 - math.fabs(i/f-c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0,0,:,:]

class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DCN(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, dilation=1, groups=1)
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class IDAUP(nn.Module):
    def __init__(self,out_channels,channels, up_f):
        super().__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])

            # DCN - 通道变化的
            proj = DeformConv(c,out_channels)

            # DCN- 通道不变的
            node = DeformConv(out_channels,out_channels)
            
            # 逆卷积 - size翻倍
            up = nn.ConvTranspose2d(out_channels,out_channels,f*2, stride=f, padding=f//2, output_padding=0,groups=out_channels, bias=False)
            file_up_weight(up)  

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
    
    def forward(self, layers, startp, endp):
        for i in range(startp+1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

# 就是调IDAUP
class DLAUP(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super().__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels)-1):
            j= -i-2
            setattr(self, 'ida_{}'.format(i), IDAUP(channels[j], in_channels[j:], scales[j:]//scales[j]))
            scales[j+1 :] = scales[j]
            in_channels[j+1:] = [channels[j] for _ in channels[j+1:]]
        
    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers)-i-2, len(layers))
            out.insert(0, layers[-1])
        return out

# feature fusion network
class FFN(nn.Module):
    def __init__(self, channels,down_ratio, last_level,out_channel=0):
        super().__init__()
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.channels = channels
        scales = [2**i for i in range(len(channels[self.first_level:]))]
        # 1
        self.up1 =  DeformConv(channels[self.first_level], channels[self.first_level+1]//2)
        
        channels[self.first_level] = channels[self.first_level+1]//2
        # 2
        self.dla_up = DLAUP(self.first_level, channels[self.first_level:], scales)
        if out_channel == 0:
            out_channel = channels[self.first_level]
        # 3
        self.ida_up = IDAUP(out_channel, channels[self.first_level:self.last_level],[2**i for i in range(self.last_level - self.first_level)])
            

    def forward(self,x):
        x[self.first_level] = self.up1(x[self.first_level]) # 特地提升通道

        x = self.dla_up(x)

        end = self.last_level - self.first_level
        for _ in range(end, len(x)):
            x.pop()
            
        self.ida_up(x, 0, len(x))
        return x[-1] #  返回最后一层

    