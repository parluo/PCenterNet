from torch import nn
import torch
import math
from DCN.dcn_v2 import DCN
import numpy as np
from torch.utils import model_zoo
from os.path import join





def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,1, stride=1, bias=False, padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual 
    
    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x,1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x

    
class Tree(nn.Module):
    def __init__(self, levels, Block, in_channels, out_channels, stride=1, dilation=1,
                root_dim=0,level_root=False, root_kernel_size=1, root_residual=False):
        '''
            levels: 用来判断是不是root的递归传入参数，递归减一
            level_root: 根叶节点， 要cat x或者降采样的x
            root_dim: Root类要cat tree1的输出，out_channel至少2倍原out_channel
        '''
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:  # 
            root_dim += in_channels

        if levels == 1: # Tree分为是root和不是两种情况
            self.tree1 = Block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = Block(out_channels, out_channels, 1, dilation=dilation)
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        
        else:
            self.tree1 = Tree(levels-1, Block, in_channels, out_channels, stride, 
                                root_dim=0, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)

            self.tree2 = Tree(levels-1, Block, out_channels, out_channels,
                    root_dim=root_dim+out_channels, root_kernel_size=root_kernel_size,dilation=dilation, root_residual=root_residual)
        
        self.level_root = level_root
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride>1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        
        if in_channels != out_channels:
            self.project = nn.Sequential(\
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=0.1)
            )
    # residual 主要是传给basiclock的add用
    def forward(self, x, residual = None, children=None): 
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x # 降采样时，调整x
        residual = self.project(bottom) if self.project else bottom # 输入输出通道不一样时,调整x
        
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

        
        
        
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.stride = stride
    
    def forward(self, x, residual=None):
        '''
            residual: 从哪里接过来融合,为None，则就是普通的residual block
        '''
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual 
        out = self.relu(out)

        return out
    
    

class DLA34(nn.Module):
    def __init__(self,levels, channels, Block=BasicBlock, residual_root=False, pretrained=True):
        super().__init__()
        self.levels = levels # [1,1,1,2,2,1]
        self.channels = channels  # [16, 32, 64, 128, 256, 512]
         
        # 1
        self.base_layer = nn.Sequential(
            nn.Conv2d(3,self.channels[0], kernel_size=7, stride=1,padding=3,bias=False),
            nn.BatchNorm2d(self.channels[0],momentum=0.1),
            nn.ReLU(inplace=True)
        )
        # 1
        self.level0 = self._make_conv_level(self.channels[0], self.channels[0], self.levels[0])
        # 1
        self.level1 = self._make_conv_level(self.channels[0], self.channels[1], self.levels[1], stride=2)
        # 3
        self.level2 = Tree(self.levels[2], Block, self.channels[1], self.channels[2], 2, level_root=False, root_residual=residual_root)
        # 6
        self.level3 = Tree(self.levels[3], Block, self.channels[2], self.channels[3], 2, level_root=True,  root_residual=residual_root)
        # 6
        self.level4 = Tree(self.levels[4], Block, self.channels[3], self.channels[4], 2, level_root=True,  root_residual=residual_root)
        # 3
        self.level5 = Tree(self.levels[5], Block, self.channels[4], self.channels[5], 2, level_root=True,  root_residual=residual_root)

        if pretrained:
            self.load_pretrained_model()


    def _make_conv_level(self, in_channels, out_channel, nums, stride=1, dilation=1):
        '''
            in_channels: 输入通道；
            out_channels: 输出通道；
            nums: 卷积块个数；
        '''
        modules = []
        for i in range(nums):
            modules.extend([
                nn.Conv2d(in_channels, out_channel, kernel_size=3, 
                    stride=stride if i==0 else 1,
                    padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(out_channel, momentum=0.1), # n c w h 在n上对c的每一层w h的元素 （xi-mean)/(sqrt(D2+ delta)) * gamma + belta
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channel
        return nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.base_layer(x) # -> N 16 w h
        y = []
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

    
    def load_pretrained_model(self, name='dla34', data='imagenet', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights= torch.load(name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        # 补个fc以strict的load原模型
        self.fc = nn.Conv2d(self.channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)


