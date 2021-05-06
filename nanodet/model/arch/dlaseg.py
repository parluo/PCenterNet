from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import logging
import numpy as np
from os.path import join

import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from ..loss.centernet_loss import Qfloss,Dfloss,RegL1Loss,FocalLoss
from ...util.centernet_util import _nms,_topk,multi_nms
from ...util.visualization import overlay_bbox_cv

from .DCNv2.DCN.dcn_v2 import DCN
#from .DCNv2.dcn_v2 import DCNv2
# import DCN  # 安装后直接导入就行


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
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


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
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


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=0.1),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=7):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        # 向模块添加持久缓冲区。这通常用于注册不应被视为模型参数的缓冲区。
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))




    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        n,c,h,w = x.shape
        x = x.reshape(n,2,c//2, h, w)
        x = x.permute(0,1,3,4,2) # 
        x = F.softmax(x, dim=4)
        x = F.linear(x, self.project.type_as(x)) * 127 / self.reg_max #！！！！！！！！！
        return x

class Gflv2branch(nn.Module):
    def __init__(self, inchannels, midchannels=64, outchannels=1):
        super().__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        # gfl v2
        conf_vector = [nn.Conv2d(self.inchannels, self.midchannels, 1)]
        conf_vector += [nn.ReLU()]
        conf_vector += [nn.Conv2d(self.midchannels, self.outchannels, 1), nn.Sigmoid()]
        self.light_branch = nn.Sequential(*conf_vector)
    def forward(self, x):
        out = self.light_branch(x)
        return out

## version 1
class ASFF3(nn.Module):
    """
    fuse 3 features
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2, x3):
        """
        input shape: n c h w
        """
        c = x1.shape[1]
        weights = torch.cat([x1,x2,x3], dim=1)
        weights = weights.softmax(dim = 1)
        w1 = weights[:,0:c,:,:]
        w2 = weights[:,c:2*c,:,:]
        w3 = weights[:,2*c:3*c,:,:]
        out = w1 * x1 + w2 * x2 + w3 * x3
        return out

class ASFF(nn.Module):
    """
    fuse 3 features layers
    """
    def __init__(self, layer):
        super().__init__()
        self.channels = [20,20,20]
        self.out_channel = self.channels[layer] # 从3th出
        self.mid_channels = [12,12,12]
        self.conv0 = self.add_conv(self.channels[0], self.mid_channels[0],1,1)
        self.conv1 = self.add_conv(self.channels[1], self.mid_channels[1],1,1)
        self.conv2 = self.add_conv(self.channels[2], self.mid_channels[2],1,1)
        self.weight_layer = nn.Conv2d(36, 3, kernel_size=1, stride=1, padding=0) #  16 + 16 + 16 = 48
        self.out_layer = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1) #  pad = (kernel_size-1) // 2
        
    def add_conv(self, in_ch, out_ch, ksize, stride, leaky=True):
        """
        Add a conv2d / batchnorm / leaky ReLU block.
        Args:
            in_ch (int): number of input channels of the convolution layer.
            out_ch (int): number of output channels of the convolution layer.
            ksize (int): kernel size of the convolution layer.
            stride (int): stride of the convolution layer.
        Returns:
            stage (Sequential) : Sequential layers composing a convolution block.
        """
        stage = nn.Sequential()
        pad = (ksize - 1) // 2
        stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                        out_channels=out_ch, kernel_size=ksize, stride=stride,
                                        padding=pad, bias=False))
        stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
        if leaky:
            stage.add_module('leaky', nn.LeakyReLU(0.1))
        else:
            stage.add_module('relu6', nn.ReLU6(inplace=True))
        return stage

    def forward(self, x0, x1, x2):
        """
        input shape: n c h w
        """
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        merge_x = torch.cat([x0,x1,x2], dim=1)
        weights = self.weight_layer(merge_x)
        w = weights.softmax(dim = 1)
        out = x0*w[:,0:1,:,:] + x1*w[:,1:2,:,:] + x2*w[:,2:3,:,:]
        out = self.out_layer(out)
        return out
        


class QV1loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.loss_qfl = Qfloss(weight=1.0)
        ################################ nightcar ,zhengcha, sgdhoude 
        self.loss_qfl = FocalLoss(1) # 1  1  0.5 
        self.loss_dfl = Dfloss(weight=0.5) # 0.5 1 0.5
        self.loss_bbox = RegL1Loss(weight=0.1) # 0.1 0.25  0.1 
        self.loss_reg = RegL1Loss(weight=1) # 1 1 0.5
        self.integral = Integral(7) # 8-1
        


    def forward(self, preds, gt_meta):
        whloss1 = self.loss_dfl(preds['wh'], gt_meta['wh']) 
        wh = self.integral(preds['wh']) # 这里复制一份, val时的decode里面要（重复了）再积分一次
        whloss2 = self.loss_bbox(wh, gt_meta['wh'])
        whloss = whloss1 +whloss2
        regloss = self.loss_reg(preds['reg'], gt_meta['reg'])

        hmloss = self.loss_qfl(preds['hm'], gt_meta['hm'])
        loss = hmloss + whloss + regloss
        if(torch.isnan(loss).sum()>0):
            print("here")
        loss_states = dict(
                loss=loss,
                loss_hm=hmloss,
                loss_dfl=whloss1,
                loss_wh=whloss2,
                loss_reg=regloss
            )
        return loss, loss_states

class CenterNetloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_hm = FocalLoss(weight=1)
        # self.loss_hm = Qfloss(weight=1)
        self.loss_wh = RegL1Loss(weight=0.1)
        self.loss_reg = RegL1Loss(weight=1)

    def forward(self, preds, gt_meta):
        hmloss = self.loss_hm(preds['hm'],gt_meta['hm'])
        whloss = self.loss_wh(preds['wh'], gt_meta['wh']) 
        regloss = self.loss_reg(preds['reg'], gt_meta['reg'])
        loss = hmloss + whloss + regloss
        loss_states = dict(
                loss=loss,
                loss_hm=hmloss,
                loss_wh=whloss,
                loss_reg=regloss
            )
        return loss, loss_states


class DLAhead(nn.Module):
    def __init__(self, num_classes):
        super(DLAhead, self).__init__()
        self.num_classes = num_classes
        self.integral = Integral(7) 
        self.loss = QV1loss()
        # self.loss = CenterNetloss()

  


    def warp_boxes(self, boxes, M, width, height):
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes
    def post_process(self, preds, meta):
        """
            
        """
        result = self.decode(preds,meta)
        preds = {}
        warp_matrix = meta['warp_matrix'][0] if len(meta['warp_matrix'].shape)!=2 else meta['warp_matrix']
        img_height = meta['img_info']['height'].cpu().numpy() \
            if isinstance(meta['img_info']['height'], torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() \
            if isinstance(meta['img_info']['width'], torch.Tensor) else meta['img_info']['width']
    
        preds = {}    
        det_bboxes, det_labels = result

        det_bboxes = det_bboxes.cpu().numpy()
        det_bboxes[:, :4] = self.warp_boxes(det_bboxes[:,:4], np.linalg.inv(warp_matrix), img_width, img_height)
        classes = det_labels.cpu().numpy()
        for i in range(self.num_classes):
            inds = (classes == i).T.tolist() # 100*1 -> 1 100的list
            preds[i] = det_bboxes[inds].tolist()
        return preds # 


    # def decode(self, preds, meta,  K = 100):
    #     """
    #     解析输出特征图为bbox
    #     """
        
    #     # hm = preds['hm'].sigmoid() # *************** 【原始forward】
    #     hm = preds['hm']  #   ************** 带【gflv2】分支的forward, 不需sigmoid
    #     # hm = _nms(hm)
    #     hm = multi_nms(hm, 12) # 多重滤波

    #     # # dfl 的积分
    #     preds['wh'] = self.integral(preds['wh']) # 默认train_loss中积分更改preds,但结果积分了两次
       
    #     wh = preds['wh']
    #     reg = preds['reg']


    #     ###############
    #     n,c,h,w = hm.shape
    #     scores, classes, xs, ys = _topk(hm, K=K)
    #     wh = wh[:,:,ys,xs].squeeze(2)  # 注意索引的y x
    #     if reg is not None:
    #         reg = reg[:,:,ys,xs].squeeze(2)
    #         xs = xs + reg[:,1,:]
    #         ys = ys + reg[:,0,:]
    #     else:
    #         None
    #     scores = scores.view(n,K,1)
    #     classes = classes.view(n,K,1)
        
    #     bboxes = 4 * torch.stack([  xs - wh[:,1,:]/2,
    #                             ys - wh[:,0,:]/2,
    #                             xs + wh[:,1,:]/2,
    #                             ys + wh[:,0,:]/2], dim=2) 
    #     dets = torch.cat([bboxes, scores], dim=2)
    #     if dets.shape[0] > 1:
    #         print('warning: val batch_size > 1, please check up! ')
    #     return dets[0],classes[0]


    #### 2021.3.3 重新调整gt为wh， 重新调整解析顺序
    def decode(self, preds, meta,  K = 100):
        """
        解析输出特征图为bbox
        """
        
        # hm = preds['hm'].sigmoid() # *************** 【原始forward】
        hm = preds['hm']  #   ************** 带【gflv2】分支的forward, 不需sigmoid
        hm = _nms(hm)
        # hm = multi_nms(hm, 5) # 多重滤波

        # # dfl 的积分
        preds['wh'] = self.integral(preds['wh']) # 默认train_loss中积分更改preds,但结果积分了两次
       
        wh = preds['wh']
        reg = preds['reg']


        ###############
        n,c,h,w = hm.shape
        scores, classes, xs, ys = _topk(hm, K=K)
        wh = wh[:,:,ys,xs].squeeze(2)  # 注意索引的y x
        if reg is not None:
            reg = reg[:,:,ys,xs].squeeze(2)
            xs = xs + reg[:,0,:]
            ys = ys + reg[:,1,:]
        else:
            None
        scores = scores.view(n,K,1)
        classes = classes.view(n,K,1)
        
        bboxes = 4 * torch.stack([  xs - wh[:,0,:]/2,
                                ys - wh[:,1,:]/2,
                                xs + wh[:,0,:]/2,
                                ys + wh[:,1,:]/2], dim=2) 
        dets = torch.cat([bboxes, scores], dim=2)
        if dets.shape[0] > 1:
            print('warning: val batch_size > 1, please check up! ')
        return dets[0],classes[0]

        # for i in range(self.num_classes):
        #     inds = (dets[:,:,5] == i)
        #     out[i] = (dets[inds][:, :5]).cpu().numpy().tolist()
        # return out


    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            # cv2.imshow('det', result)
            cv2.imwrite('result/tempout.jpg',result)
            

    def forward(self, pred):
        return pred


class DLAfpn(nn.Module):
    def __init__(self):
        super(DLAfpn, self).__init__()
    def forward(self, x):
        return x
class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        self.head = DLAhead(heads['hm'])
        self.fpn = DLAfpn()


        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio)) # 512->128 down_ratio就是4
        self.last_level = last_level # 5
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels # list
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

#     gflv2 模块
        self.gf2_branch = Gflv2branch(6) # (4+1)*2 = 10

# ####    asff 
#         self.hm_y0 = nn.Sequential(
#                   nn.Conv2d(channels[self.first_level], head_conv,
#                     kernel_size=3, padding=1, bias=True),
#                   nn.ReLU(inplace=True),
#                   nn.Conv2d(head_conv, classes, 
#                     kernel_size=final_kernel, stride=1, 
#                     padding=final_kernel // 2, bias=True))
#         self.hm_y1 = nn.Sequential(
#                   nn.Conv2d(channels[self.first_level], head_conv,
#                     kernel_size=3, padding=1, bias=True),
#                   nn.ReLU(inplace=True),
#                   nn.Conv2d(head_conv, classes, 
#                     kernel_size=final_kernel, stride=1, 
#                     padding=final_kernel // 2, bias=True))

        
        # self.asff = ASFF(3)

    def forward_train(self, gt_meta):
        preds = self(gt_meta['img'])
        loss, loss_states = self.head.loss(preds, gt_meta)

        return preds, loss, loss_states


# ### [原forward]
#     def forward(self, x):
#         x = self.base(x)
#         x = self.dla_up(x)

#         end = self.last_level - self.first_level
#         for _ in range(end, len(x)):
#             x.pop()
#         self.ida_up(x, 0, len(x))
#         z = {}
#         for head in self.heads:
#             z[head] = self.__getattr__(head)(x[-1])
#         return z



# gflv2
    def forward(self, x):
        """
        gflv2 + gflv2分支 的前向计算

        hm：将是sigmoid以后的
        """
        x = self.base(x) # 512->32
        x = self.dla_up(x) #128->16

        end = self.last_level - self.first_level
        for _ in range(end, len(x)):
            x.pop()
            
        self.ida_up(x, 0, len(x))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x[-1])

        # gfl v2中的分支
        wh = z['wh']
        n,c,h,w = wh.shape
        wh = F.softmax(wh.reshape(n,2,c//2,h,w), dim=2) #  softmax后的
        whtopk, _ = torch.topk(wh, k=2, dim=2) # top 4 2
        whtopk_mean = torch.cat([whtopk, whtopk.mean(dim=2, keepdim=True)], dim=2) # n 2 5 h w
        stat = whtopk_mean.reshape(n,-1,h,w) # n 10 h w
        out_branch = self.gf2_branch(stat)
        z['hm'] = torch.clamp(z['hm'].sigmoid(),min=1e-4,max=1-1e-4) * out_branch
        # z['hm'] = z['hm'].sigmoid() * out_branch # n 1 h w  # 你不限制范围若==0, 则log将是nan

        return z


# # ### with asff
#     def forward(self, x):
#         x = self.base(x)
#         x = self.dla_up(x)

#         y = []
#         for i in range(self.last_level - self.first_level):
#             y.append(x[i].clone())
#         self.ida_up(y, 0, len(y))

#         z = {}
#         for head in self.heads:
#             z[head] = self.__getattr__(head)(y[-1])


#         hm0 = self.hm_y0(y[0])
#         hm1 = self.hm_y1(y[1])
#         hm2 = z['hm']
#         z['hm'] = self.asff(hm0, hm1, hm2) 

#         return z




#  ###  with asff + gflv2
#     def forward(self, x):
#         """
#         gflv2 + gflv2分支 的前向计算
#         """
#         x = self.base(x)
#         x = self.dla_up(x)

#         y = []
#         for i in range(self.last_level - self.first_level):
#             y.append(x[i].clone())
#         self.ida_up(y, 0, len(y))

#         z = {}
#         for head in self.heads:
#             z[head] = self.__getattr__(head)(y[-1])

#         hm0 = self.hm(y[0])
#         hm1 = self.hm(y[1])
#         hm2 = z['hm']
#         z['hm'] = self.asff(hm0, hm1, hm2) # 已经sigmoid过了 区间[0,1]

#         # gfl v2中的分支
#         wh = z['wh']
#         n,c,h,w = wh.shape
#         wh = wh.reshape(n,c//2,2,h,w) # 
#         whtopk, ind = torch.topk(wh, k=4, dim=1)
#         whtopk_mean = torch.cat([whtopk, whtopk.mean(dim=1, keepdim=True)], dim=1) # n 5 2 h w
#         stat = whtopk_mean.reshape(n,-1,h,w) # n 10 h w
#         out_branch = self.gf2_branch(stat)
#         z['hm'] = z['hm'].sigmoid() * out_branch # n 1 h w 

#         return z
    
    def inference(self, meta):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            preds = self(meta['img'])
            torch.cuda.synchronize()
            time2 = time.time()
            print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
            results = self.head.post_process(preds, meta)
            torch.cuda.synchronize()
            print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
        return results

    

    

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
  model = DLASeg('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model

