import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
# torch.nn.functional.conv2d(input,weight,bias=None,stride=1,padding=0,dilation=1,groups=1)
# input-------输入tensor大小（minibatch，in_channels，iH, iW）
# weight------权重大小（out_channels, [公式], kH, kW）
laplace_kernel = torch.tensor(
    [[0,1,0],
    [1,-4.0,1],
    [0,1,0]]
).cuda().repeat(2,20,1,1)

#  小数的g，当作正例，
#  此前的qfloss中： |g-p|^2*(glogp + (1-g)*log(1-p)) 你会发现小数的g又当正例又当负例
def qfloss(pred, gt, beta=2.0):
    pos_inds = (gt>0).float()
    neg_inds = gt.lt(1).float() # neg  这里小数gt也要是负例

    factor = torch.pow(gt - pred, 2) # orignal: 4
    neg_factor = torch.pow((1-gt),1) # 仔细对比会发现qfloss相对于原使用的focalloss少的就是这一项压缩负例损失的因子

    pos_loss = torch.log(pred) * gt * pos_inds
    neg_loss = torch.log(1 - pred) * (1-gt) * neg_inds * neg_factor

    loss = (pos_loss + neg_loss) * factor 
    loss = loss.sum()
    nums = (gt==1).sum() # 
    if nums == 0:
        loss = loss
    else:
        loss = loss/ (1e-4 + nums)
    return loss # 返回整体的loss

# def qfloss(preds, target, beta=2.0):
#     scale_factor = (preds - target).abs().pow(beta) # |y-p|^2
#     loss = F.binary_cross_entropy_with_logits(preds, target, reduction='none') * scale_factor #  * (1+target)
#     # 这里我们是非anchor的，含有大量负样本，正负都给比例因子|y-p|^2，不太好，要更关注正例
#     nums = (target==1).sum() # 低置信度的点会平分掉loss,太多点了
    
# # # #  shape_loss用来约束hm的形状  sqf loss
# # # #  对二阶梯度正的都加约束
# #     flit_pred = F.conv2d(preds, weight=laplace_kernel, bias=None, stride=1,padding=1)
# #     flit_pred = torch.tanh(flit_pred) # [-1, 1]
# #     flit_pred = flit_pred[flit_pred>0]
# #     exp_shape_loss = torch.exp(flit_pred)-1
# #     exp_shape_loss = exp_shape_loss.sum()# + 

#     loss = loss.sum()

#     # loss = loss.sum()
#     if nums == 0:
#         loss = loss
#     else:
#         loss = loss/ (1e-4 + nums)
#     return loss # 返回整体的loss

def dfloss(pred, gt):
    n,c,h,w= pred.shape
    pred = pred.reshape(n,2,c//2, h, w) 
    pred = pred.permute(0,1,3,4,2) # 
    mask = (gt>0)

    pos_gt = gt[mask]
    pos_gt = pos_gt * (pred.shape[-1]-1) / 127 # 0-1 -> 0-15 #16得减一
    
    pos_pred = pred[mask] # cross_entropy = softmax + log + cross entropy
    

    # pos_pred = pos_pred # 让softmax更锐利 当时会导致原始的输出区分度不够

    id_left = pos_gt.long()
    id_right = id_left + 1
    w_left = id_right.float() - pos_gt
    w_right = pos_gt - id_left.float()

    # sigmoid(preds) + binary_cross_entropy()
    loss_dfl = F.cross_entropy(pos_pred, id_left, reduction='none') * w_left + w_right * F.cross_entropy(pos_pred, id_right, reduction='none')
    loss_dfl = loss_dfl.mean() # wh两个通道都属于一个点上的wh
    return loss_dfl

def focalloss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 1) # orignal: 4  对于gt是0的并没有影响，但是中心点周围的小数gt，将会得到一个小数权重，降低对loss贡献

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds # focal loss hyper parameter: alpha=2
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

# # #  shape_loss用来约束hm的形状  sqf loss
# # #  对二阶梯度正的都加约束
#     flit_pred = F.conv2d(preds, weight=laplace_kernel, bias=None, stride=1,padding=1)
#     flit_pred = torch.tanh(flit_pred) # [-1, 1]
#     flit_pred = flit_pred[flit_pred>0]
#     exp_shape_loss = torch.exp(flit_pred)-1
#     exp_shape_loss = exp_shape_loss.sum()# + 



    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

# 出于性能的考虑，原计算方式中，先挑选出前128个点，再计算，好处是生成gt负担小，而不是生成c*128*128的gt图,缺点是理解麻烦

class Dfloss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    
    def forward(self, preds, gt):
        loss = dfloss(preds, gt)
        loss = loss * self.weight
        return loss

# 全给0， loss:9.2085
class Qfloss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, preds, gt):
        preds  = torch.clamp(preds.sigmoid(),min=1e-4,max=1-1e-4)
        loss = qfloss(preds, gt) 
        return loss * self.weight

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, weight):
        super(FocalLoss, self).__init__()
        self.neg_loss = focalloss
        self.weight = weight

    def forward(self, out, gt):
        # out  = torch.clamp(out.sigmoid(),min=1e-4,max=1-1e-4) # *********原始forward**************8
        loss = self.neg_loss(out, gt)
        return loss * self.weight
        
class RegL1Loss(nn.Module):
    """
    loss * 2
    相当于w h两个的loss仅除了1倍的num_pos
    """

    def __init__(self, weight):
        super().__init__()
        self.wegiht = weight
    def forward(self, preds, gt):
        # # 这里preds*mask, gt*mask结果就是regloss一直0.21，为啥  某次特殊意外？
        # # 
        # mask = gt>0
        
        # preds = preds[mask]
        # if preds.shape[0]==0:
        #     return torch.tensor(0).cuda().float()# 防止gt没有要回归的
            
        # gt = gt[mask]
        # loss = F.l1_loss(preds, gt, reduction='mean')*2 # 如果按作者的原写法,sum/obj_num，应是2倍loss
        # # loss = loss / (mask.sum() + 1e-4)
        # return loss * self.wegiht
        # 这里preds*mask, gt*mask结果就是regloss一直0.21，为啥  某次特殊意外？
        # 
        mask = gt.gt(0).float()
        loss = F.l1_loss(preds*mask, gt*mask, reduction='none') # 如果按作者的原写法,sum/obj_num，应是2倍loss
        loss = loss.sum()
        loss = loss / (mask.sum() + 1e-4)
        return loss * self.wegiht