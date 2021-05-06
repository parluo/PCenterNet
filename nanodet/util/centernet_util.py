import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / (2)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / (2 )

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 )
    return min(r1, r2, r3)
    # 各个通道的topk
def _topk(heat, K=100):
    n, c, h, w = heat.shape #由坐标索引，要注意各个维度的含义顺序
    # topk 通道<=3
    scores, pos = torch.topk(heat.view(n,c, -1), K) 
    pos = pos % (h * w) # n*c*K n*c每个w*h上的前100
    topk_x = pos % w # n*c*k 
    # topk_y = (pos - topk_x)// w + 1
    topk_y= pos // h # 源码这里这么搞，如果再加上偏移预测，是不是有量化误差


    topk_scores, topk_score_pos = torch.topk(scores.view(n, -1), K) # n*K
    topk_cls = topk_score_pos // K # n*k 表n个前100的cls
    
    topk_x = topk_x.view(n, -1)[:,topk_score_pos].squeeze(1) # n*k
    topk_y = topk_y.view(n, -1)[:,topk_score_pos].squeeze(1)

    # out_pos = pos.view(n,-1)[:,topk_score_pos].squeeze(1)
    return topk_scores, topk_cls, topk_x, topk_y


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def multi_nms(heat,size_stop, size_start=3,step=2):
    """
    size_stop: final kernel size to maxpool the heatmap
    """
    steps = np.arange(size_start,size_stop,step)
    k = np.arange(len(steps))
    # n,c,h,w = heat.shape
    # result = torch.zeros(len(steps),n,c,h,w).cuda()
    result = None
    for i,ks in zip(k,steps):
        temp = _nms(heat, ks)
        if result is None:
            result = temp
        else:
            result = result + temp
    result = result/len(steps)
    return result
        


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def rotate(pos, theta):
    '''
    给定角度旋转坐标  
    pos: 坐标  
    theta: 角度, 单位:°  
    '''
    theta = np.pi * theta/180
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    out_pos = np.dot(rot_mat, pos)
    return out_pos

def get_affine_transform(center,
                         src_size,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    '''
    为了找到三组对应点, 然后求仿射变换的2*3的矩阵.  

    center: 原图中心点  
    src_size: 原图的尺寸, 如果int,就默认是正方形  
    rot: 旋转角度, 单位：°  
    output_size: 映射后的输出尺寸
    shift: 平移比例向量,0~1
    '''
    if not isinstance(src_size, np.ndarray) and not isinstance(src_size, list):
        src_size = np.array([src_size, src_size], dtype=np.float32)
    if not isinstance(output_size, np.ndarray):
        output_size = np.array(output_size, dtype=np.float32)

    pre_size = src_size # scale后面还有用
    shift_d = shift * pre_size
    r_d = -center # np.array([0,0])# 
    srcp3 = np.zeros((3,2), dtype=np.float32)
    dstp3 = np.zeros((3,2), dtype=np.float32)

    # srcp3 
    srcp3[0,:] = rotate(center+r_d,rot) + shift_d - r_d
    srcp3[1,:] = rotate(center-pre_size/2+r_d, rot) + shift_d - r_d
    srcp3[2,:] = rotate([center[0],center[1]-pre_size[1]/2]+r_d, rot) + shift_d - r_d
    
    # dstp3 
    dstp3[0,:] = output_size * 0.5
    # dstp3[1,:] = dstp3[1,:]
    dstp3[2,:] = [output_size[0]/2, 0]
    if inv:
        trans_matrix = cv2.getAffineTransform(dstp3, srcp3)
    else:
        trans_matrix = cv2.getAffineTransform(srcp3, dstp3)

    return trans_matrix
