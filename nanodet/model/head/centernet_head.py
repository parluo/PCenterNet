from torch import nn
import torch
import torch.nn.functional as F
import math
import cv2
from DCN.dcn_v2 import DCN
import numpy as np
from torch.utils import model_zoo
from os.path import join
from ..loss.centernet_loss import Qfloss, Dfloss, RegL1Loss,FocalLoss
from ...util.centernet_util import _nms, _topk,multi_nms
from ...util.visualization import overlay_bbox_cv
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

class CenterNetHead(nn.Module):
    def __init__(self, 
                out_classes,   
                input_channel,
                final_kernal, 
                head_conv=256,
                loss_hm_fl_weight=1.0,
                loss_wh_dfl_weight= 0.5,
                loss_wh_bbox = 0.1,
                loss_reg_weight = 1
                ):
        super().__init__()
        self.out_classes = out_classes
        self.num_classes = self.out_classes[0] # voc: 20 coco:80
        self.outchannels = 0
        for i in out_classes:
            self.outchannels = self.outchannels + i

        self.loss_hm = FocalLoss(weight=loss_hm_fl_weight)
        self.loss_dfl = Dfloss(weight=loss_wh_dfl_weight)
        self.loss_wh = RegL1Loss(weight=loss_wh_bbox)
        self.loss_reg = RegL1Loss(weight=loss_reg_weight)

        # self.hm =  nn.Sequential(
        #                 nn.Conv2d(input_channel, head_conv, kernel_size=3, padding=1, bias=True),
        #                 nn.ReLU(inplace=True),
        #                 nn.Conv2d(head_conv, self.out_classes[0], kernel_size=final_kernal, stride=1, padding=final_kernal // 2, bias=True)
        #             )
        # self.wh =  nn.Sequential(
        #     nn.Conv2d(input_channel, head_conv, kernel_size=3, padding=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(head_conv, self.out_classes[1], kernel_size=final_kernal, stride=1, padding=final_kernal // 2, bias=True)
        # )
        # self.reg = nn.Sequential(
        #                 nn.Conv2d(input_channel, head_conv, kernel_size=3, padding=1, bias=True),
        #                 nn.ReLU(inplace=True),
        #                 nn.Conv2d(head_conv, self.out_classes[2], kernel_size=final_kernal, stride=1, padding=final_kernal // 2, bias=True)
        #             )

        self.mixhead =  nn.Sequential(
                        nn.Conv2d(input_channel, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, self.outchannels, kernel_size=final_kernal, stride=1, padding=final_kernal // 2, bias=True)
                    )
                    
        self.gf2_branch = Gflv2branch(10) # (4+1)*2 = 10


    def integral(self, x):
        n,c,h,w = x.shape
        regmax = int(c//2)
        weight = torch.tensor(range(regmax)).float().cuda() # 要不要固定下来
        x = x.reshape(n,2,regmax,h,w)
        x = x.permute(0,1,3,4,2)
        x = F.softmax(x, dim=4) 
        x = F.linear(x, weight)* 127 / (regmax - 1)
        return x
        
    def loss(self,
            preds,
            gt_meta
            ):
        whloss1 = self.loss_dfl(preds['wh'], gt_meta['wh']) 
        wh = self.integral(preds['wh']) # 这里复制一份, val时的decode里面要（重复了）再积分一次
        whloss2 = self.loss_wh(wh, gt_meta['wh'])
        whloss = whloss1 +whloss2
        regloss = self.loss_reg(preds['reg'], gt_meta['reg'])

        hmloss = self.loss_hm(preds['hm'], gt_meta['hm'])
        loss = hmloss + whloss + regloss
        loss_states = dict(
                loss=loss,
                loss_hm=hmloss,
                loss_dfl=whloss1,
                loss_wh=whloss2,
                loss_reg=regloss
            )
        return loss, loss_states


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


    def decode(self, preds, meta, K=100):
        """
        解析输出特征图为bbox
        """
        # hm = preds['hm'].sigmoid() # 不用gflv2
        hm = preds['hm']
        wh = preds['wh']
        reg = preds['reg']
        n,c,h,w = hm.shape
        # hm = _nms(hm)
        hm = multi_nms(hm, 24) # 多重滤波
       
        wh = self.integral(wh)
        scores, classes, xs, ys = _topk(hm, K=K)
        wh = wh[:,:,ys,xs].squeeze(2) 
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
                                ys + wh[:,1,:]/2], dim=2)  #  默认就是4倍的比例
        dets = torch.cat([bboxes, scores], dim=2)
        # if dets.shape[0] > 1:
        #     print('warning: val batch_size > 1, please check up! ')
        return dets[0],classes[0]
        

    def forward(self, x):
        mixout = self.mixhead(x) # only the lastest layer
        hm = mixout[:,0:self.out_classes[0],:,:]
        wh = mixout[:,self.out_classes[0]:(self.out_classes[0]+self.out_classes[1]),:,:]
        reg = mixout[:,(self.out_classes[0]+self.out_classes[1]):,:,:]

        # hm = self.hm(x)
        # wh = self.wh(x)
        # reg = self.reg(x) 

        n,c,h,w = wh.shape
        whtemp = F.softmax(wh.reshape(n,2,c//2,h,w), dim=2) #  softmax后的
        whtopk, _ = torch.topk(whtemp, k=4, dim=2)
        whtopk_mean = torch.cat([whtopk, whtopk.mean(dim=2, keepdim=True)], dim=2) # n 2 5 h w
        stat = whtopk_mean.reshape(n,-1,h,w) # n 10 h w
        out_branch = self.gf2_branch(stat)
        hm = torch.clamp(hm.sigmoid(),min=1e-4,max=1-1e-4) * out_branch

        
        # hm = hm.sigmoid()
        
        # hm = multi_nms(hm, 8) # 多重滤波
        #wh = self.integral(wh) # 转onnx时把积分提前

        return {
            'hm': hm,
            'wh': wh,
            'reg': reg
        }

    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            # cv2.imshow('det', result)
            cv2.imwrite('result/tempout.jpg',result)
            # cv2.imwrite('/media/bihuasky/Files2/Dataset/voc/VOCdevkit/VOC2012/mtestimgs_result/tempout_{0}.jpg'.format(np.random.randint(0,1000000)),result)
           