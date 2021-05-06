from abc import ABCMeta, abstractmethod
import torch
import numpy as np
from torch.utils.data import Dataset
from ..transform import Pipeline
import math


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    A base class of detection dataset. Referring from MMDetection.
    A dataset should have images, annotations and preprocessing pipelines
    NanoDet use [xmin, ymin, xmax, ymax] format for box and
     [[x0,y0], [x1,y1] ... [xn,yn]] format for key points.
    instance masks should decode into binary masks for each instance like
    {
        'bbox': [xmin,ymin,xmax,ymax],
        'mask': mask
     }
    segmentation mask should decode into binary masks for each class.

    :param img_path: image data folder
    :param ann_path: annotation file path or folder
    :param use_instance_mask: load instance segmentation data
    :param use_seg_mask: load semantic segmentation data
    :param use_keypoint: load pose keypoint data
    :param load_mosaic: using mosaic data augmentation from yolov4
    :param mode: train or val or test
    """
    def __init__(self,
                 img_path,
                 ann_path,
                 input_size,
                 pipeline,
                 keep_ratio=True,
                 use_instance_mask=False,
                 use_seg_mask=False,
                 use_keypoint=False,
                 load_mosaic=False,
                 mode='train' # 若不定义就默认train,但如单独提出定义，就变了
                 ):
        self.img_path = img_path
        self.ann_path = ann_path
        self.input_size = input_size
        self.pipeline = Pipeline(pipeline, keep_ratio)
        self.keep_ratio = keep_ratio
        self.use_instance_mask = use_instance_mask
        self.use_seg_mask = use_seg_mask
        self.use_keypoint = use_keypoint
        self.load_mosaic = load_mosaic
        self.mode = mode

        self.data_info = self.get_data_info(ann_path)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if self.mode == 'val' or self.mode == 'test':
            return self.get_val_data(idx)
        else:
            while True:
                data = self.get_train_data(idx)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    @abstractmethod
    def get_data_info(self, ann_path):
        pass

    @abstractmethod
    def get_train_data(self, idx):
        pass

    @abstractmethod
    def get_val_data(self, idx):
        pass

  #   *******************************************************

    def get_another_id(self):
        return np.random.random_integers(0, len(self.data_info)-1)

   
    def _l1kernel(self, shape):
        """
        l1分布热度图
        """
        H,W = shape 
        cx = W//2
        cy = H//2
        Z = np.zeros((H,W), dtype=np.float32)
      
        for i in range(W):
            for j in range(H):
                val = 1 - 2 * (abs(i-cx)/W + abs(j-cy)/H)
                Z[j,i] = max(val, 0)

# ### 产生更小的r
#         hw = int(W/6)
#         hh = int(H/6)
#         l = cx-hw
#         r = cx+hw
#         t = cy-hh
#         b = cy+hh
#         for i in range(l,r):
#             for j in range(t,b):
#                 val = 1 - 2 * (abs(i-cx)/(2*hw) + abs(j-cy)/(2*hh))
#                 Z[j,i] = max(val, 0)



        return Z

    def _gaussain2d(self, shape, sigma1, sigma2=None,k=1):
        """
        高斯分布热度图
        """
        if sigma2 is None:
            sigma2 = sigma1
        cx,cy = [(i-1)/2 for i in shape] # 这里h->x  w->y 
        h,w = shape
        out_g = np.zeros(shape, dtype=np.float32)
        for i in range(h): # i->h
            for j in range(w): # i->w
                out_g[i][j] = np.exp(-((i-cx)**2/(2*sigma2**2) + (j-cy)**2/(2*sigma1**2)))
        
        return out_g * k
    def gaussian_radius(self, det_size, min_overlap=0.7):
        """
        centernet中原始的算半径的函数， 额
        """
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def drawhm(self, heatmap, box):
        
   
        x0, y0, w, h = box
        x0, y0 = int(x0), int(y0)
# ################ 使用1/2宽高l1 kernel  【1】
        r1, r2 = int(w//2), int(h//2)  # 1->w  2->h

# ################# 使用h*w/(h+w) 【2】
#         maxs = max(1, w*h/((w+h)*2))
#         if w>h:
#             r1 = int(maxs)
#             r2 = int(r1 * h/w)
#         else:
#             r2 = int(maxs)
#             r1 = int(r2 * w/h)


# ############### 使用CenterNet原来地半径 【3】
#         radius = self.gaussian_radius((h,w))
#         r1 = max(int(radius),1)
#         r2 = max(int(radius),1)
 


        d1, d2 = 2*r1 + 1, 2*r2 + 1

   

        # d1 d2 为1，也就是centernet的含义
        # gaussain_kernel = self._gaussain2d((d2,d1), sigma1=d1/6, sigma2=d2/6)
        gaussain_kernel = self._l1kernel((d2,d1))


        h,w = heatmap.shape[:2]


        l,r = min(x0, r1), min(w-x0, d1-r1)
        t,b = min(y0, r2), min(h-y0, d2-r2)
        
        masked_heatmap = heatmap[ y0-t : y0+b , x0-l: x0+r ]
        masked_gaussian = gaussain_kernel[ r2-t : r2+b, r1-l : r1+r]
        if min(masked_gaussian.shape)>0 and min(masked_heatmap.shape)>0:
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
        return heatmap
    def drawhw(self, wh, box):
        """
        wh[0,:,:]放h
        wh[1,:,:]放w


        2021.3.3 : 重新调整成wh，对其原预训练模型
        """
        x0, y0, w, h = box
        x0, y0 = int(x0), int(y0)
        wh[0, y0, x0] = w
        wh[1, y0, x0] = h # 注意y x顺序

    def drawreg(self, reg, box):
        """
        reg[0,:,:]放h
        reg[1,:,:]放w


        2021.3.3 : 重新调整成wh，对其原预训练模型
        """
        x0, y0, w, h = box
        x0_int, y0_int = int(x0), int(y0)
        reg[0, y0_int, x0_int] = x0-x0_int # 这里的顺序不能错！
        reg[1, y0_int, x0_int] = y0-y0_int
        
        

        #  # 
        # x0,y0,w,h = box
        # x1 = w + x0
        # y1 = h + y0
        # cx = int(x0 + w / 2)
        # cy = int(y0 + h / 2)
        # l, r = int(x0 + 1), int(x1 + 1)
        # t, b = int(y0 + 1), int(y1 + 1)
        # for i in range(l, r):
        #     for j in range(t, b):
        #         reg[:,j,i] = cy-j, cx-i
        # return reg

    
    
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return  border // i


