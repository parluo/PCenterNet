
import cv2
import os
import time
import torch
import argparse
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from warp import warp_and_resize
from color import color_aug_and_norm

meta = {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', default='image', help='demo type, eg. image, video and webcam')
    parser.add_argument('--config', default='config/centernet_coco_dataset.yml', help='model config file path')
    parser.add_argument('--model', default='model/exp/voc_hwdfl8_focal_gflv2/model_best/model_best.pth', help='model file path')
    parser.add_argument('--path', default='testimg', help='path to images or video')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    args = parser.parse_args()
    return args

args = parse_args()
load_config(cfg, args.config) # 加载yaml文件
img = cv2.imread('nanodet/data/transform/test.jpg')
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
meta['img'] = img

input_shape = img.shape
dst_shape = (input_shape[1],input_shape[0])  # w h

### warp

# meta = warp_and_resize(meta, cfg.data.train.pipeline, dst_shape)
# img = meta['img']
# cv2.imshow('d',img) 
# cv2.waitKey(0)
# print(meta['warp_matrix'])


# color
meta = color_aug_and_norm(meta, cfg.data.train.pipeline)
img = meta['img']
cv2.imshow('d',img) 
cv2.waitKey(0)

 