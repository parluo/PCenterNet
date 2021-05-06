import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from .base import BaseDataset


class CocoDataset(BaseDataset):


    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url': 'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        # 针对crowdhuman数据集 pop 2 mask类别
        # self.coco_api.cats.pop(2)
        # self.cat_ids = [1] # sorted(self.coco_api.getCatIds())

        # 别的数据集
        self.cat_ids = sorted(self.coco_api.getCatIds())

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        id = img_info['id']
        if not isinstance(id, int):
            raise TypeError('Image id must be int.')
        info = {'file_name': file_name,
                'height': height,
                'width': width,
                'id': id}
        return info

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict('bboxes':ndarray(x0,y0,x1,y1), 'labels':ndarray, 'bboxes_ignore': ,)
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            if ann.get('ignore', False): # 这里必须了解到： crowdhuman数据集有person，mask两个类别， mask类别到ignore=1,在这里会自动过滤掉，但在self.cats上有两个类别
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            # if ann['category_id'] == 2:
            #     a = 1

            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann['keypoints'])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        if self.use_instance_mask:
            annotation['masks'] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation['keypoints'] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation['keypoints'] = np.zeros((0, 51), dtype=np.float32)
        return annotation

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print('image {} read failed.'.format(image_path))
            raise FileNotFoundError('Cant load image! Please check image path!')
        ann = self.get_img_annotation(idx)
        # if 1 in ann['labels']:
        #     a = 1
        meta = dict(img=img,
                    img_info=img_info,
                    gt_bboxes=ann['bboxes'],
                    gt_labels=ann['labels'])
        if self.use_instance_mask:
            meta['gt_masks'] = ann['masks']
        if self.use_keypoint:
            meta['gt_keypoints'] = ann['keypoints']

        meta = self.pipeline(meta, self.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))
        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)



class centernet_CocoDataset(CocoDataset):
    ##  自定义的用于训练centernet的coco数据集

    ## cocoapi会自动处理类别的，不用自己管，id from 1 to x
    def __init__(self, class_names, output_size, **kwargs):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.output_w, self.output_h = output_size
        super(centernet_CocoDataset,self).__init__(**kwargs)
        

    def get_train_data(self, idx):
            """
                重写训练标签给centernet就行
                复用此框架下的pipeline预处理图像
                然后再生成hm,wh,reg
            """
            meta = CocoDataset.get_train_data(self,idx)

            num_classes = self.num_classes
            output_h = self.output_h
            output_w = self.output_w
            hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
            wh = np.zeros((2, output_h,output_w), dtype=np.float32) # 0-h, 1-w
            reg = np.zeros((2, output_h, output_w), dtype=np.float32)

            bboxes, labels = meta['gt_bboxes'], meta['gt_labels']
            scale_h, scale_w = self.input_size[0]/output_h, self.input_size[1]/output_w
            for bbox, cls_id in zip(bboxes, labels):
                bbox[[0,2]] = (bbox[[0,2]] / scale_h).clip(0, output_w-1)
                bbox[[1,3]] = (bbox[[1,3]] / scale_w).clip(0, output_h-1)
                boxh, boxw = bbox[3]-bbox[1], bbox[2]-bbox[0] # 128*128中的box
                if boxh > 0 and boxw > 0:
                    cx, cy = (bbox[0] +bbox[2])/2, (bbox[1] + bbox[3])/2 # float
                    box =[cx, cy, boxw, boxh] # 转换成【中心x,中心y, w, h】格式
                    self.drawhm(hm[cls_id], box)
                    self.drawhw(wh, box)
                    self.drawreg(reg, box)
            

            meta.pop('gt_bboxes')
            meta.pop('gt_labels')
            meta['hm'] = hm
            meta['wh'] = wh
            meta['reg'] = reg

            return meta