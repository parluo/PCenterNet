import os
import torch
import numpy as np
import cv2
import time
import logging
from collections import defaultdict
import xml.etree.ElementTree as ET
import torchvision.datasets as datasets
from pycocotools.coco import COCO
from .coco import CocoDataset
from tqdm import tqdm
from ...util.centernet_util import get_affine_transform,affine_transform,gaussian_radius

class COCO_XML(COCO):

    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
        self.dataset = dataset
        self.createIndex()

class VOC(CocoDataset):
    def __init__(self, 
                rootpath, 
                class_names, 
                output_size, 
                # random_crop, # 1
                # random_flip, # 2
                # normalize,   # 3
                **kwargs):
        """
        提供centernet模型使用xml类数据集的接口

        rootpath: voc2012数据集的存放目录
        
        """
        self.rootpath = rootpath
        self.class_names = class_names 
        self.num_classes = len(class_names)
        self.output_w, self.output_h = output_size
        # self.random_crop = True   # 1
        # self.random_flip = True   # 2
        # self.mean , self.std = [0,0,0],[1,1,1] # 3 
        super(VOC, self).__init__(**kwargs)

    def VOC_to_coco(self):
        logging.info('loading annotations into memory...')
        tic = time.time()
        if self.mode not in ['train', 'val']:
            raise NotImplementedError
        
        datasplit_file = os.path.join(self.rootpath, '{}.txt'.format(self.mode))
        with open(datasplit_file,'r') as f:
            ann_file_names = f.readlines()
        ann_file_names = [os.path.join(self.ann_path, '{}.xml'.format(filename.strip()) ) for filename in ann_file_names]

        logging.info("Found {} annotation files.".format(len(ann_file_names))) # 仅改动这一步
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append({'supercategory': supercat,
                               'id': idx + 1,
                               'name': supercat})
        ann_id = 1
        for idx, xml_name in tqdm(enumerate(ann_file_names),total=len(ann_file_names)):
            tree = ET.parse(xml_name) # 改这里
            root = tree.getroot()
            file_name = root.find('filename').text
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            info = {'file_name': file_name,
                    'height': height,
                    'width': width,
                    'id': idx + 1}
            image_info.append(info)
            for _object in root.findall('object'):
                category = _object.find('name').text
                if category not in self.class_names:
                    logging.warning("WARNING! {} is not in class_names! Pass this box annotation.".format(category))
                    continue
                for cat in categories:
                    if category == cat['name']:
                        cat_id = cat['id']
                xmin = int(_object.find('bndbox').find('xmin').text)
                ymin = int(_object.find('bndbox').find('ymin').text)
                xmax = int(_object.find('bndbox').find('xmax').text)
                ymax = int(_object.find('bndbox').find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin
                if w < 0 or h < 0:
                    logging.warning("WARNING! Find error data in file {}! Box w and h should > 0. Pass this box "
                                    "annotation.".format(xml_name))
                    continue
                coco_box = [max(xmin, 0), max(ymin, 0), min(w, width), min(h, height)]
                ann = {'image_id': idx + 1,
                       'bbox': coco_box,
                       'category_id': cat_id,
                       'iscrowd': 0,
                       'id': ann_id,
                       'area': coco_box[2] * coco_box[3]
                       }
                annotations.append(ann)
                ann_id += 1

        coco_dict = {'images': image_info,
                     'categories': categories,
                     'annotations': annotations}
        logging.info('Load {} xml files and {} boxes'.format(len(image_info), len(annotations)))
        logging.info('Done (t={:0.2f}s)'.format(time.time() - tic))
        return coco_dict

    def get_train_data(self, idx):
        """
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

    # def get_train_data1(self, idx):
    #     """
    #     原centernet中的写法
    #     centernet的标签不一样，所以要重写
    #     """
    #     img_info = self.get_per_img_info(idx)
    #     file_name = img_info['file_name']
    #     image_path = os.path.join(self.img_path, file_name)
    #     img = cv2.imread(image_path)
    #     if img is None:
    #         print('image {} read failed.'.format(image_path))
    #         raise FileNotFoundError('Cant load image! Please check image path!')
        
        
        
    #     ann = self.get_img_annotation(idx)
        
    #     h,w = img.shape[:2]
    #     c = np.array([w/2., h/2.], dtype=np.float32) # [w/2, h/2]
    
    #     s = max(h, w) * 1.0 # 最大的尺寸
    #     input_h, input_w = self.input_size

    #     flipped = False
    #     if self.mode == 'train':
    #         if self.random_crop:
    #             s = s * 1 #np.random.choice(np.arange(0.8, 1.2, 0.1))
    #             w_border = self._get_border(128, w)
    #             h_border = self._get_border(128, h)
    #             c[0] = np.random.randint(low=w_border, high=w-w_border)
    #             c[1] = np.random.randint(low=h_border, high=h-h_border)

    #         if np.random.random() < self.random_flip:
    #             flipped = True
    #             img = cv2.flip(img,0)
    #             # img  = img[:,::-1, :]
    #             c[0] = w - c[0] - 1

    #     trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
    #     inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    #     inp = (inp.astype(np.float32)/255.)
        
    #     inp = (inp - self.mean) / self.std
    #     inp = inp.transpose(2,0,1) # 这是调整维度，不是rgb


    #     output_h = self.output_h
    #     output_w = self.output_w

    #     num_classes = self.num_classes
    #     trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    #     hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    #     wh = np.zeros((2, output_h,output_w), dtype=np.float32) # 0-h, 1-w
    #     reg = np.zeros((2, output_h, output_w), dtype=np.float32)

    #     for (bbox, cls_id) in zip(ann['bboxes'], ann['labels']):
    #         if flipped:
    #             bbox[[0,2]] = w - bbox[[2,0]] - 1 # 水平翻转
    #         # 图像变了, gt框也要变
    #         bbox[:2] = affine_transform(bbox[:2], trans_output)
    #         bbox[2:] = affine_transform(bbox[2:], trans_output)
    #         bbox[[0,2]] = np.clip(bbox[[0,2]], 0, output_w - 1) # 限制范围
    #         bbox[[1,3]] = np.clip(bbox[[1,3]], 0, output_h - 1)


    #         boxh, boxw = bbox[3]-bbox[1], bbox[2]-bbox[0] # 128*128中的box
    #         cx, cy = (bbox[0] +bbox[2])/2, (bbox[1] + bbox[3])/2
    #         if boxh > 0 and boxw > 0:
    #             box =[cx,cy, boxw, boxh]
    #             self.drawhm(hm[cls_id], box)
    #             self.drawhw(wh, box)
    #             self.drawreg(reg, box)
    #     inp = torch.tensor(inp, dtype=torch.float32)
    #     # hm = torch.from_numpy(hm).cuda()
    #     # reg = torch.from_numpy(reg).cuda()
    #     # wh = torch.from_numpy(wh).cuda()
    #     ret = dict(
    #             img=inp, 
    #             hm=hm,
    #             reg=reg,
    #             wh=wh,
    #             img_info=img_info,
    #         )
    #     return ret


        
        
        # hm = np.zeros((self.num_classes, self.output_h, self.output_w), dtype=np.float32)
        # wh = np.zeros((2, self.output_h,self.output_w), dtype=np.float32)
        # reg = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
        # for (bbox, id) in zip(ann['bboxes'], ann['labels']):
        #     box =[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        #     self.drawhm(hm[id], box)
        #     self.drawhw(wh, box)
        #     self.drawreg(reg, box)
        
        # meta = dict(img=img,
        #             img_info=img_info,
        #             hm = hm,
        #             reg = reg,
        #             wh = wh
        #             )
        # meta = self.pipeline(meta, self.input_size)    #     
        # meta['img'] = torch.from_numpy(meta['img'].transpose(2,0,1))
        # return meta

    def get_data_info(self,ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.VOC_to_coco() # 仅改动这里
        self.coco_api = COCO_XML(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info


#   # # 其实也可以不重写，不过得注意self.img_path的地址得给到JPEGImages
#     # 源图像train和val都是混在一起的，所以val和train的self.img_path地址一样
#     def get_train_data(self, idx):
#         img_info = self.get_per_img_info(idx)
#         img, ann = self.vocdata[idx] 
#         img = np.array(img)  # 此时pil图像转成nparray后是rgb的
#         if img is None:
#             print('image {} read failed.'.format(image_path))
#             raise FileNotFoundError('Cant load image! Please check image path!')
#         ann = self.get_img_annotation(idx)
#         meta = dict(img=img,
#                     img_info=img_info,
#                     gt_bboxes=ann['bboxes'],
#                     gt_labels=ann['labels'])
#         if self.use_instance_mask:
#             meta['gt_masks'] = ann['masks']
#         if self.use_keypoint:
#             meta['gt_keypoints'] = ann['keypoints']

#         meta = self.pipeline(meta, self.input_size)
#         meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))
#         return meta
