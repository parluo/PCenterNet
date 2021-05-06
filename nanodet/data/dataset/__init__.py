import copy
from .coco import CocoDataset, centernet_CocoDataset
from .xml_dataset import XMLDataset
from .voc import VOC


def build_dataset(cfg, mode, rootpath=''):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop('name')
    if name == 'coco':
        return CocoDataset(mode=mode, **dataset_cfg)
    if name == 'xml_dataset':
        return XMLDataset(mode=mode, **dataset_cfg)
    if name == 'voc':
        return VOC(mode=mode, **dataset_cfg)
    if name == 'centernet_coco':
        return centernet_CocoDataset(mode=mode,**dataset_cfg)
    else:
        raise NotImplementedError('Unknown dataset type!')
