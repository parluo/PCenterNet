import copy
from .fpn import FPN
from .pan import PAN
from .ffn import FFN
from .cfpn import CFPN


def build_fpn(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = fpn_cfg.pop('name')
    if name == 'FPN':
        return FPN(**fpn_cfg)
    elif name == 'PAN':
        return PAN(**fpn_cfg)
    elif name == 'FFN': 
        return FFN(**fpn_cfg)
    elif name == 'CFPN':
        return CFPN(**fpn_cfg)
    else:
        raise NotImplementedError