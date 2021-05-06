from .gfl import GFL
from .dlaseg import DLASeg


def build_model(model_cfg):
    if model_cfg.arch.name == 'GFL':
        model = GFL(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)
    elif model_cfg.arch.name == 'DLASeg':
        num_layers = model_cfg.arch.num_layers
        out_classes = model_cfg.arch.out_classes
        heads = {
            'hm':out_classes[0],
            'wh':out_classes[1],
            'reg':out_classes[2]
        }
        head_conv = model_cfg.arch.head_conv

        model = DLASeg('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=4,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
    else:
        raise NotImplementedError
    return model
