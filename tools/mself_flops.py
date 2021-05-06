import torch
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, get_model_complexity_info
from torchsummary import summary 

def main(config, input_shape=(3, 320, 320)):
    model = build_model(config.model)
    
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM"%(total/1e6))
if __name__ == '__main__':
    cfg_path = r"config/nano_centernet_fpn.yml"
    load_config(cfg, cfg_path)
    main(config=cfg,
         input_shape=(3, 512, 512)
         )
