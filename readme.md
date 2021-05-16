### PCenterNet
---
This is a project for study on CenterNet with probability prediction.  
The initial code struction comes from [nanodet](github.com/RangiLyu/nanodet) which organized by [MMDetection](github.com/open-mmlab/mmdetection). 
The [dlaseg.py](github.com/parluo/PCenterNet/blob/master/nanodet/model/arch/dlaseg.py) is the mainly network file, and you only need to change the arch.name in `config/centernet_*_dataset.xml` during training.  
Aternatively, you could choose the `config/nano_centernet*.yml` to design your network with different backbone, FPN and head modules.

### Features
---
- Simplify and rewrite the original centernet codes, including object boxes decoding and labels generation;
- Organization the detection pipeline with MMDetection;
- A new heatmap generation method which differ from gaussain heatmap;
- Study on the combination of probability prediction and keypoint-based detection algorithm;
- Comparsion of kinds of data augement methods;


