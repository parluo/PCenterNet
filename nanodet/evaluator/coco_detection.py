import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


class CocoDetectionEvaluator:
    def __init__(self, dataset):
        assert hasattr(dataset, 'coco_api')
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ['mAP', 'AP_50', 'AP_75', 'AP_small', 'AP_m', 'AP_l']

    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score)
                    json_results.append(detection)
        return json_results

    # def evaluate(self, results, save_dir, epoch, logger, rank=-1):
    #     if results is not None:
    #         results_json = self.results2json(results)
    #         json_path = os.path.join(save_dir, 'results{}.json'.format(rank))
    #         json.dump(results_json, open(json_path, 'w'))
    #     else:
    #         json_path = os.path.join(save_dir, 'results{}.json'.format(rank))
            
    #     coco_dets = self.coco_api.loadRes(json_path)
    #     coco_eval = COCOeval(copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox")
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()
    #     aps = coco_eval.stats[:6]
    #     eval_results = {}
    #     for k, v in zip(self.metric_names, aps):
    #         eval_results[k] = v
    #         logger.scalar_summary('Val_coco_bbox/' + k, 'val', v, epoch)
    #     return eval_results

    def evaluate(self, results, save_dir, epoch, logger, rank=-1):
        """
        添加PR曲线的绘制: 对所有类别，对iou=0.5:0.05:0.95
        """
        if results is not None:
            results_json = self.results2json(results)
            json_path = os.path.join(save_dir, 'results{}.json'.format(rank))
            json.dump(results_json, open(json_path, 'w'))
        else:
            json_path = os.path.join(save_dir, 'results{}.json'.format(rank))
            
        coco_dets = self.coco_api.loadRes(json_path)
        coco_eval = COCOeval(copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox")
       
        # coco_eval.params.catIds = [1] # 针对crowdhuman
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        precision = coco_eval.eval['precision']
        """
        precision: [T,R,K,A,M]
        T: 0.5-0.95 thresholds, idx from 0 to 9
        R: recall thresholds [0:0.01:1], idx from 0 to 100
        K: category
        A: area range,(all,small,medium,large),idx from 0 to 3
        M: max dets, (1,10,100), idx from 0 to 2
        """
        precision = precision.mean(axis=0,keepdims=True)
        precision = precision.mean(axis=2,keepdims=True)
        pr = precision[0,:,0,0,2]
        x = np.arange(0.0, 1.01, 0.01)

        res = {
            'x':x,
            'pr':pr
        }
        np.save("result/shuf_dlaup_mixhead.npy",res)
        plt.plot(x, pr) #, label="iou=0.5:0.05:0.95")
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0,1.0)
        plt.ylim(0,1.01)
        plt.grid(True)
        plt.legend(loc="lower left")
        plt.savefig('result/pr_shuf_dlaup_mix.jpg')

        aps = coco_eval.stats[:6]
        eval_results = {}
        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
            logger.scalar_summary('Val_coco_bbox/' + k, 'val', v, epoch)
        return eval_results
