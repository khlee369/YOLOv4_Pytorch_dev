from darknet_parser import Darknet
from tool.torch_utils import do_detect

from coco_evaluator import evaluate

import torch
from tqdm import tqdm

cfgfile = '/workspace/GitHub/YOLO/cfg/yolov4.cfg'
weights = '/workspace/GitHub/YOLO/weights/yolov4.weights'

# evaluate parameter
class opt:
    anno_json = '/workspace/GitHub/YOLO/coco_forTest/annotations/instances_val2017_64.json'
    pred_json = './YOLOv4_pred_parserTest.json'
    img_path = '/workspace/GitHub/YOLO/coco_forTest/images/val2017_64/'
    img_size = 416
    batch_size = 4
    conf_thresh = 0.001
    nms_thresh = 0.6
    use_cuda = 1

def get_args():
    import argparse
    parser = argparse.ArgumentParser('Darknet yolov4.cfg, yolov4.weights parsing test')
    parser.add_argument('-cfgfile', type=str, default=cfgfile,
                        help='/paht/to/yolov4.cfg', dest='cfgfile')
    parser.add_argument('-weights', type=str,
                        default=weights,
                        help='/path/to/yolov4.weights', dest='weights')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    m = Darknet(args.cfgfile)
    m.print_network()
    m.load_weights(args.weights)
    torch.cuda.set_device(torch.device('cuda:1'))
    evaluate(m, torch.cuda.current_device(), opt)