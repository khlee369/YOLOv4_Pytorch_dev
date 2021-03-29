from darknet_parser import Darknet
from models import Yolov4
from tool.torch_utils import do_detect

# from coco_evaluator import evaluatoe

import torch
from tqdm import tqdm
import time

cfgfile = '/workspace/GitHub/YOLO/cfg/yolov4.cfg'
weights = '/workspace/GitHub/YOLO/weights/yolov4.weights'

print("Loading Model")
# m = Darknet(cfgfile)
# m.load_weights(weights)
# m.print_network()

model = Yolov4(n_classes=80, inference=True)
model.eval()
model.head.inference=True
m = model

print("Trasfer model to CUDA device....")
use_cuda = 1
if use_cuda:
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    m.cuda()

print("Do Detection")
a = torch.randn(1, 3, 416, 416)

print("Warm Freezing Start...")

# m.half()
# a = a.half()

for i in range(10):
    boxes = do_detect(m, a, conf_thresh=0.4, nms_thresh=0.6, use_cuda=1, verbose=False)

print("Start Measuring....")
print("Measure 10 times")
do_detect(m, a, conf_thresh=0.4, nms_thresh=0.6, use_cuda=1, verbose=True)

for _ in range(10):
    stime = time.time()
    for i in range(20):
        # boxes = do_detect(m, a, conf_thresh=0.4, nms_thresh=0.6, use_cuda=1, verbose=False)
        do_detect(m, a, conf_thresh=0.4, nms_thresh=0.6, use_cuda=1, verbose=False)

    # total = time.time() - stime
    # print("FPS : {:.2f}".format(20/total))

    print("FPS : {:.2f}".format(20 / (time.time() - stime)))
