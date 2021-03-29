# import torch
from tqdm import tqdm

import torch
from tool.torch_utils import do_detect

from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from dataset_coco import COCOImage

import json

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

coco_category = coco80_to_coco91_class()

def coco_format(result):
    coco_pred = []
    img_id, boxes, H, W = result
    for box in boxes:
        x1, y1, x2, y2, conf, conf, category_id = box
        
        x1 = x1 * W
        y1 = y1 * H
        
        x2 = x2 * W
        y2 = y2 * H
        # x1, y1, x2, y2 = map(int, (x1, y1, x2 ,y2))
        width = int(x2 - x1)
        height = int(y2 - y1)
        x, y = int(x1), int(y1)
        
        pred_out = {
            "image_id": int(img_id),
            "category_id": int(coco_category[category_id]),
#             "bbox": [x,y,width,height, int(x1), int(x2), int(y1), int(y2)],
            "bbox": [x,y,width,height],
            "score": float(conf),            
        }
        
        coco_pred.append(pred_out)
    return coco_pred

@torch.no_grad()
def evaluate(model, device, opt=None):

    if opt==None:
        class opt:
            anno_json = '/workspace/GitHub/YOLO/coco_forTest/annotations/instances_val2017_64.json'
            pred_json = './YOLOv4_training_pred.json'
            img_path = '/workspace/GitHub/YOLO/coco_forTest/images/val2017_64/'
            img_size = 416
            batch_size = 4
            conf_thresh = 0.001
            nms_thresh = 0.6
            use_cuda = 1

    model.to(device)

    # Data Loader
    anno = COCO(opt.anno_json)
    val_set = COCOImage(opt.anno_json, opt.img_path, opt.img_size)
    val_loader = DataLoader(val_set, opt.batch_size, shuffle=True, num_workers=0)

    # Accumulate results
    result_dict = dict([])
    print('STRAT detection')
    for imgs, img_ids, sizes in tqdm(val_loader):
        # model
        boxes = do_detect(model, imgs, conf_thresh=opt.conf_thresh, nms_thresh=opt.nms_thresh, use_cuda=opt.use_cuda, verbose=False)
        # process
        
        for img_id, box, H, W in zip(img_ids.numpy(), boxes, sizes[0].numpy(), sizes[1].numpy()):
            result_dict[img_id] = (img_id, box, H, W)

    # Transform results to COCO format
    print('Convert results to COCO format')
    total = []
    for img_id in tqdm(result_dict.keys()):
        one_result = coco_format(result_dict[img_id])
        total.extend(one_result)

    with open(opt.pred_json, 'w')as f:
        json.dump(total, f)

    # COCO Evaluation
    from pycocotools.cocoeval import COCOeval
    pred = anno.loadRes(opt.pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return eval