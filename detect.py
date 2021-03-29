from tool.utils import *
from tool.torch_utils import *
from darknet_parser import Darknet
from models import Yolov4

import cv2
import numpy as np
import argparse

"""hyper parameters"""
use_cuda = True
namesfile = 'data/coco.names'
class_names = load_class_names(namesfile)

@torch.no_grad()
def detect_img(model, img_path, savename='prediction.jpg', conf_thresh=0.1, nms_thresh=0.6, img_size=416):
    img = cv2.imread(img_path)
    sized = cv2.resize(img, (img_size, img_size))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    boxes = do_detect(model, sized, conf_thresh=conf_thresh, nms_thresh=nms_thresh, use_cuda=use_cuda, verbose=False)
    out_img = plot_boxes_cv2(img, boxes[0], savename=savename, class_names=class_names)

    return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

@torch.no_grad()
def detect_mp4(model, mp4_path, conf_thresh=0.1, nms_thresh=0.6, img_size=416):
    cam = cv2.VideoCapture(mp4_path)

    frames = 0
    start = time.time()
    while True:
        # Get webcam input
        ret_val, img = cam.read()

    #     # Mirror 
        # img = cv2.flip(img, 1)

    #     # Free-up unneeded cuda memory
    #     torch.cuda.empty_cache()

        # Detection
        # sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.resize(img, (img_size, img_size))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        boxes = do_detect(m, sized, conf_thresh=conf_thresh, nms_thresh=nms_thresh, use_cuda=use_cuda, verbose=False)
        out_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        # Show FPS
        frames += 1
        intv = time.time() - start
        if intv > 1:
            print("FPS of the video is {:5.2f}".format( frames / intv ))
            print(type(img), img.shape)
            start = time.time()
            frames = 0
        
        # Show webcam
        cv2.imshow('Demo webcam', out_img)
        if cv2.waitKey(0) == 27: 
            break  # Press Esc key to quit
    cam.release()
    cv2.destroyAllWindows()

def detect_webcam(model):
    pass

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='/workspace/GitHub/YOLO/cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='/workspace/GitHub/YOLO/weights/yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-nclasses', type=int, default=80,
                        help='number of classes', dest='nclasses')
    parser.add_argument('-source', type=str,
                        default='/workspace/GitHub/YOLO/sample_data/road2.mp4',
                        help='path of your image file. or video or 0 to webcam', dest='source')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    # model selection, darknet or pytorch
    if args.weightfile.split('.')[-1] == 'weights': # darknet
        m = Darknet(args.cfgfile)
        m.load_weights(args.weightfile)
    elif args.weightfile.split('.')[-1] in ['pth', 'pt', 'ph']: # pytorch
        m = Yolov4(n_classes=args.nclasses, inference=True)
        state_dict = torch.load(args.weightfile)
        m.load_state_dict(state_dict)
        m.head.inference=True
    m.eval()

    if use_cuda:
        torch.cuda.set_device(torch.device('cuda:1'))
        m.cuda()

    if args.source == '0': # webcam
        pass
    else:
        form = args.source.split('.')[-1]
        if form == 'jpg': # image
            pass
        elif form == 'mp4': # mp4
            detect_mp4(m, args.source)
            
