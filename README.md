model cfg, weight parser from [Tianxiaomo / pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)

evaluation에 사용되는 coco dataset은 test_parsing.py 내에 ```class: opt```로 하드코딩 되어있음

default는 coco val2017 이미지 64개
```py
class: opt:
  anno_json = '/workspace/GitHub/YOLO/coco_forTest/annotations/instances_val2017_64.json'
  pred_json = './YOLOv4_pred_parserTest.json'
  img_path = '/workspace/GitHub/YOLO/coco_forTest/images/val2017_64/'
```
결과는 Model_Parsing_Eval.ipynb 참조

example output (yolov4)
```
$ python test_parsing.py \
  -cfgfile '/workspace/GitHub/YOLO/cfg/yolov4.cfg' \
  -weights '/workspace/GitHub/YOLO/weights/yolov4.weights'
```
```
convalution havn't activate linear
convalution havn't activate linear
convalution havn't activate linear
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   608 x 608 x   3   ->   608 x 608 x  32
    1 conv     64  3 x 3 / 2   608 x 608 x  32   ->   304 x 304 x  64
    2 conv     64  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  64
    3 route  1
    4 conv     64  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  64
    5 conv     32  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  32
    6 conv     64  3 x 3 / 1   304 x 304 x  32   ->   304 x 304 x  64
    7 shortcut 4
    8 conv     64  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  64
    9 route  8 2
   10 conv     64  1 x 1 / 1   304 x 304 x 128   ->   304 x 304 x  64
   11 conv    128  3 x 3 / 2   304 x 304 x  64   ->   152 x 152 x 128
   12 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64
   13 route  11
   14 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64
   15 conv     64  1 x 1 / 1   152 x 152 x  64   ->   152 x 152 x  64
   16 conv     64  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x  64
   17 shortcut 14
   18 conv     64  1 x 1 / 1   152 x 152 x  64   ->   152 x 152 x  64
   19 conv     64  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x  64
   20 shortcut 17
   21 conv     64  1 x 1 / 1   152 x 152 x  64   ->   152 x 152 x  64
   22 route  21 12
   23 conv    128  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x 128
   24 conv    256  3 x 3 / 2   152 x 152 x 128   ->    76 x  76 x 256
   25 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
   26 route  24
   27 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
   28 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   29 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   30 shortcut 27
   31 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   32 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   33 shortcut 30
   34 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   35 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   36 shortcut 33
   37 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   38 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   39 shortcut 36
   40 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   41 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   42 shortcut 39
   43 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   44 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   45 shortcut 42
   46 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   47 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   48 shortcut 45
   49 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   50 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   51 shortcut 48
   52 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   53 route  52 25
   54 conv    256  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 256
   55 conv    512  3 x 3 / 2    76 x  76 x 256   ->    38 x  38 x 512
   56 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
   57 route  55
   58 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
   59 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   60 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   61 shortcut 58
   62 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   63 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   64 shortcut 61
   65 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   66 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   67 shortcut 64
   68 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   69 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   70 shortcut 67
   71 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   72 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   73 shortcut 70
   74 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   75 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   76 shortcut 73
   77 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   78 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   79 shortcut 76
   80 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   81 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   82 shortcut 79
   83 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   84 route  83 56
   85 conv    512  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 512
   86 conv   1024  3 x 3 / 2    38 x  38 x 512   ->    19 x  19 x1024
   87 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
   88 route  86
   89 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
   90 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
   91 conv    512  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x 512
   92 shortcut 89
   93 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
   94 conv    512  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x 512
   95 shortcut 92
   96 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
   97 conv    512  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x 512
   98 shortcut 95
   99 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
  100 conv    512  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x 512
  101 shortcut 98
  102 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
  103 route  102 87
  104 conv   1024  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x1024
  105 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  106 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  107 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  108 max          5 x 5 / 1    19 x  19 x 512   ->    19 x  19 x 512
  109 route  107
  110 max          9 x 9 / 1    19 x  19 x 512   ->    19 x  19 x 512
  111 route  107
  112 max          13 x 13 / 1    19 x  19 x 512   ->    19 x  19 x 512
  113 route  112 110 108 107
  114 conv    512  1 x 1 / 1    19 x  19 x2048   ->    19 x  19 x 512
  115 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  116 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  117 conv    256  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 256
  118 upsample           * 2    19 x  19 x 256   ->    38 x  38 x 256
  119 route  85
  120 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  121 route  120 118
  122 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  123 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  124 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  125 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  126 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  127 conv    128  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 128
  128 upsample           * 2    38 x  38 x 128   ->    76 x  76 x 128
  129 route  54
  130 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
  131 route  130 128
  132 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
  133 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256
  134 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
  135 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256
  136 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
  137 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256
  138 conv    255  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 255
  139 detection
  140 route  136
  141 conv    256  3 x 3 / 2    76 x  76 x 128   ->    38 x  38 x 256
  142 route  141 126
  143 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  144 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  145 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  146 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  147 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  148 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  149 conv    255  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 255
  150 detection
  151 route  147
  152 conv    512  3 x 3 / 2    38 x  38 x 256   ->    19 x  19 x 512
  153 route  152 116
  154 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  155 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  156 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  157 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  158 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  159 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  160 conv    255  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 255
  161 detection
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
STRAT detection
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:01<00:00,  9.29it/s]
Convert results to COCO format
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [00:00<00:00, 704.63it/s]
Loading and preparing results...
DONE (t=0.05s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.87s).
Accumulating evaluation results...
DONE (t=0.24s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.512
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.725
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.632
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.754
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.676
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.811
```

example output (yolov4-tiny)
```
python test_parsing.py \
  -cfgfile '/workspace/GitHub/YOLO/cfg/yolov4-tiny.cfg' \
  -weights '/workspace/GitHub/YOLO/weights/yolov4-tiny.weights'
```
```
convalution havn't activate linear
convalution havn't activate linear
layer     filters    size              input                output
    0 conv     32  3 x 3 / 2   416 x 416 x   3   ->   208 x 208 x  32
    1 conv     64  3 x 3 / 2   208 x 208 x  32   ->   104 x 104 x  64
    2 conv     64  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x  64
    3 route  2
    4 conv     32  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x  32
    5 conv     32  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  32
    6 route  5 4
    7 conv     64  1 x 1 / 1   104 x 104 x  64   ->   104 x 104 x  64
    8 route  2 7
    9 max          2 x 2 / 2   104 x 104 x 128   ->    52 x  52 x  64
   10 conv    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128
   11 route  10
   12 conv     64  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x  64
   13 conv     64  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x  64
   14 route  13 12
   15 conv    128  1 x 1 / 1    52 x  52 x 128   ->    52 x  52 x 128
   16 route  10 15
   17 max          2 x 2 / 2    52 x  52 x 256   ->    26 x  26 x 128
   18 conv    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256
   19 route  18
   20 conv    128  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 128
   21 conv    128  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 128
   22 route  21 20
   23 conv    256  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 256
   24 route  18 23
   25 max          2 x 2 / 2    26 x  26 x 512   ->    13 x  13 x 256
   26 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512
   27 conv    256  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 256
   28 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512
   29 conv    255  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 255
   30 detection
   31 route  27
   32 conv    128  1 x 1 / 1    13 x  13 x 256   ->    13 x  13 x 128
   33 upsample           * 2    13 x  13 x 128   ->    26 x  26 x 128
   34 route  33 23
   35 conv    256  3 x 3 / 1    26 x  26 x 384   ->    26 x  26 x 256
   36 conv    255  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 255
   37 detection
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
STRAT detection
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 19.44it/s]
Convert results to COCO format
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [00:00<00:00, 817.01it/s]
Loading and preparing results...
DONE (t=0.04s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.71s).
Accumulating evaluation results...
DONE (t=0.23s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.518
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.092
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.56
 ```

--------------------

pytorch -> onnx 는 demo_pytorch2onnx.py 참조

onnx -> tensorrt 는 https://github.com/khlee369/Pytorch2TensorRT 참조
