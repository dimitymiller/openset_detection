import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

import numpy as np
import json

CLASSESVOC = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
CLASSESCOCO = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def parse_args():
    parser = argparse.ArgumentParser(description='Test with Distance')
    parser.add_argument('--dataset', default = 'voc', help='voc or coco')
    parser.add_argument('--subset', default = None, help='train or val or test')
    parser.add_argument('--dir', default = None, help='directory of model weights')
    parser.add_argument('--checkpoint', default = 'latest.pth', help='what is the name of the model weights')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    args = parser.parse_args()
    return args

args = parse_args()

#load the config file for the model that will also return logits
if args.dataset == 'voc':
    args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712SplitLogits.py'
else:
    args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocoSplitLogits.py'
    
###################################################################################################
##############Setup Config file ###################################################################
cfg = Config.fromfile(args.config)

# import modules from string list.
if cfg.get('custom_imports', None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg['custom_imports'])
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

# in case the test dataset is concatenated
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True

distributed = False

samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
if samples_per_gpu > 1:
    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)



###################################################################################################
###############Load Dataset########################################################################


if args.dataset == 'voc':
    if args.subset == 'train12':
        dataset = build_dataset(cfg.data.trainDist1)
    elif args.subset == 'train07':
        dataset = build_dataset(cfg.data.trainDist2)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.valOS)
    elif args.subset == 'ood':
        dataset = build_dataset(cfg.data.ood)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.testOS)
else:
    if args.subset == 'train':
        dataset = build_dataset(cfg.data.distTrain)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.distVal)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.distTest)

if args.dataset == 'voc':
    num_classes = 15
else:
    if args.dataset == 'coco':
        num_classes = 50
    else:
        num_classes = 80

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)


# build the model and load checkpoint
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
method_list = [func for func in dir(model) if callable(getattr(model, func))]
print(method_list)
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, 'workDirs/{}/{}'.format(args.dir, args.checkpoint), map_location='cpu')

if 'CLASSES' in checkpoint['meta']:
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
model.eval()

########################################################################################################
########################## COLLECT DATA  ###############################################################
########################################################################################################

num_images = len(data_loader.dataset)

score_threshold = 0.1
print(num_images)
total = 0
allResults = {}
for i, data in enumerate(data_loader):
    if i%500 == 0:
        print('Progress: ', 100.*total/num_images)
    
    imName = data_loader.dataset.data_infos[i]['filename']
    allResults[imName] = []

    total += 1
    all_detections = None
    all_scores = []
    
    with torch.no_grad():
        result = model(return_loss = False, rescale=True, **data)[0]

    for j in range(np.shape(result)[0]):
        dets = result[j]
        if len(dets) == 0:
            continue

        bboxes = dets[:, :4]
        dists = dets[:, 5:]
        scores = dets[:, 4]
        scoresT = np.expand_dims(scores, axis=1)

        mask = np.argmax(dists, axis = 1)==j

        if np.sum(mask) == 0:
            continue

        imDets = np.concatenate((dists, bboxes, scoresT), 1)[mask]
        scores = scores[mask]
        mask2 = scores >= score_threshold

        if np.sum(mask2) == 0:
            continue
        
        imDets = imDets[mask2]

        if all_detections is None:
            all_detections = imDets
        else:
            all_detections = np.concatenate((all_detections, imDets))

    
    imName = data_loader.dataset.data_infos[i]['filename']

    if all_detections is None:
        continue
    else:
        #remove double ups
        detections, idxes = np.unique(all_detections, return_index = True, axis = 0)
   
    
    allResults[imName] = detections.tolist()
   
#save faster rcnn results
jsonRes = json.dumps(allResults)
f = open('results/{}/{}/{}/{}.json'.format('FRCNN', args.dataset, args.data, args.saveNm), 'w')
f.write(jsonRes)
f.close()

