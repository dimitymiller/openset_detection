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
import tqdm
import json
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test the data and save the raw detections')
    parser.add_argument('--dataset', default = 'voc', help='voc or coco')
    parser.add_argument('--subset', default = None, help='train or val or test')
    parser.add_argument('--dir', default = None, help='directory of object detector weights')
    parser.add_argument('--checkpoint', default = 'latest.pth', help='what is the name of the object detector weights')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    args = parser.parse_args()
    return args

args = parse_args()


#load the config file for the model that will also return logits
if args.dataset == 'voc':
    args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712OS_wLogits.py'
else:
    args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocoOS_wLogits.py'
    
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
if isinstance(cfg.data.testOS, dict):
    cfg.data.testOS.test_mode = True
elif isinstance(cfg.data.testOS, list):
    for ds_cfg in cfg.data.testOS:
        ds_cfg.test_mode = True

distributed = False

samples_per_gpu = cfg.data.testOS.pop('samples_per_gpu', 1)
if samples_per_gpu > 1:
    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    cfg.data.testOS.pipeline = replace_ImageToTensor(cfg.data.testOS.pipeline)

###################################################################################################
###############Load Dataset########################################################################
print("Building datasets")
if args.dataset == 'voc':
    num_classes = 15
    if args.subset == 'train12':
        dataset = build_dataset(cfg.data.trainCS12)
    elif args.subset == 'train07':
        dataset = build_dataset(cfg.data.trainCS07)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.valCS)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.testOS)
    else:
        print('That subset is not implemented.')
        exit()
else:
    if args.subset == 'train':
        dataset = build_dataset(cfg.data.trainCS)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.valCS)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.testOS)
    else:
        print('That subset is not implemented.')
        exit()

    if args.dataset == 'coco':
        num_classes = 50
    else:
        #for the full version of coco used to fit GMMs in the iCUB experiments
        num_classes = 80


data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)


###################################################################################################
###############Build model ########################################################################
print("Building model")

# build the model and load checkpoint
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
method_list = [func for func in dir(model) if callable(getattr(model, func))]
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, '{}/{}/{}'.format(BASE_WEIGHTS_FOLDER, args.dir, args.checkpoint), map_location='cpu')

if 'CLASSES' in checkpoint['meta']:
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
model.eval()


########################################################################################################
########################## TESTING DATA  ###############################################################
########################################################################################################
print(f"Testing {args.subset} data")
num_images = len(data_loader.dataset)

score_threshold = 0.2 # only detections with a max softmax above this score are considered valid
total = 0
allResults = {}
for i, data in enumerate(tqdm.tqdm(data_loader, total = num_images)):   
    imName = data_loader.dataset.data_infos[i]['filename']
    
    allResults[imName] = []

    total += 1
    all_detections = None
    all_scores = []
    
    with torch.no_grad():
        result = model(return_loss = False, rescale=True, **data)[0]
    
    #collect results from each class and concatenate into a list of all the results
    for j in range(np.shape(result)[0]):
        dets = result[j]

        if len(dets) == 0:
            continue

        bboxes = dets[:, :4]
        dists = dets[:, 5:]
        scores = dets[:, 4]
        scoresT = np.expand_dims(scores, axis=1)

        #winning class must be class j for this detection to be considered valid
        mask = np.argmax(dists, axis = 1)==j

        if np.sum(mask) == 0:
            continue

        #check thresholds are above the score cutoff
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

    if all_detections is None:
        continue
    else:
        #remove doubled-up detections -- this shouldn't really happen
        detections, idxes = np.unique(all_detections, return_index = True, axis = 0)

    allResults[imName] = detections.tolist()

#save results
jsonRes = json.dumps(allResults)

save_dir = f'{BASE_RESULTS_FOLDER}/FRCNN/raw/{args.dataset}/{args.subset}'
#check folders exist, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open('{}/{}.json'.format(save_dir, args.saveNm), 'w')
f.write(jsonRes)
f.close()

