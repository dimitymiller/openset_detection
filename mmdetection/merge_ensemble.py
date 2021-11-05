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
from sampling_detector import SamplingDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Test the data and save the raw detections')
    parser.add_argument('--dataset', default = 'voc', help='voc or coco')
    parser.add_argument('--subset', default = None, help='train or val or test')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    parser.add_argument('--iou', default = 0.8, type = float, help='bbox iou to merge ensemble detections')
    args = parser.parse_args()
    return args

args = parse_args()

num_models = 5

softmaxLayer = torch.nn.Softmax(dim = 1)

#used to take in collection of individual ensemble results and convert into merged ensemble results
merger = SamplingDetector(iou = args.iou)

allOutputs = [None for i in range(num_models)]
#load results from each individual model
save_dir = f'{BASE_RESULTS_FOLDER}/FRCNN/raw/{args.dataset}/{args.subset}'
for i in range(num_models):
    try:
        with open(f'{save_dir}/{args.saveNm}{i}.json', 'r') as f:
            allOutputs[i] = json.load(f)
    except:
        print(f'Missing results file for {save_dir}/{args.saveNm}{i}.json')
        exit()

ensembleResults = {}
for imIdx, imKey in enumerate(allOutputs[0].keys()):
    ensembleResults[imKey] = []
    #collect all detections for this image
    ensemble_detections = []
    for ensIdx in range(num_models):
        detections = np.array(allOutputs[ensIdx][imKey])

        if len(detections) == 0: #no detections
            continue

        if len(ensemble_detections) == 0:
            ensemble_detections = detections
        else:
            ensemble_detections = np.concatenate((ensemble_detections, detections), axis = 0)

    if len(ensemble_detections) == 0:
        continue
  
    #cluster and merge ensemble detections into final detections (don't pass in final column with softmax score)
    final_detections = merger.form_final(ensemble_detections[:, :-1])

    if len(final_detections) == 0: #no valid detections were clustered
        continue

    #calculate new max softmax score and concatenate to detections
    distsT = torch.Tensor(final_detections[:, :-4])
    softmaxScores = softmaxLayer(distsT).numpy()
    scores = np.max(softmaxScores, axis = 1)
    scoresT = np.expand_dims(scores, axis=1)


    imDets = np.concatenate((final_detections, scoresT), 1)
    ensembleResults[imKey] = imDets.tolist()


#save results
jsonRes = json.dumps(ensembleResults)

save_dir = f'{BASE_RESULTS_FOLDER}/FRCNN/raw/{args.dataset}/{args.subset}'
iouEns = str(args.iou).replace('.', '')
f = open(f'{save_dir}/{args.saveNm}Ensemble{iouEns}.json', 'w')
f.write(jsonRes)
f.close()

