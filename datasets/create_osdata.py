##########################################################################################################################################
# This script converts VOC and COCO into open-set/closed-set forms.
# VOC and COCO must be located in the datasets/data/ folder and have their typical structure (detailed in the github).
# Takes in a --dataset argument which can be either voc or coco.
# Dimity Miller, 2021
##########################################################################################################################################


import argparse
import tqdm
import matplotlib.pyplot as plt
import os
import shutil
from mmdet.datasets import build_dataset
import numpy as np
import json
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

def parse_args():
	parser = argparse.ArgumentParser(description='Test with Distance')
	parser.add_argument('--dataset', default = 'voc', help='voc or coco')
	args = parser.parse_args()
	return args

args = parse_args()

print(f'Converting {args.dataset} to a closed-set form.')

img_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
###################### Load in datasets ########################
if args.dataset == 'voc':

    vocData2007 = dict(samples_per_gpu = 1, workers_per_gpu = 4,
                train = dict(type = 'VOCDataset',
                    ann_file=  BASE_DATA_FOLDER+'/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
                    img_prefix= BASE_DATA_FOLDER+'/VOCdevkit/VOC2007/',
                    pipeline=img_pipeline),
                    test = dict(type = 'VOCDataset',
                    ann_file=  BASE_DATA_FOLDER+'/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
                    img_prefix= BASE_DATA_FOLDER+'/VOCdevkit/VOC2007/',
                    pipeline=img_pipeline))

    vocData2012 = dict(samples_per_gpu = 1, workers_per_gpu = 4,
                train = dict(type = 'VOCDataset',
                    ann_file=  BASE_DATA_FOLDER+'/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
                    img_prefix= BASE_DATA_FOLDER+'/VOCdevkit/VOC2012/',
                    pipeline=img_pipeline))

    
    print('Building datasets')
    vocDataset2007Train = build_dataset(vocData2007['train'])
    vocDataset2012Train = build_dataset(vocData2012['train'])
    vocDataset2007Test = build_dataset(vocData2007['test'])

    #for each dataset, collect classes in each image and their filenames
    imClasses = {'2007Train': [], '2012Train': [], '2007Test': []}
    fileNames = {'2007Train': [], '2012Train': [], '2007Test': []}

    dNames = ['2007Train', '2012Train', '2007Test']
    for dIdx, dataset in enumerate([vocDataset2007Train, vocDataset2012Train, vocDataset2007Test]):
        for imIdx in range(len(dataset)):
            imInfo = dataset.get_ann_info(imIdx)
            clsesPresent = list(imInfo['labels']) + list(imInfo['labels_ignore'])
            imClasses[dNames[dIdx]] += [clsesPresent]
            fileNames[dNames[dIdx]] += [dataset.data_infos[imIdx]['filename']]


    #for VOC, the first 15 classes (0-14) are 'known' classes, and the rest are 'unknown'
    cutoffCls = 14

    totalTrainImgs = len(vocDataset2007Train) + len(vocDataset2012Train)
    includedTrainImgs = 0
    totalTrainInstances = [0 for i in range(20)]
    includedTrainInstances = [0 for i in range(20)]
    filesIncluded = {'2007Train': [], '2012Train': [], '2007Test':[]}

    for dIdx, dName in enumerate(dNames):
        for imIdx, imCls in enumerate(imClasses[dName]):
            #statistics of original instance distribution in training data
            if 'Train' in dName:
                for cl in imCls:
                    totalTrainInstances[cl] += 1

            mask = np.asarray(imCls) > cutoffCls

            #if the image has any 'unknown' classes, it is not included in the new training dataset
            if np.sum(mask) != 0:
                continue

            #otherwise it is included in the new training dataset
            filesIncluded[dName] += [fileNames[dName][imIdx]]

            #statistics of new instance distribution in training data
            if 'Train' in dName:
                for cl in imCls:
                    includedTrainInstances[cl] += 1
                includedTrainImgs += 1

    #let's check the data balance in our new training dataset
    plt.figure()
    plt.plot(includedTrainInstances, label = 'New Training dataset')
    plt.plot(totalTrainInstances, label = 'Original Training dataset')
    plt.ylabel('Number of instances')
    plt.xlabel('Class ID')
    plt.legend()
    plt.title('Number of class instances between original and new training dataset')
    plt.show()

    #data balance as a percent of original training dataset?
    percentInstances = np.array(includedTrainInstances)/np.array(totalTrainInstances)
    plt.figure()
    plt.plot(percentInstances)
    plt.ylabel('% class instances retained')
    plt.xlabel('Class ID')
    plt.title('Percent of class instances retained in new training dataset')
    plt.show()

    print('Moving images to new closed-set dataset.')
    #move images that don't have the unknown classes, creating our new closed-set training dataset
    for dIdx, dName in enumerate(dNames):
        yr = dName.replace('Train', '').replace('Test', '')
        source_folder = f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}/JPEGImages/'
        destination_folder = f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}CS/JPEGImages/'

        #check destination folder exists, else create
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)

        for filename in tqdm.tqdm(filesIncluded[dName], total = len(filesIncluded[dName])):
            nm = filename.replace('JPEGImages/', '')
            shutil.copy(os.path.join(source_folder, nm), os.path.join(destination_folder, nm))

    #Fix annotations for closed-set training, validation and test dataset
    for yr in ['2007', '2012']:
        for split in ['trainval', 'train', 'val', 'test']:
            if yr == '2012' and split == 'test':
                continue #doesn't exist

            print(f'Changing annotation for VOC{yr} {split} split')
            source_file = f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}/ImageSets/Main/{split}.txt'
            destination_file = f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}CS/ImageSets/Main/{split}.txt'

            if not os.path.isdir(f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}CS/ImageSets/Main'):
                os.makedirs(f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}CS/ImageSets/Main')
            
            readFile = open(source_file, 'r')
            writeFile = open(destination_file, 'w')




            for x in readFile:
                xFormat = 'JPEGImages/'+x.replace('\n', '')+'.jpg'
                if yr == '2007':
                    if xFormat in filesIncluded['2007Train'] or xFormat in filesIncluded['2007Test']:
                        writeFile.write(x)
                else:
                    if xFormat in filesIncluded['2012Train']:
                        writeFile.write(x)
                    
            writeFile.close()

    print('Completed converting VOC to VOC-CS.')

elif args.dataset == 'coco':
    COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush')

    cocoData = dict(samples_per_gpu = 8, workers_per_gpu = 4,
                train = dict(type = 'CocoDataset',
                    ann_file=  BASE_DATA_FOLDER+'/coco/annotations/instances_train2017.json',
                    img_prefix= BASE_DATA_FOLDER+'/coco/train2017/',
                    pipeline=img_pipeline),
                test = dict(type = 'CocoDataset',
                    ann_file=  BASE_DATA_FOLDER+'/coco/annotations/instances_val2017.json',
                    img_prefix= BASE_DATA_FOLDER+'/coco/val2017/',
                    pipeline=img_pipeline))

    print('Building datasets')
    cocoTrainDataset = build_dataset(cocoData['train'])
    cocoTestDataset = build_dataset(cocoData['test'])


    imClasses = {'train': [], 'test': []}
    fileNames = {'train': [], 'test': []}
    numIgnores = {'train': [], 'test': []}

    cocoDatasets = {'train': cocoTrainDataset, 'test': cocoTestDataset}

    for split in ['train', 'test']:
        for imIdx in range(len(cocoDatasets[split])):
            imInfo = cocoDatasets[split].get_ann_info(imIdx)
            #ignore bboxes are always crowds, which will be class person, which we will always include because it is a known class
            clsesPresent = list(imInfo['labels']) 
            imClasses[split] += [clsesPresent]
            fileNames[split] += [cocoDatasets[split].data_infos[imIdx]['filename']]
            numIgnores[split] += [len(imInfo['bboxes_ignore'])]

    #first 50 classes are known (0-49), rest are unknown
    cutoffCls = 49

    totalInstances = {'train':[0 for i in range(80)], 'test':[0 for i in range(80)]}
    includedInstances = {'train':[0 for i in range(80)], 'test':[0 for i in range(80)]}
    namesIncluded = {'train': [], 'test': []}
    for split in ['train', 'test']:
        for imIdx, imCls in enumerate(imClasses[split]):
            #statistics of original instance distribution in training data
            for cl in imCls:
                totalInstances[split][cl] += 1

            mask = np.asarray(imCls) > cutoffCls
            if np.sum(mask) != 0:
                continue
                
            namesIncluded[split] += [fileNames[split][imIdx]]
            #statistics of original instance distribution in training data
            for cl in imCls:
                includedInstances[split][cl] += 1

    #check the distribution of the new training dataset
    #let's check the data balance in our new training and test dataset

    plt.figure()
    plt.plot(includedInstances['train'], label = f'New training dataset')
    plt.plot(totalInstances['train'], label = f'Original training dataset')
    plt.ylabel('Number of instances')
    plt.xlabel('Class ID')
    plt.legend()
    plt.title(f'Number of class instances between original and new training dataset')
    plt.show()

    #as a percent of original training dataset?
    percentInstances = np.array(includedInstances['train'])/np.array(totalInstances['train'])
    plt.figure()
    plt.plot(percentInstances)
    plt.ylabel('% class instances retained')
    plt.xlabel('Class ID')
    plt.title(f'Percent of class instances retained in new training dataset')
    plt.show()


    #split the training data into training data and validation data
    totalTrainIms = len(namesIncluded['train'])
    newTrainIms = totalTrainIms*0.8
    namesIncluded['trainNew'] = []
    namesIncluded['val'] = []
    includedInstancesTrain = [0 for i in range(80)]
    
    count = 0
    for idx, imCls in enumerate(imClasses['train']):
        mask = np.asarray(imCls) > cutoffCls
        if np.sum(mask) != 0:
            continue
        if count <= newTrainIms:
            for cl in imCls:
                includedInstancesTrain[cl] += 1
            namesIncluded['trainNew'] += [fileNames['train'][idx]]
        else:
            namesIncluded['val'] += [fileNames['train'][idx]]
        count += 1
        
    plt.figure()
    plt.plot(np.array(includedInstancesTrain)/np.array(includedInstances['train']))
    plt.xlabel('Class ID')
    plt.ylabel('Percent in new training split - 80% desired')
    plt.title('Class Instances split between training and validation dataset')
    plt.show()

    print('Moving images to new closed-set folder')
    source_folders = [BASE_DATA_FOLDER+'/coco/images/train2017/', BASE_DATA_FOLDER+'/coco/images/train2017/', BASE_DATA_FOLDER+'/coco/images/val2017/']
    destination_folders = [BASE_DATA_FOLDER+'/coco/images/{split}CS2017/' for split in ['train', 'val', 'test']]

    #check destination folder exists, else create
    for folder in destination_folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    for spIdx, split in enumerate(['trainNew', 'val', 'test']):
        for filename in tqdm.tqdm(namesIncluded[split]):
            shutil.copy(os.path.join(source_folders[spIdx], filename), os.path.join(destination_folders[spIdx], filename))

    #fix annotations
    source_fileTrain = BASE_DATA_FOLDER+'/coco/annotations/instances_train2017.json'
    source_fileTest = BASE_DATA_FOLDER+'/coco/annotations/instances_val2017.json'
    
    destination_fileTrain = BASE_DATA_FOLDER+'/coco/annotations/instances_trainCS2017.json'
    destination_fileVal = BASE_DATA_FOLDER+'/coco/annotations/instances_valCS2017.json'
    destination_fileTest = BASE_DATA_FOLDER+'/coco/annotations/instances_testCS2017.json'
    destination_files = {'trainNew': destination_fileTrain, 'val': destination_fileVal, 'test': destination_fileTest}


    with open(source_fileTrain) as f:
        readFileTrain = json.load(f)
    with open(source_fileTest) as f:
        readFileTest = json.load(f)

    readFiles = [readFileTrain, readFileTrain, readFileTest]
    

    for spIdx, split in enumerate(['trainNew', 'val', 'test']):
        print(f'Changing annotations for {split} split')
        readFile = readFiles[spIdx]

        writeFile = {}

        writeFile['info'] = readFile['info']
        writeFile['licenses'] = readFile['licenses']

        writeFile['categories'] = readFile['categories'][:50]

        writeFile['images'] = []
        writeFile['annotations'] = []

        for imKey in tqdm.tqdm(readFile['images']):
            name = imKey['file_name']
            if name in namesIncluded[split]:
                writeFile['images'] += [imKey]

        for annKey in tqdm.tqdm(readFile['annotations']):
            category = annKey['category_id']
            if category > 55: # this corresponds to 50th class in coco
                continue
            name = str(annKey['image_id']).zfill(12) + '.jpg'
            if name in namesIncluded[split]:
                writeFile['annotations'] += [annKey]


        with open(destination_files[split], 'w') as outFile:
            json.dump(writeFile, outFile)

else:
    print('This dataset is not implemented.')
    exit()

