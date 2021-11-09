# dataset settings

from base_dirs import BASE_DATA_FOLDER

dataset_type = 'VOCDataset'
data_root = BASE_DATA_FOLDER+'/VOCdevkit/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007CS/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012CS/ImageSets/Main/train.txt'
            ],
            img_prefix=[data_root + 'VOC2007CS/', data_root + 'VOC2012CS/'],
            pipeline=train_pipeline)),
    trainCS12=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012CS/ImageSets/Main/train.txt',
        img_prefix=data_root + 'VOC2012CS/',
        pipeline=test_pipeline),
    trainCS07=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007CS/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007CS/',
        pipeline=test_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012CS/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2012CS/',
        pipeline=test_pipeline),
    testCS=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007CS/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007CS/',
        pipeline=test_pipeline),
    testOS=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
