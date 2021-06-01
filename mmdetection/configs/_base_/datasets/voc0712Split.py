# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
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
dist_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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
                data_root + 'VOC2007Split/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012Split/ImageSets/Main/train.txt'
            ],
            img_prefix=[data_root + 'VOC2007Split/', data_root + 'VOC2012Split/'],
            pipeline=train_pipeline)),
    trainDist1=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012Split/ImageSets/Main/train.txt',
        img_prefix=data_root + 'VOC2012Split/',
        pipeline=test_pipeline),
    trainDist2=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007Split/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007Split/',
        pipeline=test_pipeline),
    valOS=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012Split/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2012Split/',
        pipeline=test_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007Split/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007Split/',
        pipeline=test_pipeline),
    ood=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/images/val2017/',
        pipeline=test_pipeline),
    testCS=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007Split/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007Split/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007Split/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007Split/',
        pipeline=test_pipeline),
    mapTest=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    testOS=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
