# dataset settings
dataset_type = 'WoodScapeDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (960, 960)
train_pipeline_syn = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 960), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AlbuDomainAdaption', domain_adaption_type='ALL', target_dir='data/real/rgb_images', p=0.5),
    dict(type='RandomRotate', degree=(-45, 45), prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
train_pipeline_real = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 960), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomRotate', degree=(-45, 45), prob=0.5),
    dict(type='RandomFlip', prob=0.5),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 960),
        # img_ratios=[1.0, 1.5, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

real = dict(
    type=dataset_type,
    data_root=data_root,
    split='real/train.txt',
    img_dir='real/rgb_images',
    ann_dir='real/motion_annotations/gtLabels',
    pipeline=train_pipeline_real)

syn = dict(
    type=dataset_type,
    data_root=data_root,
    split='syn/train_syn.txt',
    img_dir='syn/rgb_images',
    ann_dir='syn/motion_annotations/gtLabels',
    pipeline=train_pipeline_syn)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=[syn, real],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='real/val.txt',
        img_dir='real/rgb_images',
        ann_dir='real/motion_annotations/gtLabels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='test/test.txt',
        img_dir='test/rgb_images(test_set)',
        ann_dir=None,
        pipeline=test_pipeline))
