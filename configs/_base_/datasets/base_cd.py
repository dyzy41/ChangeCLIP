# dataset settings
# dataset_type = 'TXTCDDataset'
dataset_type = 'TXTCDDatasetJSON'
data_root = 'data/cd'
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='ConcatCDInput'),
    dict(type='PackCDInputs')
]
test_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConcatCDInput'),
    dict(type='PackCDInputs')
]

tta_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[dict(type='LoadAnnotations')],
                    [dict(type='ConcatCDInput')],
                    [dict(type='PackCDInputs')]])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/A',
            img_path2='train/B',
            seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/A', img_path2='test/B', seg_map_path='test/label'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
