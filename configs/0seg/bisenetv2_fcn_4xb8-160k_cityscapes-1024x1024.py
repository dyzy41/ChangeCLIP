_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/base_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
data_root = 'data/OpenEarthMap'
crop_size = (512, 512)
metainfo = dict(
                classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat'),
                palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                        [64, 0, 0]])

data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=160000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='train.txt',
        metainfo=metainfo))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='val.txt',
        metainfo=metainfo))
test_dataloader = val_dataloader
