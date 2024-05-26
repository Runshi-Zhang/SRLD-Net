

#python tools/train.py configs/3dnii/3dunet7.py
#bash ./tools/dist_train.sh configs/3dnii/3dunet7.py 2
#CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/3dnii/3dunet7.py 1
crop_size = (128, 128, 128)
stride = (85, 85, 85)
img_scale = (640, 640, 640)
custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)
# model settings
norm_cfg = dict(type='BN3d')
#norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    pad_val=0,
    seg_pad_val=255)
# class_weight = [0.8,1.1,0.9,1.2,1.25,1.4]
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SRUNet',
        in_channels=1,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        # upsample_cfg=dict(type='DeconvModule'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=norm_cfg,
        align_corners=False,

        loss_decode=[
            dict(type='PointtwoMSELoss', loss_name='loss_point', loss_weight=1.0)
        ],
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    #test_cfg=dict(mode='slide',crop_size=crop_size, stride=stride)
)

dataset_type = 'DecathlonDataset'
# data_root = '/home/zrs/PycharmProjects/mmsegmentation/configs/3dnii/alldata/'
data_root = '/home/zrs/PycharmProjects/point/configs/3dnii/bi/'
train_pipeline = [
    dict(type='LoadBiomedicalImageFromFile', decode_backend='numpy'),
    dict(type='LoadBiomedicalAnnotation', decode_backend='numpy', trans=True, train=True),
    # dict(type='BioMedical3DRandomFlip', prob=0.5, axes=(1,2)),
    # dict(type='BioMedical3DPad', pad_shape=img_scale,seg_pad_val=255),
    # dict(type='BioMedical3DRandomCrop',crop_shape = (160,160,160)),
    dict(type='nnUnetTransform', patch_size=crop_size, trans=True),
    # dict(type='BioMedical3DRandomFlip', prob=0.5, axes=(0,1,2)),
    dict(type='PackSegInputsNii')
]
test_pipeline = [
    dict(type='LoadBiomedicalImageFromFile', decode_backend='numpy'),
    # dict(type='BioMedical3DPad', pad_shape=img_scale,seg_pad_val=255),
    # dict(type='BioMedical3DRandomCrop',crop_shape = crop_size),
    dict(type='LoadBiomedicalAnnotation', decode_backend='numpy', trans=True, train=False),
    dict(type='PackSegInputsNii')
]

# metainfo=dict(classes=( 'ZY', 'AR'))
work_dir = '/home/zrs/PycharmProjects/point/work_dirs/unetbi8'
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    # pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='traindataset.json',
        # metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    # pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # metainfo=metainfo,
        data_root=data_root,
        ann_file='valdataset.json',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='EightMetric', iou_metrics=['mDice'], collect_device='gpu')

# optimizer
'''
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=10000,
        by_epoch=False)
]
'''
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.0001)
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=10000,
        eta_min=0.,
        by_epoch=False,
    )
]

# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=500)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
test_evaluator = val_evaluator

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
#load_from = r'/home/zrs/PycharmProjects/point/work_dirs/unetbi8/iter_10000_2_raw.pth'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
