_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='ATSS',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=16,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
classes = ('txd', 'jgc', 'xbs', 'wbs', 'c-pg', 'lwz', 'tc', 'a-pg', 'b-pg', 'g-pg', 'z-pg', 'bbt', 'lxb', 'xgg', 'lsd',
           'wt',)
runner = dict(type='EpochBasedRunner', max_epochs=55)
dataset_type = 'CocoDataset'
data_root = '/root/SunNet/data_loader/'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes = classes,
        img_prefix=data_root + 'final_images_test(1+3+4+5+6)/train_img/',
        ann_file=data_root + 'final_images_test(1+3+4+5+6)/train_a.json',
        ),
    val=dict(
        type=dataset_type,
        classes = classes,
        img_prefix=data_root + 'final_images_test(1+3+4+5+6)/val_img/',
        ann_file=data_root + 'final_images_test(1+3+4+5+6)/valid_a.json',
        ),
    test=dict(
        type=dataset_type,
        classes = classes,
        img_prefix=data_root + 'final_images_test(1+3+4+5+6)/test_img/',
        ann_file=data_root + 'final_images_test(1+3+4+5+6)/test_a.json',
        ))
load_from = '/root/Swin-Transformer-Object-Detection/2021_7_30/epoch_12.pth'