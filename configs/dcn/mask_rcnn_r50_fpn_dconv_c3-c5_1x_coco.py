_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
classes = ('txd', 'jgc', 'xbs', 'wbs', 'c-pg', 'lwz', 'tc', 'a-pg', 'b-pg', 'g-pg', 'z-pg', 'bbt', 'lxb', 'xgg', 'lsd',
           'wt',)
# runner = dict(type='EpochBasedRunner', max_epochs=55)
dataset_type = 'CocoDataset'
data_root = '/root/SunNet/data_loader/'
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
data = dict(
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
