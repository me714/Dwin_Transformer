_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py'
model = dict(bbox_head=dict(transform_method='minmax', use_grid_points=True))
classes = ('txd', 'jgc', 'xbs', 'wbs', 'c-pg', 'lwz', 'tc', 'a-pg', 'b-pg', 'g-pg', 'z-pg', 'bbt', 'lxb', 'xgg', 'lsd',
           'wt',)
dataset_type = 'CocoDataset'
data_root = '/root/SunNet/data_loader/'

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