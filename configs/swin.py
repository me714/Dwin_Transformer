# The new config inherits a base config to highlight the necessary modification
# _base_ = 'Swin-Transformer-Object-Detection/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py'
# _base_ = 'swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
_base_ = 'swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(mask_head=dict(num_classes=16)))

# Modify dataset related settings

classes = ('txd', 'jgc', 'xbs', 'wbs', 'c-pg', 'lwz', 'tc', 'a-pg', 'b-pg', 'g-pg', 'z-pg', 'bbt', 'lxb', 'xgg', 'lsd',
           'wt',)


data_root = '/root/SunNet/data_loader/'
data = dict(
    train=dict(
        img_prefix=data_root + 'final_images_test(1+3+4+5+6+7)/train_img/',
        classes=classes,
        ann_file=data_root + 'final_images_test(1+3+4+5+6+7)/train_a.json'),
    val=dict(
        img_prefix=data_root + 'final_images_test(1+3+4+5+6+7)/val_img/',
        classes=classes,
        ann_file=data_root + 'final_images_test(1+3+4+5+6+7)/valid_a.json')
)

# load_from = '/root/Swin-Transformer-Object-Detection/checkpoints/mask_rcnn_swin_tiny_patch4_window7.pth'
# load_from ='/home/mewtwo/Swin-Transformer-Object-Detection/work_dirs/Swin-Transformer-Object-Detection/latest.pth'
