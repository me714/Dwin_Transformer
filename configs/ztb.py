_base_ = 'swin/faster_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation


# Modify dataset related settings

classes = ('apple', 'bananas', 'pitaya', 'snow_pear', 'virgin_fruit', 'kiwi', 'green_mansions',
               'grapes', 'yellow_corn', 'green_cabbage', 'purple_cabbage', 'fresh_cut_purple_cabbage', 'cauliflower',
               'broccoli', 'tomatoes', 'beibei_pumpkin', 'golden_pumpkin', 'green_pepper', 'green_round_pepper',
               'red_round_peppers', 'yellow_pepper', 'eggplant', 'zucchini', 'okra', 'carrots', 'quail_eggs', 'papaya',
               'fresh_cut_papaya', 'spinach', 'lettuce', 'rape', 'hami_melon', 'fresh_cut_Hami_melon',
               'pleurotus_ostreatus', 'green_radish', 'baby_cabbage', 'eggs', 'cucumber', 'yellow_awn', 'green_grape',
               'blueberries', 'strawberries', 'longan', 'hawthorn', 'red_cherry',
               'honey_peach', 'nectarine', 'passion_fruit', 'plum', 'avocado', 'mangosteen', 'oranges',
               'yellow_orange', 'yellow_lemon', 'grapefruit', 'beer_pear', 'fragrant_pear', 'bean_sprouts',
               'oatmeal', 'celery')


data_root = '/home/dhu1/ztb_dataset/'
data = dict(
    samples_per_gpu=4,
    train=dict(
        img_prefix=data_root + 'train/',
        classes=classes,
        ann_file=data_root + 'train/train.json'),
    val=dict(
        img_prefix=data_root + 'valid/',
        classes=classes,
        ann_file=data_root + 'valid/val.json')
)
evaluation = dict(metric=['bbox'])

