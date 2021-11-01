from mmdet.apis import init_detector, inference_detector
import os
import cv2
import matplotlib.pyplot as plt

data_root = '/root/Swin-Transformer-Object-Detection/'
config_file = '/root/Swin-Transformer-Object-Detection/configs/paa/paa_r50_fpn_1x_coco.py'
checkpoint_file = '/root/Swin-Transformer-Object-Detection/2021_7_33/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

in_folder = '/root/SunNet/data_loader/final_images_test(1+3+4+5+6)/test_img'
out_folder = '/root/SunNet/data_loader/final_images_test(1+3+4+5+6)/test_img_yucepaa'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

for file_name in os.listdir(in_folder):
    img_path = os.path.join(in_folder, file_name)
    img = cv2.imread(img_path)

    # test a single image and show the results
    # img = 'demo/test.jpg'  # or img = mmcv.imread(img    img0, img, label_list = model.show_result(img, result, out_file=save_path)), which will only load it once
    #    img=test_img
    result = inference_detector(model, img)
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    save_path = os.path.join(out_folder, file_name)
    model.show_result(img, result, out_file=save_path)
    # img0, img, label_list = model.show_result(img, result, out_file=save_path)
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    # plt.imshow(img0)
    # plt.subplot(1, 3, 2)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(img)
    # plt.show()

    # break
    # show_result_pyplot(model, img, result)
