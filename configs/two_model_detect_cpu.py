import argparse
import os
import shutil
import time
import numpy as np
from pathlib import Path
from ensemble_boxes import *
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, xywh2xyxy, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from mmdet.apis import init_detector, inference_detector

data_root = '/AI/Swin-Transformer-Object-Detection/'
config_file = data_root + 'configs/swin.py'
checkpoint_file = data_root + '/work_dirs/swin/epoch_12.pth'

# build the model from a config file and a checkpoint file
swin_model = init_detector(config_file, checkpoint_file)


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img_before = img
        img = torch.from_numpy(img).to(device)
        # img_before = img
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        nms_pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(nms_pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            swin_img = cv2.imread(p)
            result = inference_detector(swin_model, swin_img)
            swin_bbox_list, swin_score_list, swin_label_list = swin_model.show_result(swin_img, result,
                                                                                      out_file=save_path)

            yolo_bbox_list = det[:, 0:4].cpu().detach().numpy().tolist()
            yolo_score_list = det[:, 4].cpu().detach().numpy().tolist()
            yolo_label_list = det[:, 5].cpu().detach().numpy().tolist()

            swin_list = ['txd', 'jgc', 'xbs', 'wbs', 'c-pg', 'lwz', 'tc', 'a-pg', 'b-pg', 'g-pg', 'z-pg', 'bbt', 'lxb',
                         'xgg', 'lsd', 'wt']
            yolo_list = ['wt', 'jgc', 'lsd', 'lxb', 'bbt', 'xgg', 'txd', 'lwz', 'tc', 'xbs', 'wbs', 'a-pg', 'b-pg',
                         'c-pg', 'g-pg', 'z-pg']

            swin_trueLabel_list = []
            for i in swin_label_list:
                swin_trueLabel_list.append(yolo_list.index(swin_list[i]))

            # nms_bbox, nms_score, nms_label = swin_bbox_list, swin_score_list, swin_trueLabel_list
            # for i in range(len(yolo_bbox_list)):
            #     nms_bbox.append(yolo_bbox_list[i])
            #     nms_score.append(yolo_score_list[i])
            #     nms_label.append(yolo_label_list[i])
            # nms_bbox, nms_score, nms_label = torch.from_numpy(np.array(nms_bbox)).reshape(-1, 4), torch.from_numpy(
            #     np.array(nms_score)).reshape(-1, 1), torch.from_numpy(np.array(nms_label)).reshape(-1, 1)
            # two_det = torch.cat((torch.cat((nms_bbox, nms_score), 1), nms_label), 1)

            # normalize
            # 需要将框进行归一化操作
            # for i, single in enumerate(swin_bbox_list):
            #     swin_bbox_list[i] = [single[0] / 640, single[1] / 480, single[2] / 640, single[3] / 480]
            #
            # for i, single in enumerate(yolo_bbox_list):
            #     yolo_bbox_list[i] = [single[0] / 640, single[1] / 480, single[2] / 640, single[3] / 480]

            swin_object = [0, 1, 2, 3, 7, 8, 9, 10]  # from yolo_list:wt lsd lwz tc xbs wbs
            # yolo_list = ['0wt', 'jgc', '2lsd', 'lxb', '4bbt', 'xgg', '6txd', 'lwz', '8tc', 'xbs', '10wbs', 'a-pg', '12b-pg',
            #              'c-pg', '14g-pg', 'z-pg']
            yolo_label_list_copy = yolo_label_list.copy()
            swin_trueLabel_list_copy = swin_trueLabel_list.copy()
            for i in yolo_label_list_copy:
                if i in swin_object:
                    index1 = yolo_label_list.index(i)
                    del yolo_bbox_list[index1]
                    del yolo_score_list[index1]
                    del yolo_label_list[index1]

            for i in swin_trueLabel_list_copy:
                if i not in swin_object:
                    index2 = swin_trueLabel_list.index(i)
                    del swin_bbox_list[index2]
                    del swin_score_list[index2]
                    del swin_trueLabel_list[index2]
            two_bbox, two_score, two_label = copy.deepcopy(swin_bbox_list), copy.deepcopy(swin_score_list), copy.deepcopy(swin_trueLabel_list)
            for i in range(len(yolo_bbox_list)):
                two_bbox.append(yolo_bbox_list[i])
                two_score.append(yolo_score_list[i])
                two_label.append(yolo_label_list[i])
            two_bbox, two_score, two_label = torch.from_numpy(np.array(two_bbox)).reshape(-1, 4), torch.from_numpy(
                np.array(two_score)).reshape(-1, 1), torch.from_numpy(np.array(two_label)).reshape(-1, 1)


            yolo_bbox_list, yolo_score_list, yolo_label_list = torch.from_numpy(np.array(yolo_bbox_list)).reshape(-1,
                                                                                                                  4), torch.from_numpy(
                np.array(yolo_score_list)).reshape(-1, 1), torch.from_numpy(np.array(yolo_label_list)).reshape(-1, 1)

            swin_bbox_list, swin_score_list, swin_trueLabel_list = torch.from_numpy(np.array(swin_bbox_list)).reshape(
                -1,
                4), torch.from_numpy(
                np.array(swin_score_list)).reshape(-1, 1), torch.from_numpy(np.array(swin_trueLabel_list)).reshape(-1,
                                                                                                                   1)

            # det = torch.cat((torch.cat((swin_bbox_list, swin_score_list), 1), swin_trueLabel_list), 1)    # only show swin_model inference result
            # det = torch.cat((torch.cat((yolo_bbox_list, yolo_score_list), 1), yolo_label_list),1)  # only show yolo_model inference result
            det = torch.cat((torch.cat((two_bbox, two_score), 1), two_label), 1)    #  show two_model inference result

            # bbox_list = [swin_bbox_list, yolo_bbox_list]
            # score_list = [swin_score_list, yolo_score_list]
            # label_list = [swin_trueLabel_list, yolo_label_list]
            #
            # wbf_weight = [1, 1]
            # iou_thr = 0.55
            # skip_box_thr = 0.0001
            #
            # boxes, scores, labels = weighted_boxes_fusion(bbox_list, score_list, label_list, weights=wbf_weight,
            #                                              iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            # for in_file in boxes:
            #     in_file[0], in_file[1], in_file[2], in_file[3] = int(in_file[0] * 640), int(in_file[1] * 480), int(
            #         in_file[2] * 640), int(in_file[3] * 480)
            # boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1, 1), labels.reshape(-1, 1)
            # boxes, scores, labels = torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(labels)
            # det2model = torch.cat((torch.cat((boxes, scores), 1), labels), 1)
            # det = det2model

            if det is not None and len(det):
                numers = len(det)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results   包围框、置信度、种类
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        img1 = im0.copy()
                        # if cv2.waitKey(1)==32:
                        #     count = 0
                        #     for filename in os.listdir('new_image/'):
                        #         if filename.endswith('.jpg'):
                        #             count += 1
                        #     # print(count)
                        #     print(f"保存第{count + 1}张图片")
                        #     # 保存图像，保存到上一层的imgs文件夹内，以1、2、3、4...为文件名保存图像
                        #     cv2.imwrite('new_image/{}.jpg'.format(count + 1), img1)
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=0.5)  # 线的粗细
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)  # 线的粗细


                    # print(f"\n{names[int(cls)]}的包围框坐标是{int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])}")
                    # print(f"\n{names[int(cls)]}的中心坐标是{(int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])+int(xyxy[3]))/2}")
            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))
            print(f"{s}")
            print(f"s")

            # 打印坐标、种类
            # print('%s' % (names[int(cls)]))

            # Stream results
            # view_img = True
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    txt = f".numers={numers}"
                    cv2.putText(im0, txt,
                                (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (34, 157, 255), 2)
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
            im_after = im0

            plt.figure()
            plt.subplot(1, 3, 1)
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.imshow(img_before.transpose(1, 2, 0))
            plt.subplot(1, 3, 2)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im_after)
            plt.show()
            break

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='version_4.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/AI/Swin-Transformer-Object-Detection/configs/inference/test_7.1', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='/AI/Swin-Transformer-Object-Detection/configs/inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
