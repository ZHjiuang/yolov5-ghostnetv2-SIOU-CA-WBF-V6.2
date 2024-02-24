import random
from pathlib import Path

import albumentations as A
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

PIC_ROOT = "F:/datas/PROCESS AND DATAS/own_data/OWNFATA/own_datas/yolo/train/img/"
LAB_ROOT = "F:/datas/PROCESS AND DATAS/own_data/OWNFATA/own_datas/yolo/train/label/"  # LAB_ROOT = ""
save_pic_path = "C:/Users/ZJC/Desktop/plus/"
save_lab_path = "C:/Users/ZJC/Desktop/lab/"  # save_lab_path = ""
category_id_to_name = {0: 'tree', 1: 'person', 2: 'supporter'}

project = 'runs/detect'
name = 'exp'
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


# bboxes = [[5.66, 138.95, 147.09, 164.88], [366.7, 80.84, 132.8, 181.84]]
# category_ids = [17, 18]
# category_id_to_name = {17: 'cat', 18: 'dog'}


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""

    x_center, y_center, w, h = bbox
    x_center = x_center * img.shape[1]
    y_center = y_center * img.shape[0]
    w = w * img.shape[1]
    h = h * img.shape[0]
    x_min, x_max, y_min, y_max = int(x_center - w / 2), int(x_center + w / 2), int(y_center - h / 2), int(
        y_center + h / 2)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(1)
    plt.axis('off')
    plt.imshow(img)
    plt.show()


for pic in os.listdir(PIC_ROOT):
    n = os.path.splitext(pic)[0]  # 去掉后缀
    # with open(n + '.txt', 'r') as f:
    #     lab = f.read()
    if LAB_ROOT != '':
        lab = np.loadtxt(LAB_ROOT + n + '.txt', dtype=np.float32, delimiter=' ')  # 获取标签
        lab1 = np.mat(lab)
        lab2 = lab1[:, 1:]
        bboxes = lab2.tolist()
        category_ids1 = lab1[:, 0]
        category_ids2 = category_ids1.tolist()
        category_ids = np.resize(category_ids2, len(bboxes))

    # category_ids = np.resize(category_ids, len(bboxes))

    img = cv2.imread(PIC_ROOT + pic)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if LAB_ROOT != '':
        visualize(image, bboxes, category_ids, category_id_to_name)

    trans = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([A.GaussNoise(),  # 将高斯噪声应用于输入图像。
                 ], p=0.3),  # 应用选定变换的概率
        A.OneOf([
            A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
            A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.OneOf([
            A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),  # 雨
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1),  # 光
            A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1),  # 雪
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1),
                           p=1),  # 阴影
            A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1),  # 雾
        ], p=0.3)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    for i in range(3):
        transformed = trans(image=image, bboxes=bboxes, category_ids=category_ids)
        # 可视化
        if LAB_ROOT != '':
            visualize(
                transformed['image'],
                transformed['bboxes'],
                transformed['category_ids'],
                category_id_to_name,
            )
        # 保存图像，标签
        # img_trans = trans(image=image)['image']  # 字典类型
        img_trans = transformed['image']
        if LAB_ROOT != '':
            bb = transformed['bboxes']
            ca = transformed['category_ids']

            bb1 = np.resize(bb, (len(ca), 4))
            ca1 = np.resize(ca, (len(ca), 1))
            labs = np.c_[ca1, bb1]

            # bc = ca1 + bb1
            # bc1 = (np.mat(bc))
            # np.reshape(bc1,len(ca))
            # bc2 = bc1.reshape(len(ca), 5)

            np.savetxt(f'{save_lab_path}{n}{i}.txt', labs, fmt='%.6f')  # 写入标签
        cv2.imwrite(f'{save_pic_path}{n}{i}.jpg', img_trans)  # 写入图片

# image = cv2.imread('data/images/bus.jpg')
#
# # save_dir = increment_path(Path(project) / name)
# cv2.imshow("gh", img_trans)
# # cv2.imwrite('gh.jpg', img_trans)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# def increment_path(path, exist_ok=False, sep='', mkdir=False):
#     path = Path(path)  # os-agnostic
#     if path.exists() and not exist_ok:
#         path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
#
#         # Method 1
#         for n in range(2, 9999):
#             p = f'{path}{sep}{n}{suffix}'  # increment path
#             if not os.path.exists(p):  #
#                 break
#         path = Path(p)
#
#     if mkdir:
#         path.mkdir(parents=True, exist_ok=True)  # make directory
#
#     return path
