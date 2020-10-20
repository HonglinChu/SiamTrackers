from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import numpy as np

# 这里的权重初始化？？？？
def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

#对图像进行裁剪和缩放
def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size) # 四 舍 五 入
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),         # top-left （x1，y1）
        np.round(center - (size - 1) / 2) + size)) # right-down （x2，y2）
    corners = np.round(corners).astype(int) #  

    ''' type()  返回数据结构类型(list,class, dict, numpy等)
        dtype() 返回数据元素的数据类型（int、float等） 由于list，dict可以包含不同的数据类型，因此不可以调用dtype()
                numpy.array 中要求所有的元素属于同一个数据类型，因此可以调用dtype() 函数
        astype()改变 np.array中所有元素的数据类型 备注：能用dtype才能用astype 
    '''
    # pad image if necessary 如果corners越界了，那么（x1，y1）<(0,0) 或者（x2，y2）>(img.width,img.height)
    pads = np.concatenate((-corners[:2], corners[2:] - img.shape[:2])) #得到的都是负数，没有越界

    npad = max(0, int(pads.max()))
    if npad > 0:  # 如果扩展之后的图像越界了，才回添加padding填充
        # 复制图像并制作边界
        # src，top，bottom，left，right，这四个参数指定输出图像4个方向要扩展多少像素，borderType边框类型，
        img = cv2.copyMakeBorder(img, npad, npad, npad, npad,border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int) #得到新的角点坐标
    patch = img[corners[0]:corners[2], corners[1]:corners[3]] #获取图像块

    # resize to out_size resize 到  127x127
    patch = cv2.resize(patch, (out_size, out_size),interpolation=interp)
    
    return patch
