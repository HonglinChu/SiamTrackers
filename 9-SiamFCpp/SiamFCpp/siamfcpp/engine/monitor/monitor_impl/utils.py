# -*- coding:utf-8 -*-
from queue import Queue

import cv2
import numpy as np
from PIL import Image

palette = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128,
    128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0,
    128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192,
    0, 128, 192, 0, 0, 64, 128
]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    """generate a color map for N classes"""
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


def mask_colorize(mask, num_classes, color_map):
    """
    transfor one mask to a maske with color

    :param mask: mask with shape [h, w]
    :param num_classes: number of classes
    :param color_map: color map with shape [N, 3]
    """
    color_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    color_mask.putpalette(palette)
    raw_mask = np.array(color_mask).astype(np.uint8)
    color_mask = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    for c_index in range(num_classes):
        instance_mask = (raw_mask == c_index)
        if int(cv2.__version__.split(".")[0]) < 4:
            _, contour, hier = cv2.findContours(instance_mask.astype(np.uint8),
                                                cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_NONE)
        else:
            contour, hier = cv2.findContours(instance_mask.astype(np.uint8),
                                             cv2.RETR_CCOMP,
                                             cv2.CHAIN_APPROX_NONE)
        if len(contour) > 0:
            cv2.drawContours(color_mask, contour, -1, (225, 225, 225), 5)
        color_mask[np.where(raw_mask == c_index)] = color_map[c_index]
    return color_mask


class AverageMeter(object):
    def __init__(self, max_num):
        self.queue_data = Queue(max_num)

    def update(self, val):
        if self.queue_data.full():
            self.queue_data.get()
        self.queue_data.put(val)

    def reset(self):
        self.queue_data.queue.clear()

    def get_mean(self):
        return np.mean(self.queue_data.queue)


def fast_hist(label_pred, label_true, num_classes, ignore_label=255):
    mask = (label_true >= 0) & (label_true < num_classes) & (label_true !=
                                                             ignore_label)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes**2).reshape(num_classes, num_classes)
    return hist


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_mask_from_sigmoid(predict, prob_ths):
    """

    :param predict: (N, C, H, W)   
    :param prob_ths: (C)
    """
    prob = np_sigmoid(predict)
    mask = np.zeros_like(predict)
    class_num = predict.shape[1]
    for class_id in range(class_num - 1):
        mask[:, class_id + 1][prob[:, class_id + 1] >= prob_ths[class_id]] = 1
    return mask
