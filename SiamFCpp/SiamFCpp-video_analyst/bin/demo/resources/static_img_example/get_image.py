# -*- coding: utf-8 -*
import os.path as osp

import cv2
import numpy as np

import torch

DEBUG = False

im_path = osp.join(osp.dirname(osp.realpath(__file__)), 'example.jpg')
im = cv2.imread(im_path)
bbox = (256, 283, 341, 375)  # order=x0y0x1y1
rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])  # order=xywh
# cv2.rectangle(im, bbox[:2], bbox[2:], (0, 255, 255))
box = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2,
       (bbox[2] - bbox[0] + 1), (bbox[3] - bbox[1] + 1))
z_s = np.sqrt(
    (box[2] + (box[2] + box[3]) / 2) * (box[3] + (box[2] + box[3]) / 2))

z_size = 127
x_size = 303
scale = z_size / z_s
x_s = 303 / scale
dx, dy = -50, 20
target_bbox = tuple(
    map(int, (box[0] - z_s / 2, box[1] - z_s / 2, box[0] + z_s / 2,
              box[1] + z_s / 2)))
search_bbox = tuple(
    map(int, (box[0] - x_s / 2 + dx, box[1] - x_s / 2 + dy,
              box[0] + x_s / 2 + dx, box[1] + x_s / 2 + dy)))

im_z = cv2.resize(
    im[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2], :],
    (z_size, z_size))
im_x = cv2.resize(
    im[search_bbox[1]:search_bbox[3], search_bbox[0]:search_bbox[2], :],
    (x_size, x_size))


def imarray_to_tensor(arr):
    arr = np.ascontiguousarray(
        arr.transpose(2, 0, 1)[np.newaxis, ...], np.float32)
    # return torch.tensor(arr).type(torch.Tensor)
    return arr


arr_z = imarray_to_tensor(im_z)
arr_x = imarray_to_tensor(im_x)

if DEBUG:
    cv2.rectangle(im, search_bbox[:2], search_bbox[2:], (255, 0, 0))
    cv2.rectangle(im, target_bbox[:2], target_bbox[2:], (0, 255, 255))
    cv2.imshow('im_z', im_z)
    cv2.imshow('im_x', im_x)
    cv2.imshow('example', im)
    cv2.waitKey(0)
