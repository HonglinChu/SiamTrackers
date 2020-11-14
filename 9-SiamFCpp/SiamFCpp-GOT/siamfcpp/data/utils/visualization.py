# -*- coding: utf-8 -*-

import cv2
import numpy as np

from siamfcpp.pipeline.utils.misc import tensor_to_imarray, tensor_to_numpy


def show_img_FCOS(cfg,
                  training_data,
                  distractor_boxes_recentered=[],
                  dataset='untitled'):
    r"""
    Visualize training data
    """

    target_img = tensor_to_imarray(training_data["im_z"])
    image_rand_focus = tensor_to_imarray(training_data["im_x"])

    gt_datas = [
        training_data["cls_gt"], training_data["ctr_gt"],
        training_data["box_gt"]
    ]
    gt_datas = [tensor_to_numpy(t) for t in gt_datas]
    gt_target = np.concatenate(gt_datas, axis=-1)
    if gt_target.ndim == 3:
        gt_target = gt_target[0]

    total_stride = cfg.total_stride
    score_size = cfg.score_size
    x_size = cfg.x_size
    num_conv3x3 = cfg.num_conv3x3
    score_offset = (x_size - 1 - (score_size - 1) * total_stride) // 2

    color = dict()
    color['pos'] = (0, 0, 255)
    color['ctr'] = (0, 255, 0)
    color['neg'] = (255, 0, 0)
    color['ign'] = (255, 255, 255)
    # to prove the correctness of the gt box and sample point
    gts = gt_target[gt_target[:, 0] == 1, :][:, 2:]

    show_img = cv2.resize(image_rand_focus, (x_size, x_size))
    show_img_h, show_img_w = show_img.shape[:2]

    fm_margin = score_offset
    pt1 = (int(fm_margin), ) * 2
    pt2 = (int(x_size - fm_margin), ) * 2
    cv2.rectangle(show_img, pt1, pt2, (0, 0, 255))

    gt_indexes = (gt_target[:, 0] == 1)
    if gt_indexes.any():
        print('gt_indexes.size', gt_indexes.size)
        gt = gt_target[gt_indexes, :][0, 2:]
        cv2.rectangle(show_img, (int(gt[0]), int(gt[1])),
                      (int(gt[2]), int(gt[3])), color['pos'])

    pos_cls_gt = (gt_target[:, 0] == 1)
    pos_indexes = np.argsort(pos_cls_gt)[len(gt_target) - np.sum(pos_cls_gt):]

    ctr_gt = gt_target[:, 1]

    ign_cls_gt = (gt_target[:, 0] == -1)
    ign_indexes = np.argsort(ign_cls_gt)[len(gt_target) - np.sum(ign_cls_gt):]

    neg_cls_gt = (gt_target[:, 0] == 0)
    neg_indexes = np.argsort(neg_cls_gt)[len(gt_target) - np.sum(neg_cls_gt):]

    for index in pos_indexes:
        # note that due to ma 's fcos implementation, x and y are switched
        pos = (score_offset + (index % score_size) * total_stride,
               score_offset + (index // score_size) * total_stride)
        ctr = ctr_gt[index]
        color_pos = tuple(
            (np.array(color['pos']) + ctr * np.array(color['ctr'])).astype(
                np.uint8).tolist())
        cv2.circle(show_img, pos, 2, color_pos, -1)

    for index in neg_indexes:
        # note that due to ma 's fcos implementation, x and y are switched
        pos = (score_offset + (index % score_size) * total_stride,
               score_offset + (index // score_size) * total_stride)
        ctr = ctr_gt[index]
        color_neg = tuple(
            (np.array(color['neg']) + ctr * np.array(color['ctr'])).astype(
                np.uint8).tolist())
        cv2.circle(show_img, pos, 2, color_neg, -1)

    for index in ign_indexes:
        # note that due to ma 's fcos implementation, x and y are switched
        pos = (score_offset + (index % score_size) * total_stride,
               score_offset + (index // score_size) * total_stride)
        cv2.circle(show_img, pos, 2, color['ign'], -1)

    cv2.putText(show_img, 'pos', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color['pos'])
    cv2.putText(show_img, 'neg', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color['neg'])
    cv2.putText(show_img, 'ign', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color['ign'])

    cv2.putText(show_img, dataset, (20, show_img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    print('distractor_boxes:', len(distractor_boxes_recentered))
    if len(distractor_boxes_recentered) > 0:
        for box in distractor_boxes_recentered:
            cv2.rectangle(show_img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color['neg'])
    cv2.imshow('search', show_img)
    cv2.imshow('target', target_img)
