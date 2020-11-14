# -*- coding: utf-8 -*-
"""
Procedure
* get basic scale: scale_temp_ / scale_curr_

* loop to get augmented cropping box
    * perform scale augmentation: scale_rand / scale_rand_temp
        * get augmented scale: scale_temp / scale_curr
        * get augmented size: s_temp / s_curr
    * perform random shift: dx / dy / dx_temp / dy_temp


    * get augmented object box on the original patch: box_crop_temp / box_crop_curr
    * get object boxes on the cropped patch: box_z / box_x
    * check validity of box

* perform cropping with _get_subwindow_tracking_: im_z, im_x

"""

import cv2
import numpy as np

from siamfcpp.pipeline.utils.bbox import cxywh2xyxy, xyxy2cxywh
from siamfcpp.pipeline.utils.crop import get_subwindow_tracking

_MAX_RETRY = 50


def crop_track_pair(
        im_temp,
        bbox_temp,
        im_curr,
        bbox_curr,
        config=None,
        avg_chans=None,
        rng=np.random,
        DEBUG=False,
        mask_tmp=None,
        mask_curr=None,
):
    context_amount = config["context_amount"]
    z_size = config["z_size"]
    x_size = config["x_size"]
    max_scale = config["max_scale"]
    max_shift = config["max_shift"]
    max_scale_temp = config["max_scale_temp"]
    max_shift_temp = config["max_shift_temp"]

    if avg_chans is None:
        avg_chans = np.mean(im_temp, axis=(0, 1))
    box_temp = xyxy2cxywh(bbox_temp)
    box_curr = xyxy2cxywh(bbox_curr)

    # crop size, st for tamplate & sc for current
    wt, ht = box_temp[2:]
    wt_ = wt + context_amount * (wt + ht)
    ht_ = ht + context_amount * (wt + ht)
    st_ = np.sqrt(wt_ * ht_)

    wc, hc = box_curr[2:]
    wc_ = wc + context_amount * (wc + hc)
    hc_ = hc + context_amount * (wc + hc)
    sc_ = np.sqrt(wc_ * hc_)

    assert (st_ > 0) and (
        sc_ > 0), "Invalid box: box_temp %s and box_curr %s" % (str(bbox_temp),
                                                                str(bbox_curr))

    scale_temp_ = z_size / st_
    scale_curr_ = z_size / sc_

    # loop to generate valid augmentation
    for i in range(_MAX_RETRY + 1):
        # random scale
        if i < _MAX_RETRY:
            s_max = 1 + max_scale
            s_min = 1 / s_max
            scale_rand = rng.uniform(s_min, s_max)
            s_max = 1 + max_scale_temp
            s_min = 1 / s_max
            scale_rand_temp = np.exp(rng.uniform(np.log(s_min), np.log(s_max)))
        else:
            scale_rand = scale_rand_temp = 1
            if DEBUG: print('not augmented')
        scale_curr = scale_curr_ / scale_rand
        scale_temp = scale_temp_ / scale_rand_temp
        s_curr = x_size / scale_curr
        s_temp = z_size / scale_temp

        # random shift
        if i < _MAX_RETRY:
            dx = rng.uniform(-max_shift, max_shift) * s_curr / 2
            dy = rng.uniform(-max_shift, max_shift) * s_curr / 2
            dx_temp = rng.uniform(-max_shift_temp, max_shift_temp) * s_temp / 2
            dy_temp = rng.uniform(-max_shift_temp, max_shift_temp) * s_temp / 2
        else:
            dx = dy = dx_temp = dy_temp = 0
            if DEBUG: print('not augmented')

        # calculate bbox for cropping
        box_crop_temp = np.concatenate([
            box_temp[:2] - np.array([dx_temp, dy_temp]),
            np.array([s_temp, s_temp])
        ])
        box_crop_curr = np.concatenate(
            [box_curr[:2] - np.array([dx, dy]),
             np.array([s_curr, s_curr])])

        # calculate new bbox
        box_z = np.array([(z_size - 1) / 2] * 2 + [0] * 2) + np.concatenate(
            [np.array([dx_temp, dy_temp]),
             np.array([wt, ht])]) * scale_temp
        box_x = np.array([(x_size - 1) / 2] * 2 + [0] * 2) + np.concatenate(
            [np.array([dx, dy]), np.array([wc, hc])]) * scale_curr
        bbox_z = cxywh2xyxy(box_z)
        bbox_x = cxywh2xyxy(box_x)

        # check validity of bbox
        if not (all([0 <= c <= z_size - 1 for c in bbox_z])
                and all([0 <= c <= x_size - 1 for c in bbox_x])):
            continue
        else:
            break

    # crop & resize via warpAffine
    mask_z = None
    mask_x = None
    if mask_tmp is not None:
        im_z, mask_z = get_subwindow_tracking(im_temp,
                                              box_crop_temp[:2],
                                              z_size,
                                              s_temp,
                                              avg_chans=avg_chans,
                                              mask=mask_tmp)
    else:
        im_z = get_subwindow_tracking(im_temp,
                                      box_crop_temp[:2],
                                      z_size,
                                      s_temp,
                                      avg_chans=avg_chans)

    if mask_curr is not None:
        im_x, mask_x = get_subwindow_tracking(im_curr,
                                              box_crop_curr[:2],
                                              x_size,
                                              s_curr,
                                              avg_chans=avg_chans,
                                              mask=mask_curr)
    else:
        im_x = get_subwindow_tracking(im_curr,
                                      box_crop_curr[:2],
                                      x_size,
                                      s_curr,
                                      avg_chans=avg_chans)

    return im_z, bbox_z, im_x, bbox_x, mask_z, mask_x


def crop_track_pair_for_sat(
        im_temp,
        bbox_temp,
        im_curr,
        bbox_curr,
        config=None,
        avg_chans=None,
        rng=np.random,
        DEBUG=False,
        mask_tmp=None,
        mask_curr=None,
):
    context_amount = config["context_amount"]
    z_size = config["track_z_size"]
    x_size = config["track_x_size"]
    max_scale = config["max_scale"]
    max_shift = config["max_shift"]
    max_scale_temp = config["max_scale_temp"]
    max_shift_temp = config["max_shift_temp"]

    if avg_chans is None:
        avg_chans = np.mean(im_temp, axis=(0, 1))
    box_temp = xyxy2cxywh(bbox_temp)
    box_curr = xyxy2cxywh(bbox_curr)

    # crop size, st for tamplate & sc for current
    wt, ht = box_temp[2:]
    wt_ = wt + context_amount * (wt + ht)
    ht_ = ht + context_amount * (wt + ht)
    st_ = np.sqrt(wt_ * ht_)

    wc, hc = box_curr[2:]
    wc_ = wc + context_amount * (wc + hc)
    hc_ = hc + context_amount * (wc + hc)
    sc_ = np.sqrt(wc_ * hc_)

    assert (st_ > 0) and (
        sc_ > 0), "Invalid box: box_temp %s and box_curr %s" % (str(bbox_temp),
                                                                str(bbox_curr))

    scale_temp_ = z_size / st_
    scale_curr_ = z_size / sc_

    # loop to generate valid augmentation
    for i in range(_MAX_RETRY + 1):
        # random scale
        if i < _MAX_RETRY:
            s_max = 1 + max_scale
            s_min = 1 / s_max
            scale_rand = rng.uniform(s_min, s_max)
            s_max = 1 + max_scale_temp
            s_min = 1 / s_max
            scale_rand_temp = np.exp(rng.uniform(np.log(s_min), np.log(s_max)))
        else:
            scale_rand = scale_rand_temp = 1
            if DEBUG: print('not augmented')
        scale_curr = scale_curr_ / scale_rand
        scale_temp = scale_temp_ / scale_rand_temp
        s_curr = x_size / scale_curr
        s_temp = z_size / scale_temp

        # random shift
        if i < _MAX_RETRY:
            dx = rng.uniform(-max_shift, max_shift) * s_curr / 2
            dy = rng.uniform(-max_shift, max_shift) * s_curr / 2
            dx_temp = rng.uniform(-max_shift_temp, max_shift_temp) * s_temp / 2
            dy_temp = rng.uniform(-max_shift_temp, max_shift_temp) * s_temp / 2
        else:
            dx = dy = dx_temp = dy_temp = 0
            if DEBUG: print('not augmented')

        # calculate bbox for cropping
        box_crop_temp = np.concatenate([
            box_temp[:2] - np.array([dx_temp, dy_temp]),
            np.array([s_temp, s_temp])
        ])
        box_crop_curr = np.concatenate(
            [box_curr[:2] - np.array([dx, dy]),
             np.array([s_curr, s_curr])])

        # calculate new bbox
        box_z = np.array([(z_size - 1) / 2] * 2 + [0] * 2) + np.concatenate(
            [np.array([dx_temp, dy_temp]),
             np.array([wt, ht])]) * scale_temp
        box_x = np.array([(x_size - 1) / 2] * 2 + [0] * 2) + np.concatenate(
            [np.array([dx, dy]), np.array([wc, hc])]) * scale_curr
        bbox_z = cxywh2xyxy(box_z)
        bbox_x = cxywh2xyxy(box_x)

        # check validity of bbox
        if not (all([0 <= c <= z_size - 1 for c in bbox_z])
                and all([0 <= c <= x_size - 1 for c in bbox_x])):
            continue
        else:
            break

    # sot track input z
    im_z = get_subwindow_tracking(im_temp,
                                  box_crop_temp[:2],
                                  z_size,
                                  s_temp,
                                  avg_chans=avg_chans,
                                  mask=None)
    # sot track input x
    im_x = get_subwindow_tracking(im_curr,
                                  box_crop_curr[:2],
                                  x_size,
                                  s_curr,
                                  avg_chans=avg_chans,
                                  mask=None)
    # global feature input

    global_fea_input_size = config["global_fea_input_size"]
    s_global = global_fea_input_size / scale_temp
    global_img, global_mask = get_subwindow_tracking(im_temp,
                                                     box_crop_temp[:2],
                                                     global_fea_input_size,
                                                     s_global,
                                                     avg_chans=avg_chans,
                                                     mask=mask_tmp)
    # saliency input
    seg_x_size = config["seg_x_size"]
    seg_x_resize = config["seg_x_resize"]
    s_seg_x = seg_x_size / scale_curr
    seg_img, seg_mask = get_subwindow_tracking(im_curr,
                                               box_crop_curr[:2],
                                               seg_x_resize,
                                               s_seg_x,
                                               avg_chans=avg_chans,
                                               mask=mask_curr)

    filtered_global_img = global_img * global_mask[:, :, np.newaxis]
    im_z = im_z.transpose(2, 0, 1)
    im_x = im_x.transpose(2, 0, 1)
    seg_img = seg_img.transpose(2, 0, 1)
    filtered_global_img = filtered_global_img.transpose(2, 0, 1)
    return dict(im_z=im_z,
                im_x=im_x,
                seg_img=seg_img,
                seg_mask=seg_mask,
                filtered_global_img=filtered_global_img)
