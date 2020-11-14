# -*- coding: utf-8 -*
from collections import Iterable
from typing import Tuple

import cv2
import numpy as np

from .bbox import cxywh2xyxy


def get_axis_aligned_bbox(region):
    r"""
    Get axis-aligned bbox (used to transform annotation in VOT benchmark)

    Arguments
    ---------
    region: list (nested)
        (1, 4, 2), 4 points of the rotated bbox

    Returns
    -------
    tuple
        axis-aligned bbox in format (cx, cy, w, h)
    """
    try:
        region = np.array([
            region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
            region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]
        ])
    except:
        region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
        np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return cx, cy, w, h


def get_subwindow_tracking(im,
                           pos,
                           model_sz,
                           original_sz,
                           avg_chans=(0, 0, 0),
                           mask=None):
    r"""
    Get subwindow via cv2.warpAffine

    Arguments
    ---------
    im: numpy.array
        original image, (H, W, C)
    pos: numpy.array
        subwindow position
    model_sz: int
        output size
    original_sz: int
        subwindow range on the original image
    avg_chans: tuple
        average values per channel
    mask: numpy.array
        mask, (H, W)


    Returns
    -------
    numpy.array
        image patch within _original_sz_ in _im_ and  resized to _model_sz_, padded by _avg_chans_
        (model_sz, model_sz, 3)
    """
    crop_cxywh = np.concatenate(
        [np.array(pos), np.array((original_sz, original_sz))], axis=-1)
    crop_xyxy = cxywh2xyxy(crop_cxywh)
    # warpAffine transform matrix
    M_13 = crop_xyxy[0]
    M_23 = crop_xyxy[1]
    M_11 = (crop_xyxy[2] - M_13) / (model_sz - 1)
    M_22 = (crop_xyxy[3] - M_23) / (model_sz - 1)
    mat2x3 = np.array([
        M_11,
        0,
        M_13,
        0,
        M_22,
        M_23,
    ]).reshape(2, 3)
    im_patch = cv2.warpAffine(im,
                              mat2x3, (model_sz, model_sz),
                              flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=tuple(map(int, avg_chans)))
    if mask is not None:
        mask_patch = cv2.warpAffine(mask,
                                    mat2x3, (model_sz, model_sz),
                                    flags=(cv2.INTER_NEAREST
                                           | cv2.WARP_INVERSE_MAP))
        return im_patch, mask_patch
    return im_patch


def get_crop(im,
             target_pos,
             target_sz,
             z_size,
             x_size=None,
             avg_chans=(0, 0, 0),
             context_amount=0.5,
             func_get_subwindow=get_subwindow_tracking,
             output_size=None,
             mask=None):
    r"""
    Get cropped patch for tracking

    Arguments
    ---------
    im: numpy.array
        input image
    target_pos: list-like or numpy.array
        position, (x, y)
    target_sz: list-like or numpy.array
        size, (w, h)
    z_size: int
        template patch size
    x_size: int
        search patch size, None in case of template (z_size == x_size)
    avg_chans: tuple
        channel average values, (B, G, R)
    context_amount: float
        context to be includede in template, set to 0.5 by convention
    func_get_subwindow: function object
        function used to perform cropping & resizing
    output_size: int
        the size of output if it is not None
    mask: numpy.array
        mask of the object

    Returns
    -------
        cropped & resized image, (output_size, output_size) if output_size provied,
        otherwise, (x_size, x_size, 3) if x_size provided, (z_size, z_size, 3) otherwise
    """
    wc = target_sz[0] + context_amount * sum(target_sz)
    hc = target_sz[1] + context_amount * sum(target_sz)
    s_crop = np.sqrt(wc * hc)
    scale = z_size / s_crop

    # im_pad = x_pad / scale
    if x_size is None:
        x_size = z_size
    s_crop = x_size / scale

    if output_size is None:
        output_size = x_size
    if mask is not None:
        im_crop, mask_crop = func_get_subwindow(im,
                                                target_pos,
                                                output_size,
                                                round(s_crop),
                                                avg_chans,
                                                mask=mask)
        return im_crop, mask_crop, scale
    else:
        im_crop = func_get_subwindow(im, target_pos, output_size, round(s_crop),
                                     avg_chans)
        return im_crop, scale


def _make_valid_int_pair(sz) -> Tuple[int, int]:
    """Cast size to int pair
    
    Parameters
    ----------
    sz : int or Iterable pair
        size
    
    Returns
    -------
    Tuple[int, int]
        int pair
    """
    if not isinstance(sz, Iterable):
        sz = (int(sz), ) * 2
    else:
        sz = sz[:2]
        sz = tuple(map(int, sz))
    return sz


# def get_subwindow(im, pos, model_sz, original_sz, avg_chans=(0, 0, 0)):
def get_subwindow(im: np.array, src_pos, src_sz, dst_sz,
                  avg_chans=(0, 0, 0)) -> np.array:
    """Get (arbitrary aspect ratio) subwindow via cv2.warpAffine

    Parameters
    ----------
    im: np.array
        image, (H, W, C)
    src_pos : [type]
        source position, (cx, cy)
    src_sz : [type]
        source size, (w, h)
    dst_sz : [type]
        destination size, (w, h)
    avg_chans : tuple, optional
        [description], by default (0, 0, 0)
    
    Returns
    -------
    np.array
        cropped image, (H, W, C)
    """

    src_sz = _make_valid_int_pair(src_sz)
    dst_sz = _make_valid_int_pair(dst_sz)

    crop_cxywh = np.concatenate([np.array(src_pos), np.array(src_sz)], axis=-1)
    crop_xyxy = cxywh2xyxy(crop_cxywh)
    # warpAffine transform matrix
    M_13 = crop_xyxy[0]
    M_23 = crop_xyxy[1]
    M_11 = (crop_xyxy[2] - M_13) / (dst_sz[0] - 1)
    M_22 = (crop_xyxy[3] - M_23) / (dst_sz[1] - 1)
    mat2x3 = np.array([
        M_11,
        0,
        M_13,
        0,
        M_22,
        M_23,
    ]).reshape(2, 3)
    im_patch = cv2.warpAffine(im,
                              mat2x3,
                              dst_sz,
                              flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=tuple(map(int, avg_chans)))
    return im_patch
