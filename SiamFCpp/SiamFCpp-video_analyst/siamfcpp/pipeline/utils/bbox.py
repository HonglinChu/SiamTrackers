# -*- coding: utf-8 -*
r"""
* All coordinates are 0-indexed.
* Terminology for different formats:
  * bbox: (x1, y1, x2, y2)
  *  box: (cx, cy,  w,  h)
  * rect: (x1, y1,  w,  h)
* Width/Height defined as the number of columns/rows occuppied by the box
  * thus w = x1 - x0 + 1, and so for h
* Support extra dimensions (e.g. batch, anchor, etc)
  * Assume that the last dimension (axis=-1) is the box dimension
* For utilisation examples in details, please refer to the unit test at the bottom of the code.
  * Run ```python3 bbox_transform.py``` to launch unit test
"""

import itertools
import unittest

import numpy as np


# ============================== Formal conversion ============================== #
def clip_bbox(bbox, im_size):
    r"""
    Clip boxes to image boundaries, support batch-wise operation

    Arguments
    ---------
    bbox: numpy.array or list-like
        shape=(..., 4), format=(x1, y1, x2, y2)
    im_size: numpy.array or list-like
        shape=(..., 2), format=(w, h)

    Returns
    -------
    bbox_clipped: numpy.array
        shape=(..., 4), format=(x1, y1, x2, y2)
    """
    bbox = np.array(bbox)
    im_size = np.array(im_size)
    bbox[..., 0] = np.maximum(np.minimum(bbox[..., 0], im_size[..., 0] - 1), 0)
    bbox[..., 1] = np.maximum(np.minimum(bbox[..., 1], im_size[..., 1] - 1), 0)
    bbox[..., 2] = np.maximum(np.minimum(bbox[..., 2], im_size[..., 0] - 1), 0)
    bbox[..., 3] = np.maximum(np.minimum(bbox[..., 3], im_size[..., 1] - 1), 0)
    return bbox


def calc_IoU(bbox1, bbox2):
    r"""
    Calculate IoU, batch-wise

    Arguments
    ---------
    bbox1: numpy.array or list-like
        format=(x1, y1, x2, y2)
    bbox2: numpy.array or list-like
        format=(x1, y1, x2, y2)

    Returns
    -------
    float
        Intersection over Union
    """
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)
    area1 = np.abs(bbox1[..., 2] - bbox1[..., 0] +
                   1) * np.abs(bbox1[..., 3] - bbox1[..., 1] + 1)
    area2 = np.abs(bbox2[..., 2] - bbox2[..., 0] +
                   1) * np.abs(bbox2[..., 3] - bbox2[..., 1] + 1)

    iw = np.minimum(bbox1[..., 2], bbox2[..., 2]) - np.maximum(
        bbox1[..., 0], bbox2[..., 0]) + 1
    ih = np.minimum(bbox1[..., 3], bbox2[..., 3]) - np.maximum(
        bbox1[..., 1], bbox2[..., 1]) + 1

    inter = np.maximum(iw, 0) * np.maximum(ih, 0)

    union = area1 + area2 - inter
    iou = np.maximum(inter / union, 0)

    return iou


# ============================== Formal conversion ============================== #


def cxywh2xywh(box):
    box = np.array(box, dtype=np.float32)
    return np.concatenate([
        box[..., [0]] - (box[..., [2]] - 1) / 2, box[..., [1]] -
        (box[..., [3]] - 1) / 2, box[..., [2]], box[..., [3]]
    ],
                          axis=-1)


def xywh2cxywh(rect):
    rect = np.array(rect, dtype=np.float32)
    return np.concatenate([
        rect[..., [0]] + (rect[..., [2]] - 1) / 2, rect[..., [1]] +
        (rect[..., [3]] - 1) / 2, rect[..., [2]], rect[..., [3]]
    ],
                          axis=-1)


def cxywh2xyxy(box):
    box = np.array(box, dtype=np.float32)
    return np.concatenate([
        box[..., [0]] - (box[..., [2]] - 1) / 2, box[..., [1]] -
        (box[..., [3]] - 1) / 2, box[..., [0]] +
        (box[..., [2]] - 1) / 2, box[..., [1]] + (box[..., [3]] - 1) / 2
    ],
                          axis=-1)


def xyxy2xywh(bbox):
    bbox = np.array(bbox, dtype=np.float32)
    return np.concatenate([
        bbox[..., [0]], bbox[..., [1]], bbox[..., [2]] - bbox[..., [0]] + 1,
        bbox[..., [3]] - bbox[..., [1]] + 1
    ],
                          axis=-1)


def xywh2xyxy(rect):
    rect = np.array(rect, dtype=np.float32)
    return np.concatenate([
        rect[..., [0]], rect[..., [1]], rect[..., [2]] + rect[..., [0]] - 1,
        rect[..., [3]] + rect[..., [1]] - 1
    ],
                          axis=-1)


def xyxy2cxywh(bbox):
    bbox = np.array(bbox, dtype=np.float32)
    return np.concatenate([(bbox[..., [0]] + bbox[..., [2]]) / 2,
                           (bbox[..., [1]] + bbox[..., [3]]) / 2,
                           bbox[..., [2]] - bbox[..., [0]] + 1,
                           bbox[..., [3]] - bbox[..., [1]] + 1],
                          axis=-1)


# ============================== Unit test part ============================== #

clip_bbox_test_cases = [
    dict(
        bbox=(10, 10, 20, 20),
        im_size=(30, 30),
        bbox_clipped=(10, 10, 20, 20),
    ),
    dict(
        bbox=(10, 10, 20, 20),
        im_size=(15, 15),
        bbox_clipped=(10, 10, 14, 14),
    ),
    dict(
        bbox=(-5, -5, 20, 20),
        im_size=(30, 30),
        bbox_clipped=(0, 0, 20, 20),
    ),
    dict(
        bbox=(-10, -5, 20, 30),
        im_size=(10, 5),
        bbox_clipped=(0, 0, 9, 4),
    ),
]

bbox_transform_test_cases = [
    dict(
        xyxy=(10., 20., 50., 40.),
        xywh=(10., 20., 41., 21.),
        cxywh=(30., 30., 41., 21.),
    ),
    dict(
        xyxy=(40., 40., 60., 60.),
        xywh=(40., 40., 21., 21.),
        cxywh=(50., 50., 21., 21.),
    ),
    dict(
        xyxy=(40., 60., 45., 75.),
        xywh=(40., 60., 6., 16.),
        cxywh=(42.5, 67.5, 6., 16.),
    ),
    dict(
        xyxy=(40., 60., 40., 60.),
        xywh=(40., 60., 1., 1.),
        cxywh=(40, 60., 1., 1.),
    ),
]

formats = ['xyxy', 'xywh', 'cxywh']
format_cvt_pairs = [(src, dst)
                    for (src, dst) in itertools.product(formats, formats)
                    if src != dst]
var_dict = locals()
# func_cvt_list = [var_dict["%s2%s" % (src_fmt, dst_fmt)] for (src_fmt, dst_fmt) in format_cvt_pairs]


class TestBboxTransform(unittest.TestCase):
    def test_clip_bbox(self):
        print('test for clip_bbox')
        for case in clip_bbox_test_cases:
            case_input = case['bbox'], case['im_size']
            case_answer = case['bbox_clipped']
            case_output = clip_bbox(*case_input)
            for out, ans in zip(case_output, case_answer):
                self.assertEqual(
                    out, ans, "test failed in clip_bbox\n"
                    "%s -> %s, expected %s" %
                    (case_input, case_output, case_answer))

    def test_bbox_transform(self):
        for src_fmt, dst_fmt in format_cvt_pairs:
            func_name = "%s2%s" % (src_fmt, dst_fmt)
            func_cvt = var_dict[func_name]

            print('test for %s' % func_name)
            for case in bbox_transform_test_cases:
                case_input = case[src_fmt]
                case_answer = case[dst_fmt]
                case_output = func_cvt(case_input)
                for out, ans in zip(case_output, case_answer):
                    self.assertEqual(out, ans,
                                     "test failed in %s\n"%(func_name) + \
                                     "%s -> %s, expected %s"%(case_input, case_output, case_answer))

            print('batch test for %s' % func_name)
            # for case in bbox_transform_test_cases:
            case_inputs = np.array(
                [case[src_fmt] for case in bbox_transform_test_cases])
            case_answers = np.array(
                [case[dst_fmt] for case in bbox_transform_test_cases])
            case_outputs = func_cvt(case_inputs)
            for out, ans in zip(case_outputs.reshape(-1),
                                case_answers.reshape(-1)):
                self.assertEqual(out, ans, "batch test failed in %s\n" % (func_name) + \
                                 "%s -> %s, expected %s" % (case_inputs, case_outputs, case_answers))
            for dim_out, dim_ans in zip(case_outputs.shape, case_answers.shape):
                self.assertEqual(dim_out, dim_ans,
                                 "batch test failed in %s\n" % (func_name) + \
                                 "shapes donnot match: output %s, expected %s" % (case_outputs.shape, case_answers.shape))


if __name__ == '__main__':
    unittest.main()
