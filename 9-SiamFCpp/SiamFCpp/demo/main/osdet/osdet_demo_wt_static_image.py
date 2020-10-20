# -*- coding: utf-8 -*
from paths import ROOT_PATH  # isort:skip
import argparse

import cv2
import numpy as np
from loguru import logger

import torch

import demo
from demo.resources.static_img_example.get_image import (bbox, im, im_x, im_z,
                                                         search_bbox,
                                                         target_bbox)
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.pipeline.utils.bbox import xywh2xyxy, xyxy2xywh
from videoanalyst.pipeline.utils.crop import get_subwindow
from videoanalyst.utils import complete_path_wt_root_in_cfg

color = dict(
    target=(0, 255, 0),
    pred=(0, 255, 255),
    template=(255, 0, 0),
    search=(255, 255, 0),
    border=(127, 127, 127),
)
font_size = 0.5
font_width = 1


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '--config',
        default="experiments/osdet/test/siamfcpp_googlenet-osdet.yaml",
        type=str,
        help='experiment configuration')
    parser.add_argument('--shift-x',
                        default=0.5,
                        type=float,
                        help='crop position x, [0, 1]')
    parser.add_argument('--shift-y',
                        default=0.5,
                        type=float,
                        help='crop position y, [0, 1]')
    parser.add_argument('--device',
                        default="cpu",
                        type=str,
                        help='torch.device')
    return parser


parser = make_parser()
parsed_args = parser.parse_args()

exp_cfg_path = parsed_args.config
root_cfg.merge_from_file(exp_cfg_path)
logger.info("Load experiment configuration at: %s" % exp_cfg_path)

# resolve config
root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
root_cfg = root_cfg.test
task, task_cfg = specify_task(root_cfg)
task_cfg.freeze()
# build model
model = model_builder.build(task, task_cfg.model)
# build pipeline
pipeline = pipeline_builder.build(task, task_cfg.pipeline, model)

dev = torch.device(parsed_args.device)
pipeline.set_device(dev)

if __name__ == "__main__":
    rect = xyxy2xywh(bbox)
    pipeline.init(im, rect)

    im_size = np.array((im.shape[1], im.shape[0]), dtype=np.float)
    crop_pos = np.array([parsed_args.shift_x, parsed_args.shift_y])
    im_shift = get_subwindow(im, im_size * crop_pos, im_size, im_size)

    rect_pred = pipeline.update(im_shift)
    bbox_pred = xywh2xyxy(rect_pred)
    bbox_pred = tuple(map(int, bbox_pred))

    im_ = im_shift
    cv2.rectangle(im_, bbox[:2], bbox[2:], color["target"])
    cv2.rectangle(im_, bbox_pred[:2], bbox_pred[2:], color["pred"])
    cv2.rectangle(im_, (0, 0), (im.shape[1] - 1, im.shape[0] - 1),
                  color["border"],
                  thickness=10)

    cv2.putText(im_, "original box", (20, 20), cv2.FONT_HERSHEY_COMPLEX,
                font_size, color["target"], font_width)
    cv2.putText(im_, "predicted box", (20, 40), cv2.FONT_HERSHEY_COMPLEX,
                font_size, color["pred"], font_width)
    cv2.putText(im_, "image border", (20, 60), cv2.FONT_HERSHEY_COMPLEX,
                font_size, color["border"], font_width)

    im_pred = im_
    cv2.imshow("im_pred", im_pred)
    cv2.waitKey(0)

    from IPython import embed
    embed()
