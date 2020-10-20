# -*- coding: utf-8 -*
from paths import ROOT_PATH  # isort:skip
import argparse

import cv2
import numpy as np
from loguru import logger

import torch

import demo
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.data.dataset import builder as dataset_buidler
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.pipeline.utils.bbox import xywh2xyxy, xyxy2xywh
from videoanalyst.pipeline.utils.crop import get_subwindow
from videoanalyst.utils import complete_path_wt_root_in_cfg
from videoanalyst.utils.image import load_image

color = dict(
    target=(0, 255, 0),
    pred=(0, 255, 255),
    template=(255, 0, 0),
    search=(255, 255, 0),
    border=(127, 127, 127),
)
bbox_thickness = 2
font_size = 0.5
font_width = 1
resize_factor = 0.3


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '--config',
        default="experiments/osdet/test/siamfcpp_googlenet-osdet.yaml",
        type=str,
        help='experiment configuration')
    parser.add_argument('--sequence-index',
                        default=0,
                        type=int,
                        help='template frame index')
    parser.add_argument('--template-frame',
                        default=0,
                        type=int,
                        help='template frame index')
    parser.add_argument('--search-frame',
                        default=1,
                        type=int,
                        help='search frame index')
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
test_cfg = root_cfg.test
task, task_cfg = specify_task(test_cfg)
task_cfg.freeze()
# build model
model = model_builder.build(task, task_cfg.model)
# build pipeline
pipeline = pipeline_builder.build(task, task_cfg.pipeline, model)
# build dataset
datasets = dataset_buidler.build(
    task, root_cfg.train.track.data.sampler.submodules.dataset)
dataset = datasets[0]

dev = torch.device(parsed_args.device)
pipeline.set_device(dev)

if __name__ == "__main__":

    seq = dataset[parsed_args.sequence_index]

    template_frame = {k: seq[k][parsed_args.template_frame] for k in seq}
    template_frame['image'] = load_image(template_frame['image'])

    search_frame = {k: seq[k][parsed_args.search_frame] for k in seq}
    search_frame['image'] = load_image(search_frame['image'])

    im = template_frame['image']
    bbox = template_frame['anno']
    rect = xyxy2xywh(bbox)
    pipeline.init(im, rect)

    bbox = tuple(map(int, bbox))
    cv2.rectangle(im,
                  bbox[:2],
                  bbox[2:],
                  color["target"],
                  thickness=bbox_thickness)
    cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1),
                  color["border"],
                  thickness=10)

    im = cv2.resize(im, (0, 0), fx=resize_factor, fy=resize_factor)
    im = cv2.putText(im, "template frame", (20, 20), cv2.FONT_HERSHEY_COMPLEX,
                     font_size, color["target"], font_width)
    # cv2.imshow("im", im)

    im_search = search_frame['image']
    bbox_gt = search_frame['anno']
    rect_gt = xyxy2xywh(bbox_gt)
    rect_pred = pipeline.update(im_search)
    bbox_pred = xywh2xyxy(rect_pred)

    bbox_gt = tuple(map(int, bbox_gt))
    bbox_pred = tuple(map(int, bbox_pred))

    im_ = im_search
    cv2.rectangle(im_,
                  bbox_gt[:2],
                  bbox_gt[2:],
                  color["target"],
                  thickness=bbox_thickness)
    cv2.rectangle(im_,
                  bbox_pred[:2],
                  bbox_pred[2:],
                  color["pred"],
                  thickness=bbox_thickness)
    cv2.rectangle(im_, (0, 0), (im_.shape[1] - 1, im_.shape[0] - 1),
                  color["border"],
                  thickness=10)

    im_ = cv2.resize(im_, (0, 0), fx=resize_factor, fy=resize_factor)

    im_ = cv2.putText(im_, "ground-truth box", (20, 20),
                      cv2.FONT_HERSHEY_COMPLEX, font_size, color["target"],
                      font_width)
    im_ = cv2.putText(im_, "predicted box", (20, 40), cv2.FONT_HERSHEY_COMPLEX,
                      font_size, color["pred"], font_width)
    im_ = cv2.putText(im_, "image border", (20, 60), cv2.FONT_HERSHEY_COMPLEX,
                      font_size, color["border"], font_width)
    im_pred = im_
    # cv2.imshow("im_pred", im_pred)

    im_concat = cv2.vconcat([im, im_pred])
    cv2.imshow("im_concat", im_concat)
    cv2.waitKey(0)

    from IPython import embed
    embed()
