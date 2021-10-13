# -*- coding: utf-8 -*-
import glob
import os.path as osp

import cv2
import numpy as np
from loguru import logger
from PIL import Image

_RETRY_NUM = 10


def load_image(img_file: str) -> np.array:
    """Image loader used by data module (e.g. image sampler)
    
    Parameters
    ----------
    img_file: str
        path to image file
    Returns
    -------
    np.array
        loaded image
    
    Raises
    ------
    FileExistsError
        invalid image file
    RuntimeError
        unloadable image file
    """
    if not osp.isfile(img_file):
        logger.info("Image file %s does not exist." % img_file)
    # read with OpenCV
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        # retrying
        for ith in range(_RETRY_NUM):
            logger.info("cv2 retrying (counter: %d) to load image file: %s" %
                        (ith + 1, img_file))
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if img is not None:
                break
    # read with PIL
    if img is None:
        logger.info("PIL used in loading image file: %s" % img_file)
        img = Image.open(img_file)
        img = np.array(img)
        img = img[:, :, [2, 1, 0]]  # RGB -> BGR
    if img is None:
        logger.info("Fail to load Image file %s" % img_file)

    return img


class ImageFileVideoStream:
    r"""Adaptor class to be compatible with VideoStream object
        Accept seperate video frames
    """
    def __init__(self, video_dir, init_counter=0):
        self._state = dict()
        self._state["video_dir"] = video_dir
        self._state["frame_files"] = sorted(glob.glob(video_dir))
        self._state["video_length"] = len(self._state["frame_files"])
        self._state["counter"] = init_counter  # 0

    def isOpened(self, ):
        return (self._state["counter"] < self._state["video_length"])

    def read(self, ):
        frame_idx = self._state["counter"]
        frame_file = self._state["frame_files"][frame_idx]
        frame_img = load_image(frame_file)
        self._state["counter"] += 1
        return frame_idx, frame_img

    def release(self, ):
        self._state["counter"] = 0


class ImageFileVideoWriter:
    r"""Adaptor class to be compatible with VideoWriter object
        Accept seperate video frames
    """
    def __init__(self, video_dir):
        self._state = dict()
        self._state["video_dir"] = video_dir
        self._state["counter"] = 0
        logger.info("Frame results will be dumped at: {}".format(video_dir))

    def write(self, im):
        frame_idx = self._state["counter"]
        frame_file = osp.join(self._state["video_dir"],
                              "{:06d}.jpg".format(frame_idx))
        cv2.imwrite(frame_file, im)
        self._state["counter"] += 1

    def release(self, ):
        self._state["counter"] = 0
