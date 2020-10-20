# -*- coding: utf-8 -*
import time
from typing import List

# from PIL import Image
import cv2
import numpy as np

from siamfcpp.evaluation.got_benchmark.utils.viz import show_frame
from siamfcpp.pipeline.pipeline_base import PipelineBase


class PipelineTracker(object):
    def __init__(self,
                 name: str,
                 pipeline: PipelineBase,
                 is_deterministic: bool = True):
        """Helper tracker for comptability with 
        
        Parameters
        ----------
        name : str
            [description]
        pipeline : PipelineBase
            [description]
        is_deterministic : bool, optional
            [description], by default False
        """
        self.name = name
        self.is_deterministic = is_deterministic
        self.pipeline = pipeline

    def init(self, image: np.array, box):
        """Initialize pipeline tracker
        
        Parameters
        ----------
        image : np.array
            image of the first frame
        box : np.array or List
            tracking bbox on the first frame
            formate: (x, y, w, h)
        """
        self.pipeline.init(image, box)

    def update(self, image: np.array):
        """Perform tracking
        
        Parameters
        ----------
        image : np.array
            image of the current frame
        
        Returns
        -------
        np.array
            tracking bbox
            formate: (x, y, w, h)
        """
        return self.pipeline.update(image)

    def track(self, img_files: List, box, visualize: bool = False):
        """Perform tracking on a given video sequence
        
        Parameters
        ----------
        img_files : List
            list of image file paths of the sequence
        box : np.array or List
            box of the first frame
        visualize : bool, optional
            Visualize or not on each frame, by default False
        
        Returns
        -------
        [type]
            [description]
        """
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            # image = Image.open(img_file)
            # if not image.mode == 'RGB':
            #     image = image.convert('RGB')
            image = cv2.imread(img_file, cv2.IMREAD_COLOR)

            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                boxes[f, :] = self.update(image)
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, boxes[f, :])

        return boxes, times
