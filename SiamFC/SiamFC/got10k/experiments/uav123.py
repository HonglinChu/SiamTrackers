from __future__ import absolute_import

import os
import numpy as np

from .otb import ExperimentOTB
from ..datasets import UAV123
from ..utils.metrics import rect_iou, center_error


class ExperimentUAV123(ExperimentOTB):
    r"""Experiment pipeline and evaluation toolkit for UAV123 dataset.
    
    Args:
        root_dir (string): Root directory of UAV123 dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, version='UAV123',
                 result_dir='results', report_dir='reports'):
        assert version.upper() in ['UAV123', 'UAV20L']
        self.dataset = UAV123(root_dir, version)
        self.result_dir = os.path.join(result_dir, version.upper())
        self.report_dir = os.path.join(report_dir, version.upper())
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51

    def _calc_metrics(self, boxes, anno):
        valid = ~np.any(np.isnan(anno), axis=1)
        if len(valid) == 0:
            print('Warning: no valid annotations')
            return None, None
        else:
            ious = rect_iou(boxes[valid, :], anno[valid, :])
            center_errors = center_error(
                boxes[valid, :], anno[valid, :])
            return ious, center_errors
