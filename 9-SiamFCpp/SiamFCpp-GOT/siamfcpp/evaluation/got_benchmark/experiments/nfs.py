from __future__ import absolute_import

import os

from .otb import ExperimentOTB
from ..datasets import NfS


class ExperimentNfS(ExperimentOTB):
    r"""Experiment pipeline and evaluation toolkit for NfS dataset.
    
    Args:
        root_dir (string): Root directory of NfS dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, fps=240,
                 result_dir='results', report_dir='reports'):
        self.dataset = NfS(root_dir, fps)
        self.result_dir = os.path.join(result_dir, 'NfS/%d' % fps)
        self.report_dir = os.path.join(report_dir, 'NfS/%d' % fps)
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
