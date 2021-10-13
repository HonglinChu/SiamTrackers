# -*- coding: utf-8 -*
from copy import deepcopy
from typing import Dict

from yacs.config import CfgNode

from torch import nn

from siamfcpp.pipeline.pipeline_base import PipelineBase
from siamfcpp.utils import Registry

TRACK_TESTERS = Registry('TRACK_TESTERS')
VOS_TESTERS = Registry('VOS_TESTERS')

TASK_TESTERS = dict(
    track=TRACK_TESTERS,
    vos=VOS_TESTERS,
)

class TesterBase:
    r"""
    Tester base class (e.g. procedure defined for tracker / segmenter / etc.)
    Interface descriptions:
        init(im, state):
        update(im):
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict(
        exp_name="",
        exp_save="",
    )

    def __init__(self, pipeline: PipelineBase):
        """
        Parameters
        ----------
        pipeline : PipelineBase
            Pipeline to be tested
        """
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._pipeline = pipeline

    def get_hps(self) -> dict():
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: dict()) -> None:
        r"""
        Set hyper-parameters

        Arguments
        ---------
        hps: dict
            dict of hyper-parameters, the keys must in self.__hyper_params__
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]

    def set_pipeline(self, pipeline: PipelineBase):
        r"""Setter for underlying pipeline
        """
        self._pipeline = pipeline

    def update_params(self):
        r"""
        an interface for update params
        """
    def test(self) -> Dict:
        r"""
        an interface to start testing
        Returns
        -------
        Dict
            result object which need to contain the key "main_performance" in case of hyper-parameter optimization
        """
