# -*- coding: utf-8 -*
from copy import deepcopy

from torch import nn

from siamfcpp.model.module_base import ModuleBase
from siamfcpp.utils import Registry

TRACK_PIPELINES = Registry('TRACK_PIPELINES')
VOS_PIPELINES = Registry('VOS_PIPELINES')
PIPELINES = dict(track=TRACK_PIPELINES, vos=VOS_PIPELINES)


class PipelineBase:
    r"""
    Pipeline base class (e.g. procedure defined for tracker / segmentor / etc.)
    Interface descriptions:
        init(im, state):
        update(im):
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict()

    def __init__(self, model: ModuleBase):
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._model = model

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

    def update_params(self):
        r"""
        an interface for update params
        """
    def init(self, im, state):
        r"""
        an interface for pipeline initialization (e.g. template feature extraction)
        default implementation: record initial state & do nothing

        Arguments
        ---------
        im: numpy.array
            initial frame image
        state:
            initial state (usually depending on task) (e.g. bbox for track / mask for vos)
        """
        self._state['state'] = state

    def update(self, im):
        r"""
        an interface for pipeline update
            (e.g. output target bbox for current frame given the frame and previous target bbox)
        default implementation: return previous target state (initial state)

        Arguments
        ---------
        im: numpy.array
            current frame

        Returns
        -------
        state
            predicted sstate (usually depending on task) (e.g. bbox for track / mask for vos)
        """
        state = self._state['state']
        return state
