# -*- coding: utf-8 -*
from siamfcpp.utils import Registry

TRACK_LOSSES = Registry('TRACK_LOSSES')
VOS_LOSSES = Registry('VOS_LOSSES')

TASK_LOSSES = dict(
    track=TRACK_LOSSES,
    vos=VOS_LOSSES,
)
