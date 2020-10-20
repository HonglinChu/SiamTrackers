# -*- coding: utf-8 -*
from siamfcpp.utils import Registry

TRACK_HEADS = Registry('TRACK_HEADS')
VOS_HEADS = Registry('VOS_HEADS')
TASK_HEADS = dict(
    track=TRACK_HEADS,
    vos=VOS_HEADS,
)
