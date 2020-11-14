# -*- coding: utf-8 -*
from siamfcpp.utils import Registry

TRACK_BACKBONES = Registry('TRACK_BACKBONES')
VOS_BACKBONES = Registry('VOS_BACKBONES')

TASK_BACKBONES = dict(
    track=TRACK_BACKBONES,
    vos=VOS_BACKBONES,
)
