# -*- coding: utf-8 -*
from siamfcpp.utils import Registry

TRACK_TASKMODELS = Registry('TRACK_TASKMODELS')
VOS_TASKMODELS = Registry('VOS_TASKMODELS')

TASK_TASKMODELS = dict(
    track=TRACK_TASKMODELS,
    vos=VOS_TASKMODELS,
)
