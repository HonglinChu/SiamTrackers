# -*- coding: utf-8 -*
from yacs.config import CfgNode

from .tester import builder as tester_builder
from .trainer import builder as trainer_builder

TASK_ENGINE_BUILDERS = dict(
    tester=tester_builder,
    trainer=trainer_builder,
)


def build(task: str, cfg: CfgNode, engine_type: str, *args, **kwargs):
    """
    Builder function for trainer/tester
    engine_type: trainer or tester
    """
    if engine_type in TASK_ENGINE_BUILDERS:
        engine = TASK_ENGINE_BUILDERS[engine_type].build(
            task, cfg, *args, **kwargs)
        return engine
    else:
        raise ValueError("Invalid engine_type: %s" % engine_type)
