# -*- coding: utf-8 -*-
import os

from yacs.config import CfgNode

from videoanalyst.utils import ensure_dir


def setup(cfg: CfgNode):
    """Setup for working directory
    
    Parameters
    ----------
    cfg : CfgNode
        task specific config
    """
    ensure_dir(cfg["exp_save"])
    cfg.auto = CfgNode()

    cfg.auto.exp_dir = os.path.join(cfg.exp_save, cfg.exp_name)
    ensure_dir(cfg.auto.exp_dir)

    cfg.auto.log_dir = os.path.join(cfg.auto.exp_dir, "logs")
    ensure_dir(cfg.auto.log_dir)

    cfg.auto.log_dir = os.path.join(cfg.auto.exp_dir, "snapshots")
    ensure_dir(cfg.auto.log_dir)

    cfg.auto.model_dir = os.path.join(cfg.auto.exp_dir, "datasets")
    ensure_dir(cfg.auto.model_dir)

    cfg.auto.model_dir = os.path.join(cfg.auto.exp_dir, "models")
    ensure_dir(cfg.auto.model_dir)
