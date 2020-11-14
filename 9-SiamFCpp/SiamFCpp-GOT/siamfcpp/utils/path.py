import os
import os.path as osp

from yacs.config import CfgNode


def ensure_dir(dir_path: str):
    r"""
    Ensure the existence of path (i.e. mkdir -p)
    Arguments
    ---------
    dir_path: str
        path to be ensured
    """
    if osp.exists(dir_path):
        return
    else:
        os.makedirs(dir_path)


def complete_path_wt_root_in_cfg(
        cfg: CfgNode,
        root_dir: str,
):
    r"""
    Arguments
    ---------
    cfg: CfgNode
        yacs configuration object to be completed
    root_dir: str
        root path
    """
    if isinstance(cfg, CfgNode):
        for k in cfg: #嵌套
            cfg[k] = complete_path_wt_root_in_cfg(cfg[k], root_dir)
    elif isinstance(cfg, str) and len(cfg) > 0:
        realpath = osp.join(root_dir, cfg)
        if osp.exists(realpath):
            cfg = realpath
            # print(realpath)

    return cfg
