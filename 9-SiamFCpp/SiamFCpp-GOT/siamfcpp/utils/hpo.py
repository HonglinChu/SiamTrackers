# -*- coding: utf-8 -*
import os.path as osp
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd
from yacs.config import CfgNode

from .path import ensure_dir

_HPO_RANGE_POSTFIX = "_hpo_range"
_DELIMITER = ","


def parse_hp_path_and_range(hpo_cfg: CfgNode,
                            ) -> List[Tuple[List[str], Tuple[float, float]]]:
    """Parse hyper-parameter ranges from hp config via recursive method
    
    Parameters
    ----------
    hpo_cfg : CfgNode
        hp configuration
        same structure as test configuration
    
    Returns
    -------
    List[Tuple[List[str], Tuple[float, float]]]
        list of hyper-parameter's node name path & sampling range (Tuple)
        Tuple format:
            node_name_path: List[str]
            range: Tuple[lower_bound, upper_bound]
    """
    parsed_results = []
    for k, v in hpo_cfg.items():
        if k.endswith(_HPO_RANGE_POSTFIX) and (len(v) > 0):
            node_name = k[:-len(_HPO_RANGE_POSTFIX)]
            hpo_range = v
            parsed_results.append(([node_name], hpo_range))
        elif isinstance(v, dict):
            child_results = parse_hp_path_and_range(v)

            for idx in range(len(child_results)):
                node_name_path, hpo_range = child_results[idx]
                new_node_name_path = [k] + node_name_path
                child_results[idx] = new_node_name_path, hpo_range
            parsed_results.extend(child_results)
    return parsed_results


def get_cfg_value_wt_path(target_cfg, node_name_path):
    for n in node_name_path:
        target_cfg = target_cfg[n]
    return target_cfg


def set_cfg_value_wt_path(target_cfg, node_name_path, value):
    last_node_name = node_name_path[-1]
    for n in node_name_path[:-1]:
        target_cfg = target_cfg[n]
    target_cfg[last_node_name] = value


def sample_and_update_single_hp(target_cfg: CfgNode, node_name_path: List[str],
                                hpo_range) -> float:
    """Sample random value from uniform distribution & 
       update the node value at the given path to hyper-parameter node
    
    Parameters
    ----------
    target_cfg : CfgNode
        yacs CfgNode that to be sampled
    node_name_path : List[str]
        path to the hyper-parameter node whose value is to be updated
    hpo_range
        range of , behavior infered from data type
        - Tuple[float, float]: uniform distribution
        - Tuple[int, int]: random choice of in integer from given range
        - List[v1, v2, ...]: random choice of  an element from given list        
    
    Returns
    -------
    int / float
        the sampled random value for hyper-parameter
    """
    assert len(hpo_range) > 0, "empty hpo range in {}".format(node_name_path)
    if (len(hpo_range) > 2) or (len(hpo_range) == 1):
        random_hpo_value = np.random.choice(hpo_range)
    elif isinstance(hpo_range[0], int) and isinstance(hpo_range[1], int):
        hpo_lb, hpo_ub = hpo_range
        random_hpo_value = np.random.randint(hpo_lb, hpo_ub)
    else:
        hpo_lb, hpo_ub = float(hpo_range[0]), float(hpo_range[1])
        random_hpo_value = np.random.uniform(hpo_lb, hpo_ub)

    set_cfg_value_wt_path(target_cfg, node_name_path, random_hpo_value)
    return random_hpo_value


def sample_and_update_hps(target_cfg, hpo_schedules):
    sample_results = OrderedDict()
    for schedule in hpo_schedules:
        node_name_path, hpo_range = schedule
        hp_name = "/".join(node_name_path)
        random_value = sample_and_update_single_hp(target_cfg, node_name_path,
                                                   hpo_range)
        sample_results[hp_name] = random_value
    return sample_results


def merge_result_dict(result_dicts):
    """Merge results stored in dict
    
    Parameters
    ----------
    result_dicts : List[Dict[key, value]] or Dict[key, value]
        result dicts to be merged
    
    Returns
    -------
    Dict[key, List[value]]
        merge values into list
    """
    if not isinstance(result_dicts, list):
        result_dicts = [result_dicts]
    keys = list(result_dicts[0].keys())
    merged_result = {k: [] for k in keys}
    for result_dict in result_dicts:
        for k in keys:
            if isinstance(result_dict[k], list):
                merged_result[k].extend(result_dict[k])
            else:
                merged_result[k].append(result_dict[k])
    return merged_result


def dump_result_dict(csv_file: str, result_dict) -> pd.DataFrame:
    """Dump result_dict into csv_file
    
    Parameters
    ----------
    csv_file : str
        dump csv file path
    result_dict : List[Dict[key, value]] or Dict[key, value]
        result dict
    
    Returns
    -------
    pd.DataFrame
        Dumped DataFrame
    """
    if osp.exists(csv_file):
        df = pd.read_csv(csv_file, sep=_DELIMITER, index_col=0, squeeze=True)
    else:
        ensure_dir(osp.dirname(csv_file))
        df = pd.DataFrame(columns=list(result_dict.keys()))
    merged_results = merge_result_dict(result_dict)
    df_new = pd.DataFrame(merged_results)
    df = pd.concat([df, df_new], axis=0, ignore_index=True)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(csv_file,
              sep=_DELIMITER,
              header=True,
              index=True,
              index_label='hpo_id')

    return df
