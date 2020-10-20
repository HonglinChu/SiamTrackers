# -*- coding: utf-8 -*-
from typing import Dict


def index_data(data: Dict, idx: int):
    r"""
    Arguments
    data: Dict
        data to be indexed
    idx: int
        index used for indexing in data's first dimension
    """
    ret = dict()
    for k in data:
        ret[k] = data[k][idx]

    return ret
