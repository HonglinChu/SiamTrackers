# -*- coding: utf-8 -*

import numpy as np

import torch


def imarray_to_tensor(arr):
    r"""
    Transpose & convert from numpy.array to torch.Tensor
    :param arr: numpy.array, (H, W, C)
    :return: torch.Tensor, (1, C, H, W)
    """
    arr = np.ascontiguousarray(
        arr.transpose(2, 0, 1)[np.newaxis, ...], np.float32)
    return torch.from_numpy(arr)


def tensor_to_imarray(t):
    r"""
    Perform naive detach / cpu / numpy process and then transpose
    cast dtype to np.uint8
    :param t: torch.Tensor, (1, C, H, W)
    :return: numpy.array, (H, W, C)
    """
    arr = t.detach().cpu().numpy().astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.transpose(1, 2, 0)


def tensor_to_numpy(t):
    r"""
    Perform naive detach / cpu / numpy process.
    :param t: torch.Tensor, (N, C, H, W)
    :return: numpy.array, (N, C, H, W)
    """
    arr = t.detach().cpu().numpy()
    return arr
