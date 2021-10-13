# -*- coding: utf-8 -*
import collections.abc
import re
from typing import Dict

import numpy as np

import torch
import torch.distributed as dist
from torch import nn

np_str_obj_array_pattern = re.compile(r"[aO]")
default_collate_err_msg_format = (
    "default_collator: inputs must contain numpy arrays, numbers, "
    "Unicode strings, bytes, dicts or lists; found {}")


def move_data_to_device(data_dict: Dict, dev: torch.device):
    for k in data_dict:
        data_dict[k] = data_dict[k].to(dev)

    return data_dict


def unwrap_model(model):
    r""" unwrap nn.dataparallel wrapped module for model serialization """
    return model.module if isinstance(
        model,
        (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def convert_numpy_to_tensor(raw_data):
    r"""
    convert numpy array dict or list to torch.Tensor
    """
    elem_type = type(raw_data)
    if (elem_type.__module__ == "numpy" and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"):
        return torch.from_numpy(raw_data).float()
    elif isinstance(raw_data, collections.abc.Mapping):
        data = {key: convert_numpy_to_tensor(raw_data[key]) for key in raw_data}
        if 'image' in data:
            data['image'] = data['image'].permute(2, 0, 1)
        return data
    elif isinstance(raw_data, collections.abc.Sequence):
        return [convert_numpy_to_tensor(data) for data in raw_data]
    else:
        return raw_data


def convert_tensor_to_numpy(raw_data):
    r"""
    convert numpy array dict or list to torch.Tensor
    """
    if isinstance(raw_data, torch.Tensor):
        return raw_data.cpu().numpy()
    elif isinstance(raw_data, collections.abc.Mapping):
        data = {key: convert_tensor_to_numpy(raw_data[key]) for key in raw_data}
        if 'image' in data:
            data['image'] = data['image'].transpose(1, 2, 0).astype(np.uint8)
        return data
    elif isinstance(raw_data, collections.abc.Sequence):
        return [convert_tensor_to_numpy(data) for data in raw_data]


def average_gradients(model):
    r""" Gradient averaging. 
         from https://pytorch.org/tutorials/intermediate/dist_tuto.html
         to be called after _loss.backward()_ and before _optimizer.step()_
    """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
