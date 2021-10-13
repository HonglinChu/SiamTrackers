from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from torch import nn


def group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = group_checkpoint_keys(keys)
    msg = "Some model parameters are not in the checkpoint:\n"
    for k, v in groups.items():
        msg += "{}:{}\n".format(k, v)
    return msg


def named_modules_with_dup(model: nn.Module,
                           prefix: str = "") -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():  # pyre-ignore
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from named_modules_with_dup(module, submodule_prefix)


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = group_checkpoint_keys(keys)
    msg = "The checkpoint contains parameters not used by the model:\n"
    for k, v in groups.items():
        msg += "{}:{}\n".format(k, v)
    return msg


def filter_reused_missing_keys(model: nn.Module, keys: List[str]) -> List[str]:
    """
    Filter "missing keys" to not include keys that have been loaded with another name.
    """
    keyset = set(keys)
    param_to_names = defaultdict(set)  # param -> names that points to it
    for module_prefix, module in named_modules_with_dup(model):
        for name, param in list(module.named_parameters(recurse=False)) + list(
                module.named_buffers(recurse=False)  # pyre-ignore
        ):
            full_name = (module_prefix + "." if module_prefix else "") + name
            param_to_names[param].add(full_name)
    for names in param_to_names.values():
        # if one name appears missing but its alias exists, then this
        # name is not considered missing
        if any(n in keyset for n in names) and not all(n in keyset
                                                       for n in names):
            [keyset.remove(n) for n in names if n in keyset]
    return list(keyset)
