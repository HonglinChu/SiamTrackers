# -*- coding: utf-8 -*-
from yacs.config import CfgNode

from siamfcpp.data.builder import get_config as get_data_cfg
from siamfcpp.engine.tester.builder import get_config as get_tester_cfg
from siamfcpp.engine.trainer.builder import get_config as get_trainer_cfg
from siamfcpp.model.builder import get_config as get_model_cfg
from siamfcpp.optim.builder import get_config as get_optim_cfg
from siamfcpp.pipeline.builder import get_config as get_pipeline_cfg

cfg = CfgNode()  # root_cfg
task_list = ["track","vos"]
default_str = "unknown"
cfg["task_name"] = default_str


# default configuration for test
cfg["test"] = CfgNode()
test_cfg = cfg["test"]
for task in task_list:
    test_cfg[task] = CfgNode()
    test_cfg[task]["exp_name"] = default_str #算法名字
    test_cfg[task]["exp_save"] = default_str #模型保存路径

    if task == "track":
        test_cfg[task]["model"] = get_model_cfg(task_list)[task]

    test_cfg[task]["pipeline"] = get_pipeline_cfg(task_list)[task]
    test_cfg[task]["tester"] = get_tester_cfg(task_list)[task]
    test_cfg[task]["data"] = get_data_cfg(task_list)[task]#多线程加载数据batch size; works;

# default configuration for train
cfg["train"] = CfgNode()
train_cfg = cfg["train"]
for task in task_list:
    train_cfg[task] = CfgNode()
    train_cfg[task]["exp_name"] = default_str
    train_cfg[task]["exp_save"] = default_str
    train_cfg[task]["num_processes"] = 1  #number of devices
    train_cfg[task]["device"] = "cuda"  #[cuda|cpu]

    if task == "track":
        train_cfg[task]["model"] = get_model_cfg(task_list)[task]

    train_cfg[task]["pipeline"] = get_pipeline_cfg(task_list)[task]
    train_cfg[task]["tester"] = get_tester_cfg(task_list)[task]
    train_cfg[task]["data"] = get_data_cfg(task_list)[task]
    train_cfg[task]["optim"] = get_optim_cfg()
    train_cfg[task]["trainer"] = get_trainer_cfg(task_list)[task]


def specify_task(cfg: CfgNode) -> (str, CfgNode):
    r"""
    get task's short name from config, and specify task config

    Args:
        cfg (CfgNode): config 
    Returns:
        short task name, task-specified cfg
    """
    for task in task_list:
        if cfg[task]['exp_name'] != default_str:
            return task, cfg[task]
    assert False, "unknown task!"
