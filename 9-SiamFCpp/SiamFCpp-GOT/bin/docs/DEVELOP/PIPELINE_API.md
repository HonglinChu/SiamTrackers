# PIPELINE_API

## Minimal Runnable Pipeline

Suppose that:

* you have your .yaml file in the string _exp_cfg_path_ and videoanalyst at the same level;
* you have a GPU with index 0,
then the following code segment will instantiate and configure a pipeline object which can be used immediately for your own application. It supports following APIs:
* _void init(im, state)_
* _state update(im)_

Example code:

```Python
import cv2
import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder

root_cfg.merge_from_file(exp_cfg_path)

# resolve config
task, task_cfg = specify_task(root_cfg)
task_cfg.freeze()

exp_cfg_path = osp.realpath(parsed_args.config)
# from IPython import embed;embed()
root_cfg.merge_from_file(exp_cfg_path)
logger.info("Load experiment configuration at: %s" % exp_cfg_path)


# build model
model = model_builder.build_model(task, task_cfg.model)
# build pipeline
pipeline = pipeline_builder.build('track', task_cfg.pipeline, model)
pipeline.set_device(torch.device("cuda:0"))
# register your template
im_template = cv2.imread("test file")
state_template = ...
pipeline.init(im_template)
# perform tracking based on your template
im_current = cv2.imread("test file")
state_current = pipeline.update(im_template)
```

## One-shot Detection Demo

Naturally, Siamese Tracker can be used for one-shot detection. A such API together with an runnable example are given at [demo/main/osdet_demo.py](../demo/main/osdet/osdet_demo.py).

```Bash
python3 demo/main/osdet_demo.py --shift_x=0.45 --shift_y=0.6
```

After running the above code, you should get [this result](../demo/resources/osdet_demo/osdet_demo.png).

By default, this example use a _.yaml_ configuration adapted from _siamfc-googlenet-vot_.
