# Templates

## template_module

This template is used for constructing your module.

Name to be replaced:

- TemplateModule
- TEMPLATE_MODULES
- template_module

## Process to Go

Basically, you need to overwrite/complete the following parts:

- _template_module_impl/template_module_impl.py_
  - __TemplateModuleImplementation.default_hyper_params__
    - This defines the hyper-parameters (name/type).
  - __TemplateModuleImplementation.update_params__
    - This defines the behaviors to update your hyper-parameters with those given by configuration files
      - e.g. calculating the __score_offset__ (not given) with __score_size__ and __x_size__ (given by _.yaml_ file)
  - base class method defined in _template_module_base.py_

- build, _builder.py_
  - This defines the constructor's behavior.

## Misc

### Naming uder _template_module_impl_

By default, _template_module_impl/__init__.py_ will filter out files with filename such as "*_utils.py", "*_bak.py", etc. Please refer to _template_module_impl/__init__.py_ for detail.

### Inheritance

Inheritance of your implementation class is a little tricky because of __default_hyper_params__. An example is given in _template_module_impl/inherited_template_module_impl_.

## Pros of this design

With this "design pattern", the open-close principle (_"close to modification yet open to extension"_) is applied.

For example, in order to add an extra experiment with ShuffleNetV2 [arXiv:1807.11164](https://arxiv.org/abs/1807.11164),

- No tracked files need to be modified;
- Only a few number of files need to be added.

```Git
Untracked files:
  (use "git add <file>..." to include in what will be committed)

        experiments/siamfcpp/train/siamfcpp_shufflenetv2x0_5-trn.yaml
        experiments/siamfcpp/train/siamfcpp_shufflenetv2x1_0-trn.yaml
        tools/train_test-shufflenetv2x0_5.sh
        tools/train_test-shufflenetv2x1_0.sh
        videoanalyst/model/backbone/backbone_impl/shufflenet_v2.py
```
