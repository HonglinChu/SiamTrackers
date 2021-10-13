# -*- coding: utf-8 -*
from copy import deepcopy

from loguru import logger

import torch
import torch.nn as nn

from ...module_base import ModuleBase
from ..backbone_base import TRACK_BACKBONES, VOS_BACKBONES


@TRACK_BACKBONES.register
class ShuffleNetV2_x1_0(ModuleBase):
    default_hyper_params = dict(
        pretrain_model_path="",
        crop_pad=4,
        head_width=256,
    )

    def __init__(self):
        super(ShuffleNetV2_x1_0, self).__init__()
        self._hyper_params = deepcopy(self.default_hyper_params)

    def update_params(self):
        arch = "shufflenetv2_x1.0"
        kwargs = self._hyper_params
        # build module
        self._model = _shufflenetv2(arch,
                                    False,
                                    True, [4, 8, 4], [24, 116, 232, 464, 1024],
                                    fused_channls=[116, 232, 464],
                                    **kwargs)
        super().update_params()

    def forward(self, x):
        x = self._model(x)

        return x


@TRACK_BACKBONES.register
class ShuffleNetV2_x0_5(ModuleBase):
    default_hyper_params = dict(
        pretrain_model_path="",
        crop_pad=4,
        head_width=256,
    )

    def __init__(self):
        super(ShuffleNetV2_x0_5, self).__init__()
        self._hyper_params = deepcopy(self.default_hyper_params)

    def update_params(self):
        arch = "shufflenetv2_x0.5"
        kwargs = self._hyper_params
        # build module
        self._model = _shufflenetv2('shufflenetv2_x0.5',
                                    False,
                                    True, [4, 8, 4], [24, 48, 96, 192, 1024],
                                    fused_channls=[48, 96, 192],
                                    **kwargs)

        model_file = self._hyper_params["pretrain_model_path"]
        if model_file != "":
            state_dict = torch.load(model_file,
                                    map_location=torch.device("cpu"))
            self._model.load_state_dict(state_dict, strict=False)
            logger.info("Load pretrained ShuffleNet parameters from: %s" %
                        model_file)
            logger.info("Check md5sum of pretrained ShuffleNet parameters: %s" %
                        md5sum(model_file))

    def forward(self, x):
        x = self._model(x)

        return x


# def get_basemodel():
#     arch = "shufflenetv2_x1.0"
#     model = _shufflenetv2(arch, False, True,
#                          [4, 8, 4], [24, 116, 232, 464, 1024],)
#     state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
#     # for k in state_dict:
#     #     print(k)
#     model.load_state_dict(state_dict, strict=False)
#     return model

# __all__ = [
#     'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
#     'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
# ]

# Original download link
# model_urls = {
#     'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
#     'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
#     'shufflenetv2_x1.5': None,
#     'shufflenetv2_x2.0': None,
# }


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, false_stride=False):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride
        self.false_stride = false_stride

        branch_features = oup // 2
        # assert (self.stride != 1) or (inp == branch_features << 1)

        real_stride = self.stride if not self.false_stride else 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                # self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                self.depthwise_conv(inp,
                                    inp,
                                    kernel_size=3,
                                    stride=real_stride,
                                    padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp,
                          branch_features,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            self.depthwise_conv(branch_features,
                                branch_features,
                                kernel_size=3,
                                stride=real_stride,
                                padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i,
                         o,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias,
                         groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats,
                 stages_out_channels,
                 num_classes=1000,
                 inverted_residual=InvertedResidual,
                 crop_pad=4,
                 fused_channls=[116, 232, 464],
                 head_width=256,
                 **kwargs):
        super(ShuffleNetV2, self).__init__()
        self.crop_pad = crop_pad

        if len(stages_repeats) != 3:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError(
                'expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            false_stride = (name != "stage2")
            seq = [
                inverted_residual(input_channels,
                                  output_channels,
                                  2,
                                  false_stride=false_stride)
            ]
            # seq = [inverted_residual(input_channels, output_channels, stride)]
            for i in range(repeats - 1):
                seq.append(
                    inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(output_channels),
        #     nn.ReLU(inplace=True),
        # )

        # self.fc = nn.Linear(output_channels, num_classes)
        # channel_reduce
        self.channel_reduce = nn.Sequential(
            nn.Conv2d((sum(fused_channls)), head_width, 1, 1, 0, bias=False),
            nn.BatchNorm2d(head_width),
            nn.ReLU(inplace=True),
        )

    def _forward_impl(self, x):
        # transform_input
        x = x / 255
        x_ch0 = (torch.unsqueeze(x[:, 2], 1) - 0.485) / 0.229  # R
        x_ch1 = (torch.unsqueeze(x[:, 1], 1) - 0.456) / 0.224  # G
        x_ch2 = (torch.unsqueeze(x[:, 0], 1) - 0.406) / 0.225  # B
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # See note [TorchScript super()]
        xs = []
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        xs.append(x)
        x = self.stage3(x)
        xs.append(x)
        x = self.stage4(x)
        xs.append(x)
        # x = self.conv5(x)
        # xs.append(x)
        x = torch.cat(xs, 1)

        crop_pad = self.crop_pad
        x = x[:, :, crop_pad:-crop_pad, crop_pad:-crop_pad]

        x = self.channel_reduce(x)

        # x = x.mean([2, 3])  # globalpool
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(
                'pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)

    return model


def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress, [4, 8, 4],
                         [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress, [4, 8, 4],
                         [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress, [4, 8, 4],
                         [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress, [4, 8, 4],
                         [24, 244, 488, 976, 2048], **kwargs)


if __name__ == "__main__":
    model = get_basemodel()
    inputs = torch.Tensor(1, 3, 127, 127)
    outputs = model(inputs)
    print("Output shape", outputs.shape)
