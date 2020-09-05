# Copyright 2020 Toyota Research Institute.  All rights reserved.
from torch import nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops

# This class is adapted from "BackboneWithFPN" in torchvision.models.detection.backbone_utils

class ResNetnew(nn.Module):


    def __init__(self, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3, out_channels = 256):
        super().__init__()
        # Get ResNet
        backbone = resnet.__dict__['resnet50'](pretrained=True, norm_layer=norm_layer)
        # select layers that wont be frozen
        assert 0 <= trainable_layers <= 5
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        # freeze layers only if pretrained backbone is used
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

        in_channels_stage2 = backbone.inplanes // 8
        self.z = backbone.inplanes
        self.in_channels_list = [
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list[1:],  # nonzero only
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels, out_channels),
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        print(self.z)
        del x['0']
        x = self.fpn(x)
        return list(x.values())

import torch
backbone = ResNetnew().to('cuda')

output_names = ["output_0"]
input_names = ["input1"]

x = torch.zeros(1, 3, 1024, 2048, requires_grad=True).to('cuda')
torch.onnx.export(backbone, x, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)
