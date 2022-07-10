#!/usr/bin/env python3

"""
Convnext-related models:
"imagenet_sup_rnx_tiny",
"imagenet_sup_rnx_small",
"imagenet_sup_rnx_base",
"imagenet22k_sup_rnx_base",
"imagenet22k_sup_rnx_large",
"imagenet22k_sup_rnx_xlarge",
"""
import torch
import torch.nn as nn
import torchvision as tv

from collections import OrderedDict
from timm.models.layers import trunc_normal_

from .convnext_backbone.convnext import (
    convnext_tiny, convnext_small, convnext_base,
    convnext_large, convnext_xlarge
)
from .resnet import ResNet
from ..utils import logging
logger = logging.get_logger("visual_prompt")
FEAT2DIM = {
    "tiny": 768,
    "small": 768,
    "base": 1024,
    "large": 1536,
    "xlarge": 2048,
}


class ConvNeXt(ResNet):
    """
    ConvNeXt model,
    utilizing the ResNet class for structure and prompt setup
    """

    def __init__(self, cfg):
        if cfg.DATA.FEATURE not in [
            "imagenet_sup_rnx_tiny",
            "imagenet_sup_rnx_small",
            "imagenet_sup_rnx_base",
            "imagenet22k_sup_rnx_base",
            "imagenet22k_sup_rnx_large",
            "imagenet22k_sup_rnx_xlarge",
        ]:
            raise ValueError("feature does not support ConvNeXt models")
        if cfg.MODEL.PROMPT.LOCATION == "below":
            raise ValueError("Not support prompt-below at the moment")
        super(ConvNeXt, self).__init__(cfg)

    def get_outputdim(self):
        backbone_arch = self.cfg.DATA.FEATURE.split("_")[-1]
        return FEAT2DIM[backbone_arch]

    def setup_grad(self, model):
        # TODO: change the name of layers
        """
        downsample_layers[0], stages[0]
        downsample_layers[1], stages[1]
        downsample_layers[2], stages[2]
        downsample_layers[3], stages[3]
        norm
        """
        self.norm = model.norm
        transfer_type = self.cfg.MODEL.TRANSFER_TYPE
        # split enc into 3 parts:
        #           prompt_layers  frozen_layers         tuned_layers
        # partial-0  identity       all but last block
        #                          stages[-1][-1],  stages[-1][-1] + norm
        # linear     identity        all            identity
        # end2end    identity       identity        all
        # prompt-pad   identity        all

        # partial, linear, end2end, prompt-pad
        self.prompt_layers = nn.Identity()

        if transfer_type == "partial-0":
            # last block to tune
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("downsample_layer1", model.downsample_layers[0]),
                ("stage1", model.stages[0]),
                ("downsample_layer2", model.downsample_layers[1]),
                ("stage2", model.stages[1]),
                ("downsample_layer3", model.downsample_layers[2]),
                ("stage3", model.stages[2]),
                ("downsample_layer4", model.downsample_layers[3]),
                ("stage4", model.stages[3][:-1]),
            ]))
            self.tuned_layers = nn.Sequential(OrderedDict([
                ("stage4", model.stages[3][-1]),
            ]))
            self.tune_norm = True

        elif transfer_type == "linear" or transfer_type == "side" or transfer_type == "tinytl-bias":  # noqa
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("downsample_layer1", model.downsample_layers[0]),
                ("stage1", model.stages[0]),
                ("downsample_layer2", model.downsample_layers[1]),
                ("stage2", model.stages[1]),
                ("downsample_layer3", model.downsample_layers[2]),
                ("stage3", model.stages[2]),
                ("downsample_layer4", model.downsample_layers[3]),
                ("stage4", model.stages[3]),
            ]))
            self.tuned_layers = nn.Identity()
            self.tune_norm = False

        elif transfer_type == "end2end":
            self.frozen_layers = nn.Identity()
            self.tuned_layers = nn.Sequential(OrderedDict([
                ("downsample_layer1", model.downsample_layers[0]),
                ("stage1", model.stages[0]),
                ("downsample_layer2", model.downsample_layers[1]),
                ("stage2", model.stages[1]),
                ("downsample_layer3", model.downsample_layers[2]),
                ("stage3", model.stages[2]),
                ("downsample_layer4", model.downsample_layers[3]),
                ("stage4", model.stages[3]),
            ]))
            self.tune_norm = True

        elif transfer_type == "prompt" and self.cfg.MODEL.PROMPT.LOCATION == "pad": # noqa
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("downsample_layer1", model.downsample_layers[0]),
                ("stage1", model.stages[0]),
                ("downsample_layer2", model.downsample_layers[1]),
                ("stage2", model.stages[1]),
                ("downsample_layer3", model.downsample_layers[2]),
                ("stage3", model.stages[2]),
                ("downsample_layer4", model.downsample_layers[3]),
                ("stage4", model.stages[3]),
            ]))
            self.tuned_layers = nn.Identity()
            self.tune_norm = False

        if transfer_type == "tinytl-bias":
            for k, p in self.frozen_layers.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False
        else:
            for k, p in self.frozen_layers.named_parameters():
                p.requires_grad = False

        if not self.tune_norm:
            for k, p in self.norm.named_parameters():
                p.requires_grad = False
        self.transfer_type = transfer_type

    def _setup_prompt_below(self, prompt_config, model):
        # TODO:
        # the only difference btw this function and that of the ResNet class is the name of the first layer
        if prompt_config.INITIATION == "random":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens,
                    self.cfg.DATA.CROPSIZE, self.cfg.DATA.CROPSIZE
            ))
            nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
            self.prompt_norm = tv.transforms.Normalize(
                mean=[sum([0.485, 0.456, 0.406])/3] * self.num_tokens,
                std=[sum([0.229, 0.224, 0.225])/3] * self.num_tokens,
            )

        elif prompt_config.INITIATION == "gaussian":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens,
                    self.cfg.DATA.CROPSIZE, self.cfg.DATA.CROPSIZE
            ))

            nn.init.normal_(self.prompt_embeddings.data)
            self.prompt_norm = nn.Identity()

        else:
            raise ValueError("Other initiation scheme is not supported")

        # modify first conv layer
        old_weight = model.downsample_layers[0][0].weight  # [*, 3, 4, 4]
        model.downsample_layers[0][0] = nn.Conv2d(
            self.num_tokens+3, old_weight.shape[0], kernel_size=4, stride=4
        )
        trunc_normal_(model.downsample_layers[0][0].weight, std=.02)
        torch.nn.init.constant_(model.downsample_layers[0][0].bias, 0)

        model.downsample_layers[0][0].weight[:, :3, :, :].data.copy_(old_weight)
        return model

    def get_pretrained_model(self, model_type):
        backbone_arch = model_type.split("_")[-1]
        is_22k = "22k" in model_type
        if is_22k:
            # need to specify num_classes, o.w. throw error of weight size mismatch
            num_classes = 21841
        else:
            num_classes = 1000

        if backbone_arch == "tiny":
            model = convnext_tiny(pretrained=True)
        elif backbone_arch == "small":
            model = convnext_small(pretrained=True)
        elif backbone_arch == "base":
            model = convnext_base(
                pretrained=True, in_22k=is_22k, num_classes=num_classes)
        elif backbone_arch == "large":
            model = convnext_large(
                pretrained=True, in_22k=is_22k, num_classes=num_classes)
        elif backbone_arch == "xlarge":
            model = convnext_xlarge(
                pretrained=True, in_22k=is_22k, num_classes=num_classes)
        else:
            raise ValueError("model type not supported for resnet backbone")

        model.head = nn.Identity()
        return model

    def get_features(self, x):
        """get a (batch_size, feat_dim) feature"""
        if self.frozen_layers.training:
            self.frozen_layers.eval()

        if "prompt" not in self.transfer_type:
            with torch.set_grad_enabled(self.frozen_layers.training):
                x = self.frozen_layers(x)
        else:
            # prompt tuning required frozen_layers saving grad
            x = self.incorporate_prompt(x)
            x = self.frozen_layers(x)

        x = self.tuned_layers(x)  # batch_size x 2048 x h x w
        x = self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        return x
