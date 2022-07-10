#!/usr/bin/env python3
"""
vit-moco-v3 with prompt
"""
import math
import torch
import torch.nn as nn
import torchvision as tv

from functools import partial, reduce
from operator import mul
from torch.nn import Conv2d, Dropout
from timm.models.vision_transformer import _cfg

from ..vit_backbones.vit_mae import VisionTransformer
from ...utils import logging
logger = logging.get_logger("visual_prompt")


class PromptedVisionTransformer(VisionTransformer):
    def __init__(self, prompt_config, **kwargs):
        super().__init__(**kwargs)
        self.prompt_config = prompt_config
        if self.prompt_config.DEEP and self.prompt_config.LOCATION not in ["prepend", ]:
            raise ValueError("Deep-{} is not supported".format(self.prompt_config.LOCATION))

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    len(self.blocks) - 1,
                    num_tokens, self.embed_dim
                ))
                # xavier_uniform initialization
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.prompt_embeddings.expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        else:
            raise ValueError("Other prompt locations are not supported")
        return x

    def embeddings(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            B = x.shape[0]
            num_layers = len(self.blocks)

            for i in range(num_layers):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    # prepend
                    x = torch.cat((
                        x[:, :1, :],
                        self.prompt_dropout(
                            self.deep_prompt_embeddings[i-1].expand(B, -1, -1)
                        ),
                        x[:, (1 + self.num_tokens):, :]
                    ), dim=1)
                    x = self.blocks[i](x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if self.prompt_config.VIT_POOL_TYPE == "imgprompt_pool":
            assert self.prompt_config.LOCATION == "prepend"
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.prompt_config.VIT_POOL_TYPE == "original":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.prompt_config.VIT_POOL_TYPE == "img_pool":
            assert self.prompt_config.LOCATION == "prepend"
            x = x[:, self.num_tokens+1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        elif self.prompt_config.VIT_POOL_TYPE == "prompt_pool":
            assert self.prompt_config.LOCATION == "prepend"
            x = x[:, 1:self.num_tokens+1, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            raise ValueError("pooling type for output is not supported")

        return outcome


def build_model(model_type, prompt_cfg):
    if "vitb" in model_type:
        return vit_base_patch16(prompt_cfg)
    elif "vitl" in model_type:
        return vit_large_patch16(prompt_cfg)
    elif "vith" in model_type:
        return vit_huge_patch14(prompt_cfg)


def vit_base_patch16(prompt_cfg, **kwargs):
    model = PromptedVisionTransformer(
        prompt_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(prompt_cfg, **kwargs):
    model = PromptedVisionTransformer(
        prompt_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(prompt_cfg, **kwargs):
    model = PromptedVisionTransformer(
        prompt_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


