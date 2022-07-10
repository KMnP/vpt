#!/usr/bin/env python3
"""
borrow from https://github.com/facebookresearch/mae/blob/main/models_vit.py
"""
from functools import partial

import torch
import torch.nn as nn

from .adapter_block import Pfeiffer_Block
from ..vit_backbones.vit_mae import VisionTransformer
from timm.models.layers import PatchEmbed
from ...utils import logging
logger = logging.get_logger("visual_prompt")


class ADPT_VisionTransformer(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(
        self, 
        adapter_cfg, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        representation_size=None, 
        distilled=False,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        embed_layer=PatchEmbed, 
        norm_layer=None,
        act_layer=None, 
        weight_init='',
        **kwargs):

        super(ADPT_VisionTransformer, self).__init__(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            num_classes=num_classes, 
            embed_dim=embed_dim, 
            depth=depth,
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            representation_size=representation_size, 
            distilled=distilled,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate, 
            embed_layer=embed_layer, 
            norm_layer=norm_layer,
            act_layer=act_layer, 
            weight_init=weight_init,
            **kwargs
        )

        self.adapter_cfg = adapter_cfg
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if adapter_cfg.STYLE == "Pfeiffer":
            self.blocks = nn.Sequential(*[
                Pfeiffer_Block(
                    adapter_config=adapter_cfg, 
                    dim=embed_dim, 
                    num_heads=num_heads, 
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias, 
                    drop=drop_rate,
                    attn_drop=attn_drop_rate, 
                    drop_path=dpr[i], 
                    norm_layer=norm_layer, 
                    act_layer=act_layer) for i in range(depth)])
        else:
            raise ValueError("Other adapter styles are not supported.")



def build_model(model_type, adapter_cfg):
    if "vitb" in model_type:
        return vit_base_patch16(adapter_cfg)
    elif "vitl" in model_type:
        return vit_large_patch16(adapter_cfg)
    elif "vith" in model_type:
        return vit_huge_patch14(adapter_cfg)


def vit_base_patch16(adapter_cfg, **kwargs):
    model = ADPT_VisionTransformer(
        adapter_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(adapter_cfg, **kwargs):
    model = ADPT_VisionTransformer(
        adapter_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(adapter_cfg, **kwargs):
    model = ADPT_VisionTransformer(
        adapter_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
