#!/usr/bin/env python3
"""
borrow from https://github.com/facebookresearch/moco-v3/blob/main/vits.py
"""
import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

from .adapter_block import Pfeiffer_Block
from ..vit_backbones.vit_moco import VisionTransformerMoCo
from ...utils import logging
logger = logging.get_logger("visual_prompt")


class ADPT_VisionTransformerMoCo(VisionTransformerMoCo):
    def __init__(
        self, 
        adapter_cfg,
        stop_grad_conv1=False,
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
        
        super(ADPT_VisionTransformerMoCo, self).__init__(
            stop_grad_conv1=stop_grad_conv1,
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



def vit_base(adapter_cfg, **kwargs):
    model = ADPT_VisionTransformerMoCo(
        adapter_cfg,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
