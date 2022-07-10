#!/usr/bin/env python3
"""
vit with prompt: also included different VPT ablations
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        
        if prompt_config.LOCATION == "pad":
            img_size += 2 * prompt_config.NUM_TOKENS
        
        super(PromptedTransformer, self).__init__(
            config, img_size, vis)
        
        self.prompt_config = prompt_config
        self.vit_config = config
        
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        if self.prompt_config.LOCATION == "add":
            num_tokens = self.embeddings.position_embeddings.shape[1]
        elif self.prompt_config.LOCATION == "add-1":
            num_tokens = 1
        else:
            num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            if self.prompt_config.LOCATION == "below":
                self.embeddings.patch_embeddings = Conv2d(
                    in_channels=num_tokens+3,
                    out_channels=config.hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size
                )
                # add xavier_uniform initialization
                nn.init.uniform_(
                    self.embeddings.patch_embeddings.weight, -val, val)
                nn.init.zeros_(
                    self.embeddings.patch_embeddings.bias)

                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, img_size[0], img_size[1]))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            elif self.prompt_config.LOCATION == "pad":
                self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * num_tokens, img_size[0]
                ))
                self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, img_size[0] - 2 * num_tokens, 2 * num_tokens
                ))

                nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
                nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

                self.prompt_norm = tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )

            elif self.prompt_config.LOCATION == "prepend-pixel":
                p_size = config.patches["size"][0]  # 16 or 14
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, 3, p_size, p_size * num_tokens))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                self.prompt_norm = nn.Identity()

            elif self.prompt_config.LOCATION == "prepend-pixel-gaussian":
                p_size = config.patches["size"][0]  # 16 or 14
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, 3, p_size, p_size * num_tokens))

                nn.init.normal_(self.prompt_embeddings.data)
                self.prompt_norm = nn.Identity()

            elif self.prompt_config.LOCATION == "prepend-pixel-imgnettransform":
                p_size = config.patches["size"][0]  # 16 or 14
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, 3, p_size, p_size * num_tokens))

                self.prompt_norm = tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
                nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)

            # including ['prepend', 'add', 'add-1']
            else:
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

                if self.prompt_config.LOCATION in ["prepend", "add", "add-1"] and self.prompt_config.DEEP:  # noqa

                    if self.prompt_config.NUM_DEEP_LAYERS is None:
                        total_d_layer = config.transformer["num_layers"]-1

                    else:
                        if self.prompt_config.REVERSE_DEEP:
                            total_d_layer = self.prompt_config.NUM_DEEP_LAYERS
                            del self.prompt_embeddings

                        else:
                            total_d_layer = self.prompt_config.NUM_DEEP_LAYERS - 1

                    self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                        total_d_layer, num_tokens, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            # val = 0.5
        elif self.prompt_config.INITIATION == "final-cls":
            # use final cls-token embds for all prompts
            cls_embds = np.load(self.prompt_config.CLSEMB_PATH)
            cls_embds = torch.from_numpy(cls_embds).to(torch.float32)
            cls_embds = cls_embds[:, -1, :].unsqueeze(0)
            assert num_tokens == cls_embds.shape[1]
            # (1,  num_tokens, 768)

            if self.prompt_config.LOCATION == "prepend":
                self.prompt_embeddings = nn.Parameter(cls_embds)

                if self.prompt_config.DEEP:  # noqa
                    assert self.prompt_config.NUM_DEEP_LAYERS is None
                    total_d_layer = config.transformer["num_layers"] - 1
                    # (total_d_layer, num_tokens, prompt_dim)
                    self.deep_prompt_embeddings = nn.Parameter(
                        cls_embds.expand(total_d_layer, -1, -1))

        elif self.prompt_config.INITIATION == "cls-nolastl":
            # use the corresponding cls-token embds for all prompts, excluding the last output
            cls_embds = np.load(self.prompt_config.CLSEMB_PATH)
            cls_embds = torch.from_numpy(cls_embds).to(torch.float32)
            cls_embds = cls_embds[:, :-1, :].transpose(1, 0)
            assert num_tokens == cls_embds.shape[1]
            # (12,  num_tokens, 768)

            if self.prompt_config.LOCATION == "prepend":
                self.prompt_embeddings = nn.Parameter(cls_embds[:1, :, :])

                if self.prompt_config.DEEP:  # noqa
                    assert self.prompt_config.NUM_DEEP_LAYERS is None
                    # (total_d_layer, num_tokens, prompt_dim)
                    self.deep_prompt_embeddings = nn.Parameter(
                        cls_embds[1:, :, :])

        elif self.prompt_config.INITIATION == "cls-nofirstl":
            # use the corresponding cls-token embds for all prompts, excluding the first input
            cls_embds = np.load(self.prompt_config.CLSEMB_PATH)
            cls_embds = torch.from_numpy(cls_embds).to(torch.float32)
            cls_embds = cls_embds[:, 1:, :].transpose(1, 0)
            assert num_tokens == cls_embds.shape[1]
            # (12,  num_tokens, 768)

            if self.prompt_config.LOCATION == "prepend":
                self.prompt_embeddings = nn.Parameter(cls_embds[:1, :, :])

                if self.prompt_config.DEEP:  # noqa
                    assert self.prompt_config.NUM_DEEP_LAYERS is None
                    # (total_d_layer, num_tokens, prompt_dim)
                    self.deep_prompt_embeddings = nn.Parameter(
                        cls_embds[1:, :, :])
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
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION in ["prepend-pixel", "prepend-pixel-gaussian", "prepend-pixel-imgnettransform"]:
            prompt_embeds = self.embeddings.patch_embeddings(
                self.prompt_norm(self.prompt_embeddings))  # (1, hidden_dim, 1, n_prompt)
            prompt_embeds = prompt_embeds.flatten(2).transpose(-1, -2)  # (1, n_prompt, hidden_dim)  # noqa
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat((
                    x[:, :1, :],
                    prompt_embeds.expand(B, -1, -1),
                    x[:, 1:, :]
                ), dim=1)

        elif self.prompt_config.LOCATION == "add-1":
            # add to the input patches + CLS
            # assert self.prompt_config.NUM_TOKENS == x.shape[1]
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            L = x.shape[1]
            prompt_emb = self.prompt_dropout(self.prompt_proj(
                self.prompt_embeddings).expand(B, -1, -1))
            x = x + prompt_emb.expand(-1, L, -1)
            # (batch_size, cls_token + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "add":
            # add to the input patches + CLS
            # assert self.prompt_config.NUM_TOKENS == x.shape[1]
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = x + self.prompt_dropout(
                self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
            # (batch_size, cls_token + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "pad":
            prompt_emb_lr = self.prompt_norm(
                self.prompt_embeddings_lr).expand(B, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(
                self.prompt_embeddings_tb).expand(B, -1, -1, -1)

            x = torch.cat((
                prompt_emb_lr[:, :, :, :self.num_tokens],
                x, prompt_emb_lr[:, :, :, self.num_tokens:]
                ), dim=-1)
            x = torch.cat((
                prompt_emb_tb[:, :, :self.num_tokens, :],
                x, prompt_emb_tb[:, :, self.num_tokens:, :]
            ), dim=-2)
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "below":
            # (batch, 3, height, width)
            x = torch.cat((
                    x,
                    self.prompt_embeddings.expand(B, -1, -1, -1),
                ), dim=1)
            x = self.embeddings(x)
            # (batch_size, cls_token + n_patches, hidden_dim)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    if self.prompt_config.DEEP_SHARED:
                        # use the same shallow prompt embd for all layers
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.prompt_embeddings).expand(B, -1, -1))
                    else:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    if self.prompt_config.LOCATION == "prepend":
                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            deep_prompt_emb,
                            hidden_states[:, (1+self.num_tokens):, :]
                        ), dim=1)

                    elif self.prompt_config.LOCATION == "add":
                        hidden_states = hidden_states + deep_prompt_emb

                    elif self.prompt_config.LOCATION == "add-1":
                        L = hidden_states.shape[1]
                        hidden_states = hidden_states + deep_prompt_emb.expand(
                            -1, L, -1)
                    else:
                        raise ValueError("prompt location {} is not supported".format(self.prompt_config.LOCATION))

                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward_reverse_deep_prompt(self, x):
        hidden_states = self.embeddings(x)

        attn_weights = []
        weights = None
        B = x.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]
        num_deep_layers = self.deep_prompt_embeddings.shape[0]
        assert num_deep_layers == self.prompt_config.NUM_DEEP_LAYERS

        # no prompt
        for i in range(num_layers - num_deep_layers):
            hidden_states, weights = self.encoder.layer[i](hidden_states)
            if self.encoder.vis:
                attn_weights.append(weights)

        # insert prompt
        for deep_idx in range(num_deep_layers):
            i = num_layers - num_deep_layers + deep_idx
            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                self.deep_prompt_embeddings[deep_idx]).expand(B, -1, -1))

            if self.prompt_config.LOCATION == "prepend":
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    deep_prompt_emb,
                    hidden_states[:, (1+self.num_tokens):, :]
                ), dim=1)
            elif self.prompt_config.LOCATION == "add":
                hidden_states = hidden_states + deep_prompt_emb

            elif self.prompt_config.LOCATION == "add-1":
                L = hidden_states.shape[1]
                hidden_states = hidden_states + deep_prompt_emb.expand(
                    -1, L, -1)
            else:
                raise ValueError("prompt location {} is not supported".format(self.prompt_config.LOCATION))

            hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward_noexpand_deep_prompt(self, embedding_output):
        # insert deep prompts up to some layers, and reduce the input sequence back to the original
        if self.prompt_config.LOCATION != "prepend":
            raise ValueError("prompt location {} is not supported".format(self.prompt_config.LOCATION))
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)

                elif i == self.deep_prompt_embeddings.shape[0] + 1:
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)

                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        if self.prompt_config.REVERSE_DEEP:
            encoded, attn_weights = self.forward_reverse_deep_prompt(x)

        elif self.prompt_config.FORWARD_DEEP_NOEXPAND:
            embedding_output = self.incorporate_prompt(x)
            encoded, attn_weights = self.forward_noexpand_deep_prompt(
                embedding_output)

        else:
            # this is the default version:
            embedding_output = self.incorporate_prompt(x)

            if self.prompt_config.DEEP:
                encoded, attn_weights = self.forward_deep_prompt(
                    embedding_output)
            else:
                encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        if self.prompt_cfg.VIT_POOL_TYPE == "original":
            x = x[:, 0]
        elif self.prompt_cfg.VIT_POOL_TYPE == "imgprompt_pool":
            assert self.prompt_cfg.LOCATION == "prepend"
            x = x[:, 1:, :].mean(dim=1)
        elif self.prompt_cfg.VIT_POOL_TYPE == "img_pool":
            assert self.prompt_cfg.LOCATION == "prepend"
            x = x[:, self.transformer.num_tokens+1:, :].mean(dim=1)
        elif self.prompt_cfg.VIT_POOL_TYPE == "prompt_pool":
            assert self.prompt_cfg.LOCATION == "prepend"
            x = x[:, 1:self.transformer.num_tokens+1, :].mean(dim=1)
        else:
            raise ValueError("pooling type for output is not supported")

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.transformer.prompt_config.LOCATION != "below":
                self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            else:
                # [768, 4, 16, 16]
                self.transformer.embeddings.patch_embeddings.weight[:, :3, :, :].copy_(np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

