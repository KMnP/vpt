#!/usr/bin/env python3
"""
Modified from: fbcode/multimo/models/encoders/mlp.py
"""
import math
import torch

from torch import nn
from typing import List, Type

from ..utils import logging
logger = logging.get_logger("visual_prompt")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        return x
