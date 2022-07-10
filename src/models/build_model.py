#!/usr/bin/env python3
"""
Model construction functions.
"""
from tabnanny import verbose
import torch

from .resnet import ResNet
from .convnext import ConvNeXt
from .vit_models import ViT, Swin, SSLViT
from ..utils import logging
logger = logging.get_logger("visual_prompt")
# Supported model types
_MODEL_TYPES = {
    "resnet": ResNet,
    "convnext": ConvNeXt,
    "vit": ViT,
    "swin": Swin,
    "ssl-vit": SSLViT,
}


def build_model(cfg):
    """
    build model here
    """
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.TYPE)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    train_type = cfg.MODEL.TYPE
    model = _MODEL_TYPES[train_type](cfg)

    log_model_info(model, verbose=cfg.DBG)
    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")

    return model, device


def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=True,
            )
    else:
        model = model.to(cur_device)
    return model, cur_device
