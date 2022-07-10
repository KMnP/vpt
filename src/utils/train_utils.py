#!/usr/bin/env python3
import torch


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (GB)."""
    if not torch.cuda.is_available():
        return 0
    # Number of bytes in a megabyte
    _B_IN_GB = 1024 * 1024 * 1024

    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / _B_IN_GB


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
