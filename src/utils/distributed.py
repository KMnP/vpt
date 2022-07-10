#!/usr/bin/env python3

"""Distributed helpers."""

import torch
import torch.distributed as dist
_LOCAL_PROCESS_GROUP = None


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_master_process(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def run(
    local_rank,
    num_proc,
    func,
    init_method,
    shard_id,
    num_shards,
    backend,
    cfg,
    args,
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            loco/config/defaults.py
    """
    # Initialize the process group.
    # shard_id = get_rank()
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    func(cfg, args)


def destroy_process_group():
    """Destroys the default process group."""
    torch.distributed.destroy_process_group()


def scaled_all_reduce(cfg, tensors):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of
    the process group (equivalent to cfg.NUM_GPUS).
    """
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.NUM_GPUS / cfg.NUM_SHARDS)
    return tensors


def cat_all_gather(tensors):
    """Performs the concatenated all_gather operation on the provided tensors.
    """
    tensors_gather = [
        torch.ones_like(tensors)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensors, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def local_cat_all_gather(tensors):
    """Performs the concatenated all_gather operation on the provided tensors.
    """
    tensors_gather = [
        torch.ones_like(tensors)
        for _ in range(get_local_size())
    ]
    torch.distributed.all_gather(
        tensors_gather,
        tensors,
        async_op=False,
        group=_LOCAL_PROCESS_GROUP,
    )
    output = torch.cat(tensors_gather, dim=0)
    return output


def get_local_size():
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def get_local_rank():
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)
