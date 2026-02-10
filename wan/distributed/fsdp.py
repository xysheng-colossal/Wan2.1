# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from functools import partial
import os

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


def _resolve_sharding_strategy(default_strategy):
    strategy_name = os.getenv("WAN_FSDP_SHARDING_STRATEGY", "").strip().upper()
    if not strategy_name:
        return default_strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    }
    if strategy_name not in strategy_map:
        raise ValueError(
            "WAN_FSDP_SHARDING_STRATEGY must be FULL_SHARD or SHARD_GRAD_OP, "
            f"but got {strategy_name}"
        )
    return strategy_map[strategy_name]


def shard_model(
    model,
    device_id,
    # param_dtype=torch.float32,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    # Inference defaults to SHARD_GRAD_OP to avoid repeated per-step all-gather.
    # Set WAN_FSDP_SHARDING_STRATEGY=FULL_SHARD to restore previous behavior.
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    # All ranks load the same checkpoint; avoid one-time state sync by default.
    sync_module_states=False,
):
    sharding_strategy = _resolve_sharding_strategy(sharding_strategy)
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)
    return model
