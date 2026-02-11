# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import inspect
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
    sharding_strategy=None,
    sync_module_states=None,
    use_legacy_behavior=False,
    serialize_communication=False,
):
    if sharding_strategy is None:
        # Keep FULL_SHARD as the default strategy unless explicitly overridden.
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if sync_module_states is None:
        # Legacy default follows previous behavior for DiT FSDP construction.
        sync_module_states = True if use_legacy_behavior else False

    sharding_strategy = _resolve_sharding_strategy(sharding_strategy)
    fsdp_kwargs = dict(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks
        ),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        ),
        device_id=device_id,
        sync_module_states=sync_module_states,
    )
    fsdp_signature = inspect.signature(FSDP.__init__)
    supported_args = fsdp_signature.parameters

    reshard_env = os.getenv("WAN_FSDP_RESHARD_AFTER_FORWARD", "").strip().lower()
    if "reshard_after_forward" in supported_args and reshard_env:
        if reshard_env in ("0", "false", "off", "no"):
            fsdp_kwargs["reshard_after_forward"] = False
        elif reshard_env in ("1", "true", "on", "yes"):
            fsdp_kwargs["reshard_after_forward"] = True
        else:
            raise ValueError(
                "WAN_FSDP_RESHARD_AFTER_FORWARD must be one of "
                "{0,1,false,true,off,on,no,yes}"
            )

    if serialize_communication:
        # Serialize FSDP all-gather scheduling where available.
        if "forward_prefetch" in supported_args:
            fsdp_kwargs["forward_prefetch"] = False
        if "limit_all_gathers" in supported_args:
            fsdp_kwargs["limit_all_gathers"] = True

    model = FSDP(**fsdp_kwargs)
    return model
