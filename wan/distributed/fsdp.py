# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import inspect
from functools import partial
import logging
import os

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from .comm_order import (
    is_fsdp_allgather_wait_a2a_enabled,
    wait_current_stream_on_last_a2a_done,
    wait_stream_on_last_a2a_done,
)

logger = logging.getLogger(__name__)


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


class _FSDPBlockGroup(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, *args, **kwargs):
        for block in self.blocks:
            x = block(x, *args, **kwargs)
        return x

    def clear_runtime_cache(self):
        for block in self.blocks:
            if hasattr(block, "clear_runtime_cache"):
                block.clear_runtime_cache()


def _resolve_block_group_size():
    raw = os.getenv("WAN_FSDP_BLOCK_GROUP_SIZE", "").strip()
    if raw == "":
        return 1
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError("WAN_FSDP_BLOCK_GROUP_SIZE must be a positive integer") from exc
    if value <= 0:
        raise ValueError("WAN_FSDP_BLOCK_GROUP_SIZE must be a positive integer")
    return value


def _is_wan_dit_model(model):
    # Block grouping is intended for Wan DiT blocks only.
    required_attrs = ("blocks", "patch_embedding", "time_projection")
    return all(hasattr(model, attr) for attr in required_attrs)


def _maybe_group_blocks_for_fsdp(model):
    group_size = _resolve_block_group_size()
    if group_size <= 1:
        return
    if not _is_wan_dit_model(model):
        return
    blocks = list(model.blocks)
    if len(blocks) == 0:
        return
    if isinstance(blocks[0], _FSDPBlockGroup):
        return

    grouped_blocks = []
    for idx in range(0, len(blocks), group_size):
        grouped_blocks.append(_FSDPBlockGroup(blocks[idx:idx + group_size]))
    model.blocks = nn.ModuleList(grouped_blocks)


def _resolve_wrap_mode():
    raw = os.getenv("WAN_FSDP_WRAP_MODE", "").strip().lower()
    if raw in ("", "block", "block_only"):
        return "block"
    if raw in ("ffn", "ffn_only"):
        return "ffn_only"
    raise ValueError(
        "WAN_FSDP_WRAP_MODE must be one of {block, ffn_only}, "
        f"but got {raw}"
    )


def _collect_wrap_targets(model, wrap_mode):
    if not hasattr(model, "blocks"):
        return []

    if wrap_mode == "ffn_only" and _is_wan_dit_model(model):
        targets = [block.ffn for block in model.blocks if hasattr(block, "ffn")]
        if len(targets) > 0:
            return targets

    return list(model.blocks)


def _register_fsdp_allgather_wait_a2a_hooks(model):
    if not is_fsdp_allgather_wait_a2a_enabled():
        return

    mode_logged = False

    def _pre_forward_hook(fsdp_module, _inputs):
        streams = getattr(fsdp_module, "_streams", None)
        waited = False
        if isinstance(streams, dict):
            for stream_name, stream_obj in streams.items():
                lower_name = str(stream_name).lower()
                if "all_gather" in lower_name or "allgather" in lower_name or "unshard" in lower_name:
                    waited = wait_stream_on_last_a2a_done(stream_obj) or waited
        if not waited:
            wait_current_stream_on_last_a2a_done()

    for module in model.modules():
        if isinstance(module, FSDP):
            module.register_forward_pre_hook(_pre_forward_hook)
            mode_logged = True

    if mode_logged and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        logger.info(
            "FSDP all-gather wait-after-A2A enabled "
            "(WAN_FSDP_ALLGATHER_WAIT_A2A=%s)",
            os.getenv("WAN_FSDP_ALLGATHER_WAIT_A2A", "0"),
        )


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
    requested_wrap_mode = _resolve_wrap_mode()
    if requested_wrap_mode == "block":
        _maybe_group_blocks_for_fsdp(model)
    elif requested_wrap_mode == "ffn_only" and not _is_wan_dit_model(model):
        # Keep non-Wan models on block wrapping to avoid affecting T5 behavior.
        requested_wrap_mode = "block"

    wrap_targets = _collect_wrap_targets(model, requested_wrap_mode)
    if len(wrap_targets) == 0:
        raise ValueError("No FSDP wrap targets found for model.")

    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        logger.info(
            "FSDP wrap mode: %s (targets=%d)",
            requested_wrap_mode,
            len(wrap_targets),
        )

    fsdp_kwargs = dict(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: any(m is target for target in wrap_targets),
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
    _register_fsdp_allgather_wait_a2a_hooks(model)
    return model
