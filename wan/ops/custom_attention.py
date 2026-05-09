import logging
import os

import torch
import torch_npu
from mindiesd import attention_forward


logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647
TARGET_SELF_ATTN_SHAPE = (1, 32760, 10, 128)


def _env_flag(name: str, default: int = 0) -> bool:
    return int(os.getenv(name, default)) == 1


def _supports_custom_self_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    if query.shape != key.shape or query.shape != value.shape:
        return False
    if query.dim() != 4:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if query.device.type != "npu":
        return False
    if _env_flag("WAN_CUSTOM_SELF_ATTN_REQUIRE_SHAPE", 1):
        return tuple(query.shape) == TARGET_SELF_ATTN_SHAPE
    return True


def _reference_self_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    return attention_forward(
        query,
        key,
        value,
        opt_mode="manual",
        op_type="fused_attn_score",
        layout="BNSD",
    )


def _direct_bsnd_self_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
):
    if scale is None:
        scale = query.shape[-1] ** -0.5
    return torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=query.shape[-2],
        input_layout="BSND",
        scale=scale,
        pre_tockens=MAX_TOKEN,
        next_tockens=MAX_TOKEN,
    )[0]


def custom_self_attention_or_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
):
    if not _env_flag("WAN_CUSTOM_SELF_ATTN"):
        return _reference_self_attention(query, key, value)

    if not _supports_custom_self_attention(query, key, value):
        if _env_flag("WAN_CUSTOM_SELF_ATTN_LOG_FALLBACK"):
            logger.info("WAN_CUSTOM_SELF_ATTN fallback for shape=%s dtype=%s", tuple(query.shape), query.dtype)
        return _reference_self_attention(query, key, value)

    impl = os.getenv("WAN_CUSTOM_SELF_ATTN_IMPL", "direct_bsnd")
    if impl == "direct_bsnd":
        out = _direct_bsnd_self_attention(query, key, value, scale)
    elif impl == "mindiesd_bsnd":
        out = attention_forward(
            query,
            key,
            value,
            scale=scale,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BSND",
        )
    else:
        raise ValueError(f"Unsupported WAN_CUSTOM_SELF_ATTN_IMPL={impl}")

    if _env_flag("WAN_CUSTOM_SELF_ATTN_VERIFY"):
        ref = _reference_self_attention(query, key, value)
        rtol = float(os.getenv("WAN_CUSTOM_SELF_ATTN_VERIFY_RTOL", "1e-2"))
        atol = float(os.getenv("WAN_CUSTOM_SELF_ATTN_VERIFY_ATOL", "1e-2"))
        diff = (out.float() - ref.float()).abs()
        max_diff = diff.max().detach().cpu().item()
        mean_diff = diff.mean().detach().cpu().item()
        logger.info("WAN_CUSTOM_SELF_ATTN diff: max=%s mean=%s", max_diff, mean_diff)
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)

    return out

