import argparse
import os
import time

import torch

from wan.ops.custom_attention import (
    _direct_bsnd_self_attention,
    _reference_self_attention,
    custom_self_attention_or_fallback,
)


def _sync():
    torch.npu.synchronize()


def _bench(name, fn, q, k, v, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            fn(q, k, v)
        _sync()

        start = time.perf_counter()
        for _ in range(iters):
            out = fn(q, k, v)
        _sync()
        elapsed = time.perf_counter() - start

    print(f"{name}: avg_ms={elapsed * 1000 / iters:.3f} iters={iters}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32760)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--impl", default="direct_bsnd")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shape = (args.batch, args.seq_len, args.heads, args.head_dim)
    torch.manual_seed(20260509)
    q = torch.randn(shape, dtype=dtype, device="npu")
    k = torch.randn(shape, dtype=dtype, device="npu")
    v = torch.randn(shape, dtype=dtype, device="npu")

    print(f"shape={shape} dtype={dtype} impl={args.impl}")
    ref = _bench("reference_mindiesd_bnsd", _reference_self_attention, q, k, v, args.warmup, args.iters)

    os.environ["WAN_CUSTOM_SELF_ATTN"] = "1"
    os.environ["WAN_CUSTOM_SELF_ATTN_IMPL"] = args.impl
    os.environ["WAN_CUSTOM_SELF_ATTN_REQUIRE_SHAPE"] = "0"
    custom = _bench(
        f"custom_{args.impl}",
        lambda q_, k_, v_: custom_self_attention_or_fallback(q_, k_, v_),
        q,
        k,
        v,
        args.warmup,
        args.iters,
    )

    direct = _direct_bsnd_self_attention(q, k, v, None)
    _sync()
    diff = (custom.float() - ref.float()).abs()
    direct_diff = (direct.float() - ref.float()).abs()
    print(f"custom_diff max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
    print(f"direct_diff max={direct_diff.max().item():.6f} mean={direct_diff.mean().item():.6f}")
    torch.testing.assert_close(custom, ref, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    main()

