import torch

import torch.distributed as dist


def _preprocess_qkv_a2a(input_: torch.Tensor, seq_world_size: int):
    assert (
            input_.dim() == 4
    ), f"input_ must be 4D tensor, got {input_.dim()} and shape {input_.shape}"

    bs, shard_seqlen, hc, hs = input_.shape
    assert (
            hc % seq_world_size == 0
    ), f"head count {hc} must be divisible by sequence world size {seq_world_size}"
    shard_hc = hc // seq_world_size

    input_t = (
        input_.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
        .transpose(0, 2)
        .contiguous()
    )
    return input_t, bs, shard_seqlen, shard_hc, hs


def _postprocess_qkv_a2a(output: torch.Tensor, bs: int, seqlen: int, shard_hc: int, hs: int):
    output = output.reshape(seqlen, bs, shard_hc, hs)
    return output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)


def all_to_all_4D_qkv_packed(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        group=None,
        use_sync: bool = False,
):
    """
    Packed QKV all-to-all for Ulysses attention.

    This is equivalent to calling all_to_all_4D(..., scatter_idx=2, gather_idx=1)
    on q, k, and v separately, but it launches a single collective by packing the
    three preprocessed tensors along the per-rank sequence dimension.
    """
    assert q.shape == k.shape == v.shape, (
        f"q, k and v must have the same shape, got {q.shape}, {k.shape}, {v.shape}"
    )

    seq_world_size = dist.get_world_size(group)
    q_t, bs, shard_seqlen, shard_hc, hs = _preprocess_qkv_a2a(q, seq_world_size)
    k_t, k_bs, k_shard_seqlen, k_shard_hc, k_hs = _preprocess_qkv_a2a(k, seq_world_size)
    v_t, v_bs, v_shard_seqlen, v_shard_hc, v_hs = _preprocess_qkv_a2a(v, seq_world_size)
    assert (
        (bs, shard_seqlen, shard_hc, hs)
        == (k_bs, k_shard_seqlen, k_shard_hc, k_hs)
        == (v_bs, v_shard_seqlen, v_shard_hc, v_hs)
    )

    packed_input = torch.cat([q_t, k_t, v_t], dim=1)
    packed_output = torch.empty_like(packed_input)

    if seq_world_size > 1:
        dist.all_to_all_single(packed_output, packed_input, group=group)
        if use_sync:
            torch.npu.synchronize()
    else:
        packed_output = packed_input

    q_out, k_out, v_out = packed_output.split(shard_seqlen, dim=1)
    seqlen = shard_seqlen * seq_world_size
    return (
        _postprocess_qkv_a2a(q_out, bs, seqlen, shard_hc, hs),
        _postprocess_qkv_a2a(k_out, bs, seqlen, shard_hc, hs),
        _postprocess_qkv_a2a(v_out, bs, seqlen, shard_hc, hs),
    )


def all_to_all_4D(
        input_: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input_ (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
            input_.dim() == 4
    ), f"input_ must be 4D tensor, got {input_.dim()} and shape {input_.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input_ (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input_.shape
        seqlen = shard_seqlen * seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t, bs, shard_seqlen, shard_hc, hs = _preprocess_qkv_a2a(input_, seq_world_size)

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.npu.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = _postprocess_qkv_a2a(output, bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input_ (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input_.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input_.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.npu.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")
