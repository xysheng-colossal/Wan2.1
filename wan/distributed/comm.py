import torch

import torch.distributed as dist


def _prepare_heads_to_sequence(input_: torch.Tensor, seq_world_size: int) -> torch.Tensor:
    bs, shard_seqlen, hc, hs = input_.shape
    shard_hc = hc // seq_world_size
    return (
        input_.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
        .transpose(0, 2)
        .contiguous()
    )


def _copy_heads_to_sequence(output: torch.Tensor, input_: torch.Tensor, seq_world_size: int) -> None:
    bs, shard_seqlen, hc, hs = input_.shape
    shard_hc = hc // seq_world_size
    output.copy_(input_.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2))


def _restore_heads_to_sequence(input_: torch.Tensor, bs: int, seqlen: int) -> torch.Tensor:
    _, _, _, shard_hc, hs = input_.shape
    return input_.reshape(seqlen, bs, shard_hc, hs).transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)


def all_to_all_4D_qkv_packed(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, group=None, use_sync: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack the three Q/K/V Ulysses all-to-all operations into one collective.

    This is equivalent to calling all_to_all_4D(..., scatter_idx=2, gather_idx=1)
    for query, key, and value independently. The packed buffer concatenates the
    prepared tensors along the local sequence shard dimension.
    """
    assert query.dim() == key.dim() == value.dim() == 4
    assert query.shape == key.shape == value.shape

    seq_world_size = dist.get_world_size(group)
    bs, shard_seqlen, hc, _ = query.shape
    assert hc % seq_world_size == 0
    shard_hc = hc // seq_world_size

    packed_input = torch.empty(
        (seq_world_size, 3 * shard_seqlen, bs, shard_hc, query.shape[-1]),
        dtype=query.dtype,
        device=query.device,
    )
    _copy_heads_to_sequence(packed_input[:, :shard_seqlen], query, seq_world_size)
    _copy_heads_to_sequence(packed_input[:, shard_seqlen:2 * shard_seqlen], key, seq_world_size)
    _copy_heads_to_sequence(packed_input[:, 2 * shard_seqlen:], value, seq_world_size)
    packed_output = torch.empty_like(packed_input)

    if seq_world_size > 1:
        dist.all_to_all_single(packed_output, packed_input, group=group)
        if use_sync:
            torch.npu.synchronize()
    else:
        packed_output = packed_input

    query_out, key_out, value_out = packed_output.split(shard_seqlen, dim=1)
    seqlen = shard_seqlen * seq_world_size
    return (
        _restore_heads_to_sequence(query_out, bs, seqlen),
        _restore_heads_to_sequence(key_out, bs, seqlen),
        _restore_heads_to_sequence(value_out, bs, seqlen),
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
        input_t = _prepare_heads_to_sequence(input_, seq_world_size)

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
        return _restore_heads_to_sequence(output, bs, seqlen)

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
