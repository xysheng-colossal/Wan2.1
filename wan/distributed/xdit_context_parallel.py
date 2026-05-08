# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch_npu
from .parallel_mgr import (get_sequence_parallel_rank,
                            get_sequence_parallel_world_size,
                            get_sp_group,
                            )
from ..modules.attn_layer import xFuserLongContextAttention
from .comm import all_to_all_4D

from ..modules.model import sinusoidal_embedding_1d
from wan.utils.rainfusion import Rainfusion
from mindiesd import rotary_position_embedding


def _get_hccl_comm_name(group):
    backend = group._get_backend(torch.device("npu"))
    get_name = backend.get_hccl_comm_name
    rank = dist.get_rank(group)
    try:
        return get_name(rank)
    except TypeError:
        return get_name()


def _out_proj_reduce_scatter(context_layer, proj, group):
    b, seqlen, shard_hc, hs = context_layer.shape
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return proj(context_layer.flatten(2))

    shard_dim = shard_hc * hs
    dim = shard_dim * world_size
    assert proj.in_features == dim, (
        f"o_proj in_features {proj.in_features} does not match gathered dim {dim}"
    )
    assert seqlen % world_size == 0, (
        f"sequence length {seqlen} must be divisible by world size {world_size}"
    )

    rank = dist.get_rank(group)
    weight_shard = proj.weight[:, rank * shard_dim:(rank + 1) * shard_dim]
    mm_weight = weight_shard.transpose(0, 1).contiguous()
    mm_input = context_layer.permute(1, 0, 2, 3).reshape(seqlen * b, shard_dim).contiguous()
    hcom = _get_hccl_comm_name(group)
    output = torch_npu.npu_mm_reduce_scatter_base(
        mm_input,
        mm_weight,
        hcom,
        world_size,
        reduce_op="sum",
    )

    shard_seqlen = seqlen // world_size
    output = output.reshape(shard_seqlen, b, dim).permute(1, 0, 2).contiguous()
    if proj.bias is not None:
        output = output + proj.bias
    return output

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    if pad_size == 0:
        return original_tensor
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=torch.float32,
        device=original_tensor.device
        ).to(original_tensor.dtype)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs_list):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        x_i = x[i, :s].reshape(1, s, n, c)
        cos, sin = freqs_list[i]
        x_i = rotary_position_embedding(x_i, cos, sin, rotated_mode="rotated_interleaved", fused=True)
        output.append(x_i)
    return torch.cat(output)


def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    t_idx=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.rainfusion_config and self.rainfusion_config["atten_mask_all"] is None:
        self.rainfusion_config["grid_size"] = Rainfusion.get_grid_size(x[0].shape, self.patch_size)
        logging.info(f"Rainfusion grid size: {self.rainfusion_config['grid_size']}")
        self.rainfusion_config["atten_mask_all"] = Rainfusion.get_atten_mask(
            grid_size=self.rainfusion_config["grid_size"],
            sparsity=self.rainfusion_config["sparsity"]
        )
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    # with amp.autocast(dtype=torch.float32):
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).float())
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # Context Parallel
    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    if self.freqs_list is None:
        c = (self.dim // self.num_heads) // 2
        s = x.shape[1]
        freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        freqs_list=[]

        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)

            # apply rotary embedding
            sp_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            freqs_i = pad_freqs(freqs_i, s * sp_size)
            s_per_rank = s
            freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                        s_per_rank), :, :]
            cos, sin = torch.chunk(torch.view_as_real(freqs_i_rank.to(torch.complex64)), 2, dim=-1)
            cos = cos.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
            sin = sin.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
            freqs_i_rank = (cos, sin)
            freqs_list.append(freqs_i_rank)
        self.freqs_list = freqs_list

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs_list,
        context=context,
        context_lens=context_lens,
        rainfusion_config=self.rainfusion_config,
        t_idx=t_idx,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     args,
                     dtype=torch.bfloat16,
                     rainfusion_config=None, 
                     t_idx=None):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = xFuserLongContextAttention(args, rainfusion_config=rainfusion_config)(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size,
        t_idx=t_idx,
    )

    if int(os.getenv("WAN_OUT_PROJ_RS", 0)) == 1:
        context_layer = x
        x = _out_proj_reduce_scatter(context_layer, self.o, get_sp_group().ulysses_group)
        if int(os.getenv("WAN_OUT_PROJ_RS_VERIFY", 0)) == 1:
            ref = all_to_all_4D(input_=context_layer, scatter_idx=1, gather_idx=2, group=get_sp_group().ulysses_group)
            ref = self.o(ref.flatten(2))
            rtol = float(os.getenv("WAN_OUT_PROJ_RS_VERIFY_RTOL", "1e-2"))
            atol = float(os.getenv("WAN_OUT_PROJ_RS_VERIFY_ATOL", "1e-2"))
            torch.testing.assert_close(x, ref, rtol=rtol, atol=atol)
    else:
        # TODO: padding after attention.
        # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

        # output
        x = x.flatten(2)
        x = self.o(x)
    return x
