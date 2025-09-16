#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu
import torch.distributed as dist
from mindiesd import attention_forward


MAX_TOKEN = 2147483647

def all_to_all_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    joint_tensor_key: torch.Tensor,
    joint_tensor_value: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    **kwargs
):
    scale = kwargs.get("scale", 1.0)
    algo = kwargs.get("algo", 0)
    self_attention = kwargs.get("self_attention", None)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    global SEQ
    b, s, n, d = k.shape
    each_n = int(n // world_size)

    output_q_list = [torch.empty([b, s_i, each_n, d], device=q.device, dtype=q.dtype) for s_i in SEQ]
    output_k_list = [torch.empty([b, s_i, each_n, d], device=k.device, dtype=k.dtype) for s_i in SEQ]
    output_v_list = [torch.empty([b, s_i, each_n, d], device=v.device, dtype=v.dtype) for s_i in SEQ]
    q_list = [t.contiguous() for t in torch.tensor_split(q, world_size, scatter_dim)]
    k_list = [t.contiguous() for t in torch.tensor_split(k, world_size, scatter_dim)]
    v_list = [t.contiguous() for t in torch.tensor_split(v, world_size, scatter_dim)]    
    dist.all_to_all(output_q_list, q_list)
    dist.all_to_all(output_k_list, k_list)
    dist.all_to_all(output_v_list, v_list)

    query = torch.cat(output_q_list, dim=gather_dim).contiguous()
    key = torch.cat(output_k_list, dim=gather_dim).contiguous()
    value = torch.cat(output_v_list, dim=gather_dim).contiguous()
    query_layer_list = query.split(1, dim=2)
    key_layer_list = key.split(1, dim=2)
    value_layer_list = value.split(1, dim=2)
    output = []
    for_loop = query.shape[2]
    for i in range(for_loop):
        if algo == 0:
            out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
        elif algo == 1:
            out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
        else:
            raise ValueError(f"select flash attention algorithm only support 0, 1, but got f{algo}")
        output.append(out)
    output = torch.cat(output, dim=2)

    output_shape = [b, SEQ[0], each_n, d] if rank < world_size - 1 else [b, SEQ[-1], each_n, d]
    output_list = [torch.empty(output_shape, device=output.device, dtype=output.dtype) for _ in SEQ]

    SEQ_joint = [i for i in SEQ]
    output_con = [chunk.contiguous() for chunk in torch.split(output, SEQ_joint, dim=gather_dim)]

    dist.all_to_all(output_list, output_con)
    output = torch.cat(output_list, dim=scatter_dim)

    return output

def all_to_all_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    joint_tensor_key: torch.Tensor,
    joint_tensor_value: torch.Tensor,
    ulysses_pg: None,
    ring_pg: None,
    scatter_dim: int,
    gather_dim: int,
    **kwargs
):
    ulysses_ranks_even = [0, 2, 4, 6, 8, 10, 12, 14]
    ulysses_ranks_odd = [1, 3, 5, 7, 9, 11, 13, 15]
    scale = kwargs.get("scale", 1.0)
    algo = kwargs.get("algo", 0)
    self_attention = kwargs.get("self_attention", None)
    ulysses_world_size = dist.get_world_size(ulysses_pg)
    rank = dist.get_rank()

    b, s, n, d = k.shape
    each_n = int(n // ulysses_world_size)

    target_ranks = ulysses_ranks_even if rank in ulysses_ranks_even else ulysses_ranks_odd
    
    output_q_list = [torch.empty([b, SEQ[rank_idx], each_n, d], device=q.device, dtype=q.dtype) for rank_idx in target_ranks]
    output_k_list = [torch.empty([b, SEQ[rank_idx], each_n, d], device=k.device, dtype=k.dtype) for rank_idx in target_ranks]
    output_v_list = [torch.empty([b, SEQ[rank_idx], each_n, d], device=v.device, dtype=v.dtype) for rank_idx in target_ranks]
    q_list = [t.contiguous() for t in torch.tensor_split(q, ulysses_world_size, scatter_dim)]
    k_list = [t.contiguous() for t in torch.tensor_split(k, ulysses_world_size, scatter_dim)]
    v_list = [t.contiguous() for t in torch.tensor_split(v, ulysses_world_size, scatter_dim)]    
    dist.all_to_all(output_q_list, q_list, group=ulysses_pg)
    dist.all_to_all(output_k_list, k_list, group=ulysses_pg)
    dist.all_to_all(output_v_list, v_list, group=ulysses_pg)

    query_layer = torch.cat(output_q_list, dim=gather_dim).contiguous()
    key = torch.cat(output_k_list, dim=gather_dim).contiguous()
    value = torch.cat(output_v_list, dim=gather_dim).contiguous()

    if rank in ulysses_ranks_odd:
        b, s, n, d = key.shape
        pad_s = SEQ[0] * 8 - s
        padding = torch.zeros([b, pad_s, n, d], dtype=key.dtype, device=key.device)
        key = torch.cat([key, padding], dim=gather_dim)
        value = torch.cat([value, padding], dim=gather_dim)
    b, s, n, d = key.shape
    k_full = torch.empty([2, b, s, n, d], dtype=key.dtype, device=key.device)
    v_full = torch.empty([2, b, s, n, d], dtype=value.dtype, device=value.device)
    dist.all_gather_into_tensor(k_full, key, group=ring_pg)
    dist.all_gather_into_tensor(v_full, value, group=ring_pg)
    k_full = k_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)
    v_full = v_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)
    key_layer = k_full[:, :sum(SEQ), :, :]
    value_layer = v_full[:, :sum(SEQ), :, :]

    if algo == 0:
        out = attention_forward(query_layer, key_layer, value_layer,
                                opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
    elif algo == 1:
        out = attention_forward(query_layer, key_layer, value_layer,
                                opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
    else:
        raise ValueError(f"select flash attention algorithm only support 0, 1, but got f{algo}")

    output_shape = [b, SEQ[rank], each_n, d]
    output_list = [torch.empty(output_shape, device=out.device, dtype=out.dtype) for _ in ulysses_ranks_even]

    if rank in ulysses_ranks_even:
        SEQ_joint = [SEQ[rank] for rank in ulysses_ranks_even]
    else:
        SEQ_joint = [SEQ[rank] for rank in ulysses_ranks_odd]
    
    output_con = [chunk.contiguous() for chunk in torch.split(out, SEQ_joint, dim=gather_dim)]
    dist.all_to_all(output_list, output_con, group=ulysses_pg)
    output = torch.cat(output_list, dim=scatter_dim)

    return output

SEQ = None

def split_sequence(input_, dim=1):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size == 1:
        return input_
    
    tensor_list = torch.chunk(input_, world_size, dim=dim)
    global SEQ
    if not SEQ and input_.shape[dim] % world_size != 0:
        SEQ = [None] * world_size
        for i in range(world_size):
            SEQ[i] = tensor_list[i].shape[1]
    output = tensor_list[rank].contiguous()
    return output

def gather_sequence(input_, dim=1):
    input_ = input_.contiguous()
    world_size = dist.get_world_size()
    if world_size == 1:
         return input_
    
    global SEQ
    if not SEQ:
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        b, s, d = input_.shape
        tensor_list = [torch.empty([b, s_i, d], device=input_.device, dtype=input_.dtype) for s_i in SEQ]
    dist.all_gather(tensor_list, input_)

    output = torch.cat(tensor_list, dim=dim)
    SEQ = None

    return output


def pad(input, base=256, dim=-2):
    shape_value = input.size(dim)
    if shape_value % base == 0:
        return input
    pad_size = ((shape_value // base) + 1) * base - shape_value
    padding_shape = list(input.shape)
    padding_shape[dim] = pad_size
    padding = torch.zeros(padding_shape, dtype=input.dtype, device=input.device)
    return torch.cat([input, padding], dim=dim)


def la_preprocess_input(query, key, value):
    if query.dtype != torch.float16:
        query = query.to(torch.float16)
        key = key.to(torch.float16)
        value = value.to(torch.float16)
    
    query = pad(query, 256, -2)
    query = pad(query, 128, -1)
    key = pad(key, 256, -2)
    key = pad(key, 128, -1)
    value = pad(value, 256, -2)
    value = pad(value, 128, -1)

    return query, key, value


def la_postprocess_output(out, dtype, q_seqlen, head_dim):
    if out.dtype != dtype:
        out = out.to(dtype)
    out = out[:, :, :q_seqlen, :head_dim]
    return out