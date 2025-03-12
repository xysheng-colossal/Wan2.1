import logging
import torch
from torch import Tensor
import torch_npu
import torch.distributed as dist
import math
import os
from yunchang import LongContextAttention
try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")
from typing import Any
from yunchang.comm.all_to_all import SeqAllToAll4D
from .new_parallel import all_to_all_v1, all_to_all_v2, pad, la_preprocess_input, la_postprocess_output

logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647

class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        # args: Any,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        attn_type: AttnType = AttnType.FA,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            attn_type = attn_type,
        )
        self.use_kv_cache = use_kv_cache
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )
        self.world_size = dist.get_world_size()
        # self.args = args
        self.video_size = [[720, 1280], [1280, 720], [960, 960]]

        self.algo = int(os.getenv('ALGO', 0))
        if self.algo == 1:
            torch.ops.load_library("/home/gwn/test_la/0304/plugin/build/libPTAExtensionOPS.so")
        self.self_attention = None

    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        scale=None
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy  
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """

        all_gather = False
        use_all_head = True
        # use_la = False
        # if self.args.video_size in self.video_size:
        #     all_gather = False
        # else:
        #     all_gather = True
        if not all_gather:
            if self.world_size == 2 or self.world_size == 4 or self.world_size == 8:
                # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
                # scatter 2, gather 1
                if self.use_pack_qkv:
                    qkv = torch.cat([query, key, value]).continous()
                    qkv = SeqAllToAll4D.apply(
                        self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
                    )
                    query_layer, key_layer, value_layer = torch.chunk(qkv, 3, dim=0)
                else:
                    query_layer = SeqAllToAll4D.apply(
                        self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                    )
                    key_layer = SeqAllToAll4D.apply(
                        self.ulysses_pg, key, self.scatter_idx, self.gather_idx
                    )
                    value_layer = SeqAllToAll4D.apply(
                        self.ulysses_pg, value, self.scatter_idx, self.gather_idx
                    )
            elif self.world_size == 16:
                # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
                # scatter 2, gather 1
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                )
                key_value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, torch.cat((key, value), dim=0), self.scatter_idx, self.gather_idx
                )

                b, s, n, d = key_value_layer.shape
                kv_full = torch.empty([2, b, s, n, d], dtype=query_layer.dtype, device=query_layer.device)
                dist.all_gather_into_tensor(kv_full, key_value_layer, group=self.ring_pg)
                kv_full = kv_full.permute(1, 3, 0, 2, 4).reshape(b, n, -1, d)

                query_layer = query_layer.transpose(1, 2)
                key_layer, value_layer = kv_full.chunk(2, dim=0)

            if use_all_head:
                if self.algo == 0:
                    out = torch_npu.npu_fusion_attention(
                            query_layer,
                            key_layer,
                            value_layer,
                            head_num=query_layer.shape[1],
                            input_layout="BNSD",
                            scale=scale,
                            pre_tockens=MAX_TOKEN,
                            next_tockens=MAX_TOKEN
                        )[0]
                    out = out.transpose(1, 2)
                elif self.algo == 1:
                    q_seqlen = query_layer.shape[2]
                    q_dtype = query_layer.dtype
                    head_dim = query_layer.shape[-1]

                    query, key, value = la_preprocess_input(query_layer, key_layer, value_layer)
                    _, out = torch.ops.mindie.la_mindie_sd(query, key, value, None, None, None, \
                        query_layer.shape[-1]**-0.5, 5, "BNSD", 1.0, MAX_TOKEN, 1, True)
                    out = la_postprocess_output(out, q_dtype, q_seqlen, head_dim)
                    out = out.transpose(1, 2)
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, but got f{self.algo}")

            else:
                query_layer_list = query_layer.split(1, dim=1)
                key_layer_list = key_layer.split(1, dim=1)
                value_layer_list = value_layer.split(1, dim=1)
                output = []
                for_loop = query_layer.shape[1]
                for i in range(for_loop):
                    if self.algo == 0:
                        out = torch_npu.npu_fusion_attention(
                            query_layer_list[i],
                            key_layer_list[i],
                            value_layer_list[i],
                            head_num=1,
                            input_layout="BNSD",
                            scale=query.shape[-1]**-0.5,
                            pre_tockens=MAX_TOKEN,
                            next_tockens=MAX_TOKEN
                        )[0]
                    elif self.algo == 1:
                        q_seqlen = query_layer_list[0].shape[2]
                        q_dtype = query_layer.dtype
                        head_dim = query_layer.shape[-1]

                        query, key, value = la_preprocess_input(query_layer, key_layer, value_layer)
                        _, out = torch.ops.mindie.la_mindie_sd(query, key, value, None, None, None, \
                            query_layer.shape[-1]**-0.5, 1, "BNSD", 1.0, MAX_TOKEN, 1, True)
                        out = la_postprocess_output(out, q_dtype, q_seqlen, head_dim)

                    output.append(out)
                out_concat = torch.cat(output, dim=1)
                out = out_concat.transpose(1, 2)

            if type(out) == tuple:
                context_layer, _, _ = out
            else:
                context_layer = out

            # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
            # scatter 1, gather 2
            output = SeqAllToAll4D.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            )
        else:
            if self.world_size == 2 or self.world_size == 4 or self.world_size == 8:
                output = all_to_all_v1(
                    query,
                    key,
                    value,
                    joint_tensor_key,
                    joint_tensor_value,
                    2,
                    1,
                    scale=query.shape[-1]**-0.5,
                    algo = self.algo,
                    self_attention = self.self_attention)
            elif self.world_size == 16:
                output = all_to_all_v2(
                    query,
                    key,
                    value,
                    joint_tensor_key,
                    joint_tensor_value,
                    self.ulysses_pg,
                    self.ring_pg,
                    2,
                    1,
                    scale=query.shape[-1]**-0.5,
                    algo = self.algo,
                    self_attention = self.self_attention)
        return output

