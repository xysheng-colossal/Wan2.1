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
from .new_parallel import all_to_all_v1, all_to_all_v2

logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647

class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        args: Any,
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
        self.args = args
        self.video_size = [[720, 1280], [1280, 720], [960, 960]]

        self.algo = 0
        # self.algo = int(os.getenv('ALGO'))
        self.self_attention = None
        if self.algo == 1:
            import torch_atb
            self_attention_param = torch_atb.SelfAttentionParam()
            if self.world_size == 8:
                self_attention_param.head_num = 1
                self_attention_param.kv_head_num = 1
            elif self.world_size == 16:
                self_attention_param.head_num = 3
                self_attention_param.kv_head_num = 3
            self_attention_param.qk_scale = 1.0 / math.sqrt(1.0 * 128)
            self_attention_param.input_layout = torch_atb.TYPE_BNSD
            self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
            self.self_attention = torch_atb.Operation(self_attention_param)

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
        is_joint = False
        if (joint_tensor_query is not None and 
            joint_tensor_key is not None and 
            joint_tensor_value is not None):
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
                )
            elif joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
                is_joint = True
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)
                is_joint = True
        elif (joint_tensor_query is None and 
            joint_tensor_key is None and 
            joint_tensor_value is None):
            pass
        else:
            raise ValueError(
                f"joint_tensor_query, joint_tensor_key, and joint_tensor_value should be None or not None simultaneously."
            )
        if is_joint:
            ulysses_world_size = dist.get_world_size(self.ulysses_pg)
            ulysses_rank = dist.get_rank(self.ulysses_pg)
            attn_heads_per_ulysses_rank = (
                joint_tensor_key.shape[-2] // ulysses_world_size
            )
            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]

        all_gather = False
        if self.args.video_size in self.video_size:
            all_gather = False
        else:
            all_gather = True
        if not all_gather:
            if self.world_size == 8:
                # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
                # scatter 2, gather 1
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                )
                key_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, key, self.scatter_idx, self.gather_idx
                )
                value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, value, self.scatter_idx, self.gather_idx
                )

                key_layer = torch.cat([key_layer, joint_tensor_key], dim=1).transpose(1, 2)
                value_layer = torch.cat([value_layer, joint_tensor_value], dim=1).transpose(1, 2)
                query_layer = query_layer.transpose(1, 2)
                query_layer_list = query_layer.split(1, dim=1)
                key_layer_list = key_layer.split(1, dim=1)
                value_layer_list = value_layer.split(1, dim=1)
                output = []
                for_loop = query_layer.shape[1]
                for i in range(for_loop):
                    if self.algo == 1:
                        seqlen = torch.tensor([[query_layer.shape[2]], [key_layer.shape[2]]], dtype=torch.int32)
                        intensors = [query_layer_list[i], key_layer_list[i], value_layer_list[i], seqlen]
                        out = self.self_attention.forward(intensors)[0]
                    elif self.algo == 0:
                        out = torch_npu.npu_fusion_attention(
                            query_layer_list[i],
                            key_layer_list[i],
                            value_layer_list[i],
                            head_num=1,
                            input_layout="BNSD",
                            scale=scale,
                            pre_tockens=MAX_TOKEN,
                            next_tockens=MAX_TOKEN
                        )[0]
                    output.append(out)
                out_concat = torch.cat(output, dim=1)
                out = out_concat.transpose(1, 2)
            elif self.world_size == 16:
                # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
                # scatter 2, gather 1
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                )
                key_value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, torch.cat((key, value), dim=0), self.scatter_idx, self.gather_idx
                )
                joint_tensor_key_value = torch.cat((joint_tensor_key, joint_tensor_value), dim=0)
                joint_tensor_key_value = joint_tensor_key_value.transpose(1, 2)

                b, s, n, d = key_value_layer.shape
                kv_full = torch.empty([2, b, s, n, d], dtype=query_layer.dtype, device=query_layer.device)
                dist.all_gather_into_tensor(kv_full, key_value_layer, group=self.ring_pg)

                kv_full = kv_full.permute(1, 3, 0, 2, 4).reshape(b, n, -1, d)
                kv_full = torch.cat((kv_full, joint_tensor_key_value), dim=2)

                query_layer = query_layer.transpose(1, 2)
                key_layer, value_layer = kv_full.chunk(2, dim=0)
                if self.algo == 1:
                    seqlen = torch.tensor([[query_layer.shape[2]], [key_layer.shape[2]]], dtype=torch.int32)
                    intensors = [query_layer, key_layer, value_layer, seqlen]
                    out = self.self_attention.forward(intensors)[0]
                elif self.algo == 0:
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
            if self.world_size == 8:
                output = all_to_all_v1(
                    query,
                    key,
                    value,
                    joint_tensor_key,
                    joint_tensor_value,
                    2,
                    1,
                    scale=scale,
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
                    scale=scale,
                    algo = self.algo,
                    self_attention = self.self_attention)
        # out e.g., [s/p::h]
        return output