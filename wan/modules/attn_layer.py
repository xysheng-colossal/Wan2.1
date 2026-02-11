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

from ..distributed.parallel_mgr import get_sp_group
from ..distributed.comm import all_to_all_4D
from wan.utils.rainfusion import Rainfusion
from wan.utils.attention_profile import get_attention_profiler

from mindiesd import attention_forward

logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647
_QKV_A2A_MODE_LOGGED = False
_A2A_FENCE_MODE_LOGGED = False
_A2A_EVENT_GATE_LOGGED = False
_A2A_COMM_STREAM = {}


def _device_sync_if_needed(enabled):
    if not enabled:
        return
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_a2a_fence_mode():
    raw_mode = os.getenv("WAN_A2A_FENCE", "0").strip().lower()
    if raw_mode in ("0", "off", "false", ""):
        return 0, "disabled"
    if raw_mode in ("1", "pre_qkv", "pre_qkv_only", "light"):
        return 1, "pre_qkv_only"
    if raw_mode in ("2", "full", "all"):
        return 2, "full"
    return 0, f"invalid({raw_mode})->disabled"


def _resolve_a2a_event_gate():
    raw = os.getenv("WAN_A2A_EVENT_GATE", "0").strip().lower()
    if raw in ("0", "off", "false", ""):
        return False, "disabled"
    if raw in ("1", "on", "true", "gate"):
        return True, "enabled"
    return False, f"invalid({raw})->disabled"


def _get_device_api():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu
    if torch.cuda.is_available():
        return torch.cuda
    return None


def _get_a2a_comm_stream():
    device_api = _get_device_api()
    if device_api is None:
        return None
    device_idx = device_api.current_device()
    stream = _A2A_COMM_STREAM.get(device_idx)
    if stream is None:
        stream = device_api.Stream()
        _A2A_COMM_STREAM[device_idx] = stream
    return stream


def _all_to_all_with_event_gate(input_, scatter_idx, gather_idx, group, enable_event_gate):
    if not enable_event_gate:
        return all_to_all_4D(
            input_=input_,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=group,
        )

    device_api = _get_device_api()
    comm_stream = _get_a2a_comm_stream()
    if device_api is None or comm_stream is None:
        return all_to_all_4D(
            input_=input_,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=group,
        )

    current_stream = device_api.current_stream()
    ready_event = device_api.Event()
    done_event = device_api.Event()
    ready_event.record(current_stream)
    comm_stream.wait_event(ready_event)
    with device_api.stream(comm_stream):
        output = all_to_all_4D(
            input_=input_,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=group,
        )
        done_event.record(comm_stream)
    current_stream.wait_event(done_event)
    return output

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
        rainfusion_config=None,
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
        self.video_size = ['480*832', '832*480', '480*720', '720*480']

        self.algo = int(os.getenv('ALGO', 0))

        if self.args.size in self.video_size:
            self.use_all_head = True
        else:
            self.use_all_head = False
        
        self.ulysses_pg = get_sp_group().ulysses_group
        self.ring_pg = get_sp_group().ring_group

        self.rainfusion_config = rainfusion_config
        self.rainfusion_fa = None
        if self.rainfusion_config is not None:
            self.rainfusion_fa = Rainfusion(
                grid_size=rainfusion_config["grid_size"],
                skip_timesteps=rainfusion_config["skip_timesteps"],
                sparsity=rainfusion_config["sparsity"],
            )

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
        scale=None,
        t_idx=-1,
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

        profiler = get_attention_profiler()

        def profile_start():
            return profiler.start() if profiler is not None else None

        def profile_stop(stage_name, start_time):
            if profiler is not None:
                profiler.stop(stage_name, start_time, step_idx=t_idx)

        total_start = profile_start()

        use_packed_qkv_a2a = os.getenv("WAN_PACKED_QKV_A2A", "1") != "0"
        a2a_fence_mode, a2a_fence_mode_name = _resolve_a2a_fence_mode()
        a2a_event_gate_enabled, a2a_event_gate_name = _resolve_a2a_event_gate()
        fence_before_qkv = a2a_fence_mode >= 1
        fence_full = a2a_fence_mode >= 2
        global _QKV_A2A_MODE_LOGGED
        global _A2A_FENCE_MODE_LOGGED
        global _A2A_EVENT_GATE_LOGGED
        if not _QKV_A2A_MODE_LOGGED:
            if not dist.is_initialized() or dist.get_rank() == 0:
                mode = "packed_qkv_single_all_to_all" if use_packed_qkv_a2a else "legacy_qkv_three_all_to_all"
                logger.info(f"Attention all_to_all mode: {mode} (WAN_PACKED_QKV_A2A={'1' if use_packed_qkv_a2a else '0'})")
            _QKV_A2A_MODE_LOGGED = True
        if not _A2A_FENCE_MODE_LOGGED:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(
                    f"Attention all_to_all fence mode: {a2a_fence_mode_name} "
                    f"(WAN_A2A_FENCE={os.getenv('WAN_A2A_FENCE', '0')})"
                )
            _A2A_FENCE_MODE_LOGGED = True
        if not _A2A_EVENT_GATE_LOGGED:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(
                    f"Attention all_to_all event gate: {a2a_event_gate_name} "
                    f"(WAN_A2A_EVENT_GATE={os.getenv('WAN_A2A_EVENT_GATE', '0')})"
                )
            _A2A_EVENT_GATE_LOGGED = True
        if use_packed_qkv_a2a:
            _device_sync_if_needed(fence_before_qkv)
            t0 = profile_start()
            qkv = torch.cat([query, key, value], dim=-1)
            profile_stop("qkv_concat", t0)

            t0 = profile_start()
            qkv_layer = _all_to_all_with_event_gate(
                input_=qkv,
                scatter_idx=2,
                gather_idx=1,
                group=self.ulysses_pg,
                enable_event_gate=a2a_event_gate_enabled,
            )
            profile_stop("qkv_all_to_all", t0)

            t0 = profile_start()
            q_dim = query.shape[-1]
            k_dim = key.shape[-1]
            v_dim = value.shape[-1]
            query_layer, key_layer, value_layer = torch.split(
                qkv_layer, [q_dim, k_dim, v_dim], dim=-1
            )
            profile_stop("qkv_split", t0)
        else:
            _device_sync_if_needed(fence_before_qkv)
            t0 = profile_start()
            query_layer = _all_to_all_with_event_gate(
                input_=query,
                scatter_idx=2,
                gather_idx=1,
                group=self.ulysses_pg,
                enable_event_gate=a2a_event_gate_enabled,
            )
            profile_stop("q_all_to_all", t0)

            _device_sync_if_needed(fence_full)
            t0 = profile_start()
            key_layer = _all_to_all_with_event_gate(
                input_=key,
                scatter_idx=2,
                gather_idx=1,
                group=self.ulysses_pg,
                enable_event_gate=a2a_event_gate_enabled,
            )
            profile_stop("k_all_to_all", t0)

            _device_sync_if_needed(fence_full)
            t0 = profile_start()
            value_layer = _all_to_all_with_event_gate(
                input_=value,
                scatter_idx=2,
                gather_idx=1,
                group=self.ulysses_pg,
                enable_event_gate=a2a_event_gate_enabled,
            )
            profile_stop("v_all_to_all", t0)

        if get_sp_group().ring_world_size > 1:
            ring_size = get_sp_group().ring_world_size
            b, s, n, d = key_layer.shape
            k_full = torch.empty([ring_size, b, s, n, d], dtype=query_layer.dtype, device=query_layer.device)
            t0 = profile_start()
            dist.all_gather_into_tensor(k_full, key_layer, group=self.ring_pg)
            key_layer = k_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)
            profile_stop("kv_ring_k_all_gather", t0)

            v_full = torch.empty([ring_size, b, s, n, d], dtype=query_layer.dtype, device=query_layer.device)
            t0 = profile_start()
            dist.all_gather_into_tensor(v_full, value_layer, group=self.ring_pg)
            value_layer = v_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)
            profile_stop("kv_ring_v_all_gather", t0)

        t0 = profile_start()
        if self.rainfusion_config is not None:
            out = self.rainfusion_fa(
                query_layer,
                key_layer,
                value_layer,
                atten_mask_all=self.rainfusion_config["atten_mask_all"],
                text_len=0,
                t_idx=t_idx,
            )
        elif self.use_all_head:
            if self.algo == 0:
                out = attention_forward(query_layer, key_layer, value_layer,
                                        opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
            elif self.algo == 1:
                out = attention_forward(query_layer, key_layer, value_layer,
                                        opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
            else:
                raise ValueError(f"select flash attention algorithm only support 0, 1, but got {self.algo}")
        else:
            query_layer_list = query_layer.split(1, dim=2)
            key_layer_list = key_layer.split(1, dim=2)
            value_layer_list = value_layer.split(1, dim=2)
            output = []
            for_loop = query_layer.shape[2]
            for i in range(for_loop):
                if self.algo == 0:
                    out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                        opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
                elif self.algo == 1:
                    out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                        opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, but got f{self.algo}")

                output.append(out)
            out = torch.cat(output, dim=2)
        profile_stop("attn_kernel", t0)

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        _device_sync_if_needed(fence_full)
        t0 = profile_start()
        output = _all_to_all_with_event_gate(
            input_=context_layer,
            scatter_idx=1,
            gather_idx=2,
            group=self.ulysses_pg,
            enable_event_gate=a2a_event_gate_enabled,
        )
        profile_stop("out_all_to_all", t0)
        profile_stop("attn_forward_total", total_start)

        return output
