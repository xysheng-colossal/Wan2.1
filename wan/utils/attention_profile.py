# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist


_GLOBAL_ATTN_PROFILER = None


class AttentionProfiler:
    def __init__(self, enabled, device, rank, name="Inference", output_file=None):
        self.enabled = enabled
        self.device = device
        self.rank = rank
        self.name = name
        self.output_file = output_file
        self.stage_totals_ms = defaultdict(float)
        self.stage_counts = defaultdict(int)
        self.stage_max_call_ms = defaultdict(float)

    def _sync(self):
        if not self.enabled:
            return
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self):
        if not self.enabled:
            return None
        self._sync()
        return time.perf_counter()

    def stop(self, stage_name, start_time):
        if not self.enabled or start_time is None:
            return 0.0
        self._sync()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.stage_totals_ms[stage_name] += elapsed_ms
        self.stage_counts[stage_name] += 1
        if elapsed_ms > self.stage_max_call_ms[stage_name]:
            self.stage_max_call_ms[stage_name] = elapsed_ms
        return elapsed_ms

    def _reduce_stage(self, total_ms, count, max_call_ms):
        if not dist.is_initialized():
            return total_ms, total_ms, total_ms, float(count), max_call_ms

        world_size = dist.get_world_size()
        total_tensor = torch.tensor([total_ms], dtype=torch.float32, device=self.device)
        total_max = total_tensor.clone()
        total_sum = total_tensor.clone()
        dist.all_reduce(total_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)

        count_tensor = torch.tensor([float(count)], dtype=torch.float32, device=self.device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        max_call_tensor = torch.tensor([max_call_ms], dtype=torch.float32, device=self.device)
        dist.all_reduce(max_call_tensor, op=dist.ReduceOp.MAX)

        return (
            total_max.item(),
            (total_sum / world_size).item(),
            total_sum.item(),
            count_tensor.item(),
            max_call_tensor.item(),
        )

    def report(self):
        if not self.enabled or not self.stage_totals_ms:
            return

        summaries = []
        if self.rank == 0:
            logging.info(f"[{self.name} AttnProfile] ==== Attention Internal Breakdown (ms) ====")

        for stage_name in sorted(self.stage_totals_ms):
            total_ms = self.stage_totals_ms[stage_name]
            count = self.stage_counts[stage_name]
            max_call_ms = self.stage_max_call_ms[stage_name]
            total_max, total_avg, total_sum, count_sum, max_call = self._reduce_stage(
                total_ms, count, max_call_ms
            )
            call_avg = total_sum / max(count_sum, 1.0)
            summaries.append((stage_name, total_max, total_avg, call_avg, max_call, int(count_sum)))
            if self.rank == 0:
                logging.info(
                    f"[{self.name} AttnProfile] {stage_name}: "
                    f"total={total_max:.3f}/{total_avg:.3f}, "
                    f"call_avg={call_avg:.3f}, call_max={max_call:.3f}, calls={int(count_sum)}"
                )

        self._append_report_file(summaries)

    def _append_report_file(self, summaries):
        if self.rank != 0 or not self.output_file:
            return
        try:
            output_dir = os.path.dirname(os.path.abspath(self.output_file))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            world_size = dist.get_world_size() if dist.is_initialized() else 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"ATTN_RUN,{ts},{self.name},world_size={world_size}\n")
                for stage_name, total_max, total_avg, call_avg, max_call, calls in summaries:
                    f.write(
                        "ATTN,"
                        f"{stage_name},"
                        f"{total_max:.3f},{total_avg:.3f},"
                        f"{call_avg:.3f},{max_call:.3f},{calls}\n"
                    )
                f.write("ATTN_END\n")
        except Exception as exc:
            logging.warning(
                f"[{self.name} AttnProfile] failed to append profile file {self.output_file}: {exc}"
            )


def configure_attention_profiler(enabled, device, rank, name="Inference", output_file=None):
    global _GLOBAL_ATTN_PROFILER
    if enabled:
        _GLOBAL_ATTN_PROFILER = AttentionProfiler(
            enabled=True,
            device=device,
            rank=rank,
            name=name,
            output_file=output_file,
        )
    else:
        _GLOBAL_ATTN_PROFILER = None


def get_attention_profiler():
    return _GLOBAL_ATTN_PROFILER


def report_attention_profiler():
    global _GLOBAL_ATTN_PROFILER
    if _GLOBAL_ATTN_PROFILER is not None:
        _GLOBAL_ATTN_PROFILER.report()
        _GLOBAL_ATTN_PROFILER = None
