# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist


_GLOBAL_BLOCK_PROFILER = None


class BlockProfiler:
    def __init__(
        self,
        enabled,
        device,
        rank,
        name="Inference",
        top_k=8,
        output_file=None,
    ):
        self.enabled = enabled
        self.device = device
        self.rank = rank
        self.name = name
        self.top_k = top_k
        self.output_file = output_file
        self.block_totals_ms = defaultdict(float)
        self.block_counts = defaultdict(int)
        self.block_max_call_ms = defaultdict(float)

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

    def stop(self, block_idx, start_time):
        if not self.enabled or start_time is None:
            return 0.0
        self._sync()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        block_name = f"block_{int(block_idx):02d}"
        self.block_totals_ms[block_name] += elapsed_ms
        self.block_counts[block_name] += 1
        if elapsed_ms > self.block_max_call_ms[block_name]:
            self.block_max_call_ms[block_name] = elapsed_ms
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
        if not self.enabled or not self.block_totals_ms:
            return

        summaries = []
        if self.rank == 0:
            logging.info(f"[{self.name} BlockProfile] ==== DiT Block Breakdown (ms) ====")

        for block_name in sorted(self.block_totals_ms):
            total_ms = self.block_totals_ms[block_name]
            count = self.block_counts[block_name]
            max_call_ms = self.block_max_call_ms[block_name]
            total_max, total_avg, total_sum, count_sum, max_call = self._reduce_stage(
                total_ms, count, max_call_ms
            )
            call_avg = total_sum / max(count_sum, 1.0)
            summaries.append((block_name, total_max, total_avg, call_avg, max_call, int(count_sum)))
            if self.rank == 0:
                logging.info(
                    f"[{self.name} BlockProfile] {block_name}: "
                    f"total={total_max:.3f}/{total_avg:.3f}, "
                    f"call_avg={call_avg:.3f}, call_max={max_call:.3f}, calls={int(count_sum)}"
                )

        top_summaries = []
        if summaries:
            total_max_sum = sum(item[1] for item in summaries)
            if total_max_sum > 0.0:
                sorted_blocks = sorted(summaries, key=lambda x: x[1], reverse=True)
                for rank_idx, item in enumerate(sorted_blocks[: self.top_k], start=1):
                    share = item[1] / total_max_sum
                    top_summaries.append((rank_idx, item[0], item[1], share))
                if self.rank == 0:
                    logging.info(f"[{self.name} BlockProfile] ==== DiT Block Hotspots (max_rank) ====")
                    for rank_idx, block_name, total_max, share in top_summaries:
                        logging.info(
                            f"[{self.name} BlockProfile] top{rank_idx}: "
                            f"{block_name} {total_max:.3f} ms ({share:.2%})"
                        )

        self._append_report_file(summaries, top_summaries)

    def _append_report_file(self, summaries, top_summaries):
        if self.rank != 0 or not self.output_file:
            return
        try:
            output_dir = os.path.dirname(os.path.abspath(self.output_file))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            world_size = dist.get_world_size() if dist.is_initialized() else 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"BLOCK_RUN,{ts},{self.name},world_size={world_size}\n")
                for block_name, total_max, total_avg, call_avg, max_call, calls in summaries:
                    f.write(
                        "BLOCK,"
                        f"{block_name},"
                        f"{total_max:.3f},{total_avg:.3f},"
                        f"{call_avg:.3f},{max_call:.3f},{calls}\n"
                    )
                for rank_idx, block_name, total_max, share in top_summaries:
                    f.write(f"BLOCK_TOP,{rank_idx},{block_name},{total_max:.3f},{share:.6f}\n")
                f.write("BLOCK_END\n")
        except Exception as exc:
            logging.warning(
                f"[{self.name} BlockProfile] failed to append profile file {self.output_file}: {exc}"
            )


def configure_block_profiler(enabled, device, rank, name="Inference", output_file=None):
    global _GLOBAL_BLOCK_PROFILER
    if enabled:
        _GLOBAL_BLOCK_PROFILER = BlockProfiler(
            enabled=True,
            device=device,
            rank=rank,
            name=name,
            output_file=output_file,
        )
    else:
        _GLOBAL_BLOCK_PROFILER = None


def get_block_profiler():
    return _GLOBAL_BLOCK_PROFILER


def report_block_profiler():
    global _GLOBAL_BLOCK_PROFILER
    if _GLOBAL_BLOCK_PROFILER is not None:
        _GLOBAL_BLOCK_PROFILER.report()
        _GLOBAL_BLOCK_PROFILER = None
