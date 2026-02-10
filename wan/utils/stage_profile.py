# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import time
from collections import defaultdict

import torch
import torch.distributed as dist


class StageProfiler:
    def __init__(self, enabled, device, rank, name="Inference", top_k=5):
        self.enabled = enabled
        self.device = device
        self.rank = rank
        self.name = name
        self.top_k = top_k
        self.stage_totals_ms = {}
        self.step_metrics_ms = defaultdict(list)

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

    def stop(self, stage_name, start_time, per_step=False):
        if not self.enabled or start_time is None:
            return 0.0
        self._sync()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if stage_name not in self.stage_totals_ms:
            self.stage_totals_ms[stage_name] = 0.0
        self.stage_totals_ms[stage_name] += elapsed_ms
        if per_step:
            self.step_metrics_ms[stage_name].append(elapsed_ms)
        return elapsed_ms

    def _reduce_scalar(self, value):
        if not dist.is_initialized():
            return value, value

        world_size = dist.get_world_size()
        value_tensor = torch.tensor([value], dtype=torch.float64, device=self.device)
        max_tensor = value_tensor.clone()
        sum_tensor = value_tensor.clone()
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        return max_tensor.item(), (sum_tensor / world_size).item()

    def _reduce_series(self, values):
        if len(values) == 0:
            return [], []
        if not dist.is_initialized():
            return values, values

        world_size = dist.get_world_size()
        value_tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        max_tensor = value_tensor.clone()
        sum_tensor = value_tensor.clone()
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        avg_tensor = sum_tensor / world_size
        return max_tensor.tolist(), avg_tensor.tolist()

    @staticmethod
    def _percentile(values, percentile):
        if len(values) == 0:
            return 0.0
        sorted_values = sorted(values)
        idx = int((len(sorted_values) - 1) * percentile)
        return sorted_values[idx]

    def report(self):
        if not self.enabled:
            return

        if len(self.stage_totals_ms) > 0:
            if self.rank == 0:
                logging.info(f"[{self.name} StageProfile] ==== Stage Totals (ms, max_rank / avg_rank) ====")
            for stage_name, total_ms in self.stage_totals_ms.items():
                max_ms, avg_ms = self._reduce_scalar(total_ms)
                if self.rank == 0:
                    logging.info(
                        f"[{self.name} StageProfile] {stage_name}: {max_ms:.3f} / {avg_ms:.3f}"
                    )

        if len(self.step_metrics_ms) == 0:
            return

        if self.rank == 0:
            logging.info(f"[{self.name} StageProfile] ==== Per-Step Breakdown (max_rank) ====")
        for metric_name, values in self.step_metrics_ms.items():
            max_series, avg_series = self._reduce_series(values)
            if len(max_series) == 0:
                continue
            if self.rank == 0:
                avg_ms = sum(max_series) / len(max_series)
                p50_ms = self._percentile(max_series, 0.50)
                p90_ms = self._percentile(max_series, 0.90)
                min_ms = min(max_series)
                max_ms = max(max_series)
                avg_rank_avg_ms = sum(avg_series) / len(avg_series)
                logging.info(
                    f"[{self.name} StageProfile] {metric_name}: "
                    f"avg={avg_ms:.3f}, p50={p50_ms:.3f}, p90={p90_ms:.3f}, "
                    f"min={min_ms:.3f}, max={max_ms:.3f}, avg_rank_avg={avg_rank_avg_ms:.3f}"
                )

                slowest = sorted(
                    enumerate(max_series), key=lambda x: x[1], reverse=True
                )[: self.top_k]
                slowest_str = ", ".join(
                    [f"step{idx}:{val:.3f}" for idx, val in slowest]
                )
                logging.info(
                    f"[{self.name} StageProfile] {metric_name} slowest {self.top_k}: {slowest_str}"
                )
