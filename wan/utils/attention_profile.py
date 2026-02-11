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
        self.top_k = 5
        self.stage_totals_ms = defaultdict(float)
        self.stage_counts = defaultdict(int)
        self.stage_max_call_ms = defaultdict(float)
        self.step_stage_totals_ms = defaultdict(lambda: defaultdict(float))

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

    def stop(self, stage_name, start_time, step_idx=None):
        if not self.enabled or start_time is None:
            return 0.0
        self._sync()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.stage_totals_ms[stage_name] += elapsed_ms
        self.stage_counts[stage_name] += 1
        if elapsed_ms > self.stage_max_call_ms[stage_name]:
            self.stage_max_call_ms[stage_name] = elapsed_ms
        if step_idx is not None:
            try:
                step_id = int(step_idx)
                if step_id >= 0:
                    self.step_stage_totals_ms[stage_name][step_id] += elapsed_ms
            except Exception:
                pass
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

    def _reduce_series(self, values):
        if len(values) == 0:
            return [], []
        if not dist.is_initialized():
            return values, values

        world_size = dist.get_world_size()
        value_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
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

    @staticmethod
    def _dense_step_series(step_map):
        if not step_map:
            return []
        max_step = int(max(step_map.keys()))
        if max_step < 0:
            return []
        dense = [0.0] * (max_step + 1)
        for step_idx, value in step_map.items():
            if step_idx < 0:
                continue
            dense[int(step_idx)] = float(value)
        return dense

    def report(self):
        if not self.enabled or not self.stage_totals_ms:
            return

        summaries = []
        summary_map = {}
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
            summary_map[stage_name] = (total_max, total_avg, call_avg, max_call, int(count_sum))
            if self.rank == 0:
                logging.info(
                    f"[{self.name} AttnProfile] {stage_name}: "
                    f"total={total_max:.3f}/{total_avg:.3f}, "
                    f"call_avg={call_avg:.3f}, call_max={max_call:.3f}, calls={int(count_sum)}"
                )

        ratio_summaries = self._build_ratio_summaries(summary_map)
        if self.rank == 0 and ratio_summaries:
            logging.info(f"[{self.name} AttnProfile] ==== Attention Ratios ====")
            for key, max_ratio, avg_ratio in ratio_summaries:
                logging.info(
                    f"[{self.name} AttnProfile] {key}: "
                    f"max_ratio={max_ratio:.2%}, avg_ratio={avg_ratio:.2%}"
                )

        step_summaries = self._build_step_summaries()
        if self.rank == 0 and step_summaries:
            logging.info(f"[{self.name} AttnProfile] ==== Attention Per-Step Breakdown (ms, max_rank) ====")
            for (
                metric_name,
                avg_ms,
                p50_ms,
                p90_ms,
                min_ms,
                max_ms,
                avg_rank_avg_ms,
                slowest,
            ) in step_summaries:
                logging.info(
                    f"[{self.name} AttnProfile] {metric_name}: "
                    f"avg={avg_ms:.3f}, p50={p50_ms:.3f}, p90={p90_ms:.3f}, "
                    f"min={min_ms:.3f}, max={max_ms:.3f}, avg_rank_avg={avg_rank_avg_ms:.3f}"
                )
                slowest_str = ", ".join([f"step{idx}:{val:.3f}" for idx, val in slowest])
                logging.info(
                    f"[{self.name} AttnProfile] {metric_name} slowest {self.top_k}: {slowest_str}"
                )

        self._append_report_file(summaries, ratio_summaries, step_summaries)

    @staticmethod
    def _sum_stage(summary_map, stage_names):
        total_max = 0.0
        total_avg = 0.0
        for name in stage_names:
            if name in summary_map:
                total_max += summary_map[name][0]
                total_avg += summary_map[name][1]
        return total_max, total_avg

    def _build_ratio_summaries(self, summary_map):
        if "attn_forward_total" not in summary_map:
            return []

        total_max, total_avg = summary_map["attn_forward_total"][0], summary_map["attn_forward_total"][1]
        if total_max <= 0 or total_avg <= 0:
            return []

        kernel_max, kernel_avg = self._sum_stage(summary_map, ["attn_kernel"])
        packed_comm_max, packed_comm_avg = self._sum_stage(
            summary_map,
            [
                "qkv_all_to_all",
                "kv_ring_k_all_gather",
                "kv_ring_v_all_gather",
                "out_all_to_all",
            ],
        )
        legacy_comm_max, legacy_comm_avg = self._sum_stage(
            summary_map,
            [
                "q_all_to_all",
                "k_all_to_all",
                "v_all_to_all",
                "kv_ring_k_all_gather",
                "kv_ring_v_all_gather",
                "out_all_to_all",
            ],
        )
        comm_max = packed_comm_max if packed_comm_max > 0 else legacy_comm_max
        comm_avg = packed_comm_avg if packed_comm_avg > 0 else legacy_comm_avg

        ratio_summaries = []
        ratio_summaries.append(("kernel_share", kernel_max / total_max, kernel_avg / total_avg))
        ratio_summaries.append(("comm_share", comm_max / total_max, comm_avg / total_avg))
        pack_overhead_max, pack_overhead_avg = self._sum_stage(
            summary_map, ["qkv_concat", "qkv_split"]
        )
        if pack_overhead_max > 0 or pack_overhead_avg > 0:
            ratio_summaries.append(
                ("pack_overhead_share", pack_overhead_max / total_max, pack_overhead_avg / total_avg)
            )

        uncovered_max = max(total_max - kernel_max - comm_max, 0.0)
        uncovered_avg = max(total_avg - kernel_avg - comm_avg, 0.0)
        ratio_summaries.append(
            ("uncovered_share", uncovered_max / total_max, uncovered_avg / total_avg)
        )
        return ratio_summaries

    def _build_step_summaries(self):
        if not self.step_stage_totals_ms:
            return []

        step_stage_maps = dict(self.step_stage_totals_ms)
        packed_comm_keys = [
            "qkv_all_to_all",
            "kv_ring_k_all_gather",
            "kv_ring_v_all_gather",
            "out_all_to_all",
        ]
        legacy_comm_keys = [
            "q_all_to_all",
            "k_all_to_all",
            "v_all_to_all",
            "kv_ring_k_all_gather",
            "kv_ring_v_all_gather",
            "out_all_to_all",
        ]
        comm_keys = packed_comm_keys if "qkv_all_to_all" in step_stage_maps else legacy_comm_keys
        comm_step_map = defaultdict(float)
        for name in comm_keys:
            step_map = step_stage_maps.get(name)
            if not step_map:
                continue
            for step_idx, value in step_map.items():
                comm_step_map[int(step_idx)] += float(value)
        if comm_step_map:
            step_stage_maps["comm_total"] = comm_step_map

        step_summaries = []
        for metric_name in sorted(step_stage_maps):
            dense_values = self._dense_step_series(step_stage_maps[metric_name])
            if not dense_values:
                continue
            max_series, avg_series = self._reduce_series(dense_values)
            if len(max_series) == 0:
                continue
            avg_ms = sum(max_series) / len(max_series)
            p50_ms = self._percentile(max_series, 0.50)
            p90_ms = self._percentile(max_series, 0.90)
            min_ms = min(max_series)
            max_ms = max(max_series)
            avg_rank_avg_ms = sum(avg_series) / len(avg_series)
            slowest = sorted(enumerate(max_series), key=lambda x: x[1], reverse=True)[: self.top_k]
            step_summaries.append(
                (
                    metric_name,
                    avg_ms,
                    p50_ms,
                    p90_ms,
                    min_ms,
                    max_ms,
                    avg_rank_avg_ms,
                    slowest,
                )
            )
        return step_summaries

    def _append_report_file(self, summaries, ratio_summaries, step_summaries):
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
                for key, max_ratio, avg_ratio in ratio_summaries:
                    f.write(f"ATTN_RATIO,{key},{max_ratio:.6f},{avg_ratio:.6f}\n")
                for (
                    metric_name,
                    avg_ms,
                    p50_ms,
                    p90_ms,
                    min_ms,
                    max_ms,
                    avg_rank_avg_ms,
                    slowest,
                ) in step_summaries:
                    f.write(
                        "ATTN_STEP,"
                        f"{metric_name},"
                        f"{avg_ms:.3f},{p50_ms:.3f},{p90_ms:.3f},"
                        f"{min_ms:.3f},{max_ms:.3f},{avg_rank_avg_ms:.3f}\n"
                    )
                    slowest_str = ";".join([f"step{idx}:{val:.3f}" for idx, val in slowest])
                    f.write(f"ATTN_SLOW,{metric_name},{slowest_str}\n")
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
