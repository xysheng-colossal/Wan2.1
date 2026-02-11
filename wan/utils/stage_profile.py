# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist


class StageProfiler:
    def __init__(
        self,
        enabled,
        device,
        rank,
        name="Inference",
        top_k=5,
        output_file=None,
        run_info=None,
    ):
        self.enabled = enabled
        self.device = device
        self.rank = rank
        self.name = name
        self.top_k = top_k
        self.output_file = output_file
        self.run_info = dict(run_info or {})
        self.stage_totals_ms = defaultdict(float)
        self.step_metrics_ms = defaultdict(list)
        self.stage_peak_alloc_bytes = defaultdict(float)
        self.stage_peak_reserved_bytes = defaultdict(float)
        self.stage_peak_delta_alloc_bytes = defaultdict(float)
        self.request_peak_alloc_bytes = 0.0
        self.request_peak_reserved_bytes = 0.0
        self._reset_peak_memory_stats()

    def _sync(self):
        if not self.enabled:
            return
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_memory_backend(self):
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu
        if torch.cuda.is_available():
            return torch.cuda
        return None

    def _get_device_index(self):
        device = self.device
        if isinstance(device, int):
            return device
        if isinstance(device, torch.device):
            if device.index is not None:
                return int(device.index)
            return None
        if isinstance(device, str):
            if ":" in device:
                try:
                    return int(device.split(":")[-1])
                except Exception:
                    return None
            return None
        return None

    def _iter_device_args(self):
        args = []
        device_index = self._get_device_index()
        if device_index is not None:
            args.append((device_index,))
        if isinstance(self.device, torch.device):
            args.append((self.device,))
        args.append(tuple())
        seen = set()
        for item in args:
            if item not in seen:
                seen.add(item)
                yield item

    def _memory_call(self, fn_name):
        backend = self._get_memory_backend()
        if backend is None:
            return 0.0
        fn = getattr(backend, fn_name, None)
        if fn is None:
            return 0.0
        for args in self._iter_device_args():
            try:
                return float(fn(*args))
            except Exception:
                continue
        return 0.0

    def _memory_stats(self):
        backend = self._get_memory_backend()
        if backend is None:
            return None
        stats_fn = getattr(backend, "memory_stats", None)
        if not callable(stats_fn):
            return None
        for args in self._iter_device_args():
            try:
                stats = stats_fn(*args)
                if isinstance(stats, dict):
                    return stats
            except Exception:
                continue
        return None

    @staticmethod
    def _stats_value(stats, keys):
        if not stats:
            return 0.0
        for key in keys:
            value = stats.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return 0.0

    def _get_runtime_memory_bytes(self):
        stats = self._memory_stats()
        alloc = self._stats_value(
            stats,
            [
                "allocated_bytes.all.current",
                "allocated_bytes.current",
                "active_bytes.all.current",
                "active_bytes.current",
            ],
        )
        reserved = self._stats_value(
            stats,
            [
                "reserved_bytes.all.current",
                "reserved_bytes.current",
                "segment_bytes.all.current",
                "segment_bytes.current",
            ],
        )
        if alloc <= 0.0:
            alloc = self._memory_call("memory_allocated")
        if reserved <= 0.0:
            reserved = self._memory_call("memory_reserved")
        if reserved <= 0.0:
            reserved = self._memory_call("memory_cached")
        if reserved <= 0.0:
            reserved = alloc
        return alloc, reserved

    def _get_peak_memory_bytes(self):
        stats = self._memory_stats()
        peak_alloc = self._stats_value(
            stats,
            [
                "allocated_bytes.all.peak",
                "allocated_bytes.peak",
                "active_bytes.all.peak",
                "active_bytes.peak",
            ],
        )
        peak_reserved = self._stats_value(
            stats,
            [
                "reserved_bytes.all.peak",
                "reserved_bytes.peak",
                "segment_bytes.all.peak",
                "segment_bytes.peak",
            ],
        )
        if peak_alloc <= 0.0:
            peak_alloc = self._memory_call("max_memory_allocated")
        if peak_reserved <= 0.0:
            peak_reserved = self._memory_call("max_memory_reserved")
        if peak_reserved <= 0.0:
            peak_reserved = self._memory_call("max_memory_cached")
        if peak_reserved <= 0.0:
            _, reserved = self._get_runtime_memory_bytes()
            peak_reserved = reserved
        return peak_alloc, peak_reserved

    def _reset_peak_memory_stats(self):
        if not self.enabled:
            return
        backend = self._get_memory_backend()
        if backend is None:
            return
        reset_fn = getattr(backend, "reset_peak_memory_stats", None)
        if callable(reset_fn):
            for args in self._iter_device_args():
                try:
                    reset_fn(*args)
                    return
                except Exception:
                    continue
        reset_alloc = getattr(backend, "reset_max_memory_allocated", None)
        if callable(reset_alloc):
            for args in self._iter_device_args():
                try:
                    reset_alloc(*args)
                    break
                except Exception:
                    continue
        reset_reserved = getattr(backend, "reset_max_memory_reserved", None)
        if callable(reset_reserved):
            for args in self._iter_device_args():
                try:
                    reset_reserved(*args)
                    break
                except Exception:
                    continue

    def start(self):
        if not self.enabled:
            return None
        self._sync()
        alloc, reserved = self._get_runtime_memory_bytes()
        return {
            "ts": time.perf_counter(),
            "alloc": alloc,
            "reserved": reserved,
        }

    def stop(self, stage_name, start_time, per_step=False):
        if not self.enabled or start_time is None:
            return 0.0
        self._sync()
        if isinstance(start_time, dict):
            start_ts = start_time.get("ts")
            start_alloc = start_time.get("alloc", 0.0)
        else:
            start_ts = start_time
            start_alloc = None

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        self.stage_totals_ms[stage_name] += elapsed_ms
        if per_step:
            self.step_metrics_ms[stage_name].append(elapsed_ms)

        cur_alloc, cur_reserved = self._get_runtime_memory_bytes()
        self.stage_peak_alloc_bytes[stage_name] = max(
            self.stage_peak_alloc_bytes[stage_name], cur_alloc
        )
        self.stage_peak_reserved_bytes[stage_name] = max(
            self.stage_peak_reserved_bytes[stage_name], cur_reserved
        )
        if start_alloc is not None:
            self.stage_peak_delta_alloc_bytes[stage_name] = max(
                self.stage_peak_delta_alloc_bytes[stage_name],
                max(cur_alloc - start_alloc, 0.0),
            )

        peak_alloc, peak_reserved = self._get_peak_memory_bytes()
        self.request_peak_alloc_bytes = max(self.request_peak_alloc_bytes, peak_alloc)
        self.request_peak_reserved_bytes = max(self.request_peak_reserved_bytes, peak_reserved)
        return elapsed_ms

    def _reduce_scalar(self, value):
        if not dist.is_initialized():
            return value, value

        world_size = dist.get_world_size()
        value_tensor = torch.tensor([value], dtype=torch.float32, device=self.device)
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
    def _bytes_to_mb(value):
        return value / (1024.0 * 1024.0)

    @staticmethod
    def _to_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    def update_run_info(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                self.run_info[key] = value

    @staticmethod
    def _safe_kv_value(value):
        text = str(value)
        return text.replace(",", "_")

    def report(self):
        if not self.enabled:
            return

        stage_summaries = []
        step_summaries = []
        memory_summaries = []
        request_memory_summary = None
        perf_summaries = []
        hotspot_summaries = []

        if self.stage_totals_ms:
            if self.rank == 0:
                logging.info(f"[{self.name} StageProfile] ==== Stage Totals (ms, max_rank / avg_rank) ====")
            for stage_name in sorted(self.stage_totals_ms):
                total_ms = self.stage_totals_ms[stage_name]
                max_ms, avg_ms = self._reduce_scalar(total_ms)
                stage_summaries.append((stage_name, max_ms, avg_ms))
                if self.rank == 0:
                    logging.info(
                        f"[{self.name} StageProfile] {stage_name}: {max_ms:.3f} / {avg_ms:.3f}"
                    )

        memory_stage_names = sorted(
            set(self.stage_totals_ms.keys()) | set(self.stage_peak_alloc_bytes.keys())
        )
        if memory_stage_names:
            if self.rank == 0:
                logging.info(f"[{self.name} StageProfile] ==== Stage Memory (MB, max_rank / avg_rank) ====")
            for stage_name in memory_stage_names:
                alloc_peak_mb = self._bytes_to_mb(self.stage_peak_alloc_bytes.get(stage_name, 0.0))
                reserved_peak_mb = self._bytes_to_mb(
                    self.stage_peak_reserved_bytes.get(stage_name, 0.0)
                )
                delta_alloc_peak_mb = self._bytes_to_mb(
                    self.stage_peak_delta_alloc_bytes.get(stage_name, 0.0)
                )
                alloc_peak_max, alloc_peak_avg = self._reduce_scalar(alloc_peak_mb)
                reserved_peak_max, reserved_peak_avg = self._reduce_scalar(reserved_peak_mb)
                delta_alloc_peak_max, delta_alloc_peak_avg = self._reduce_scalar(
                    delta_alloc_peak_mb
                )
                memory_summaries.append(
                    (
                        stage_name,
                        alloc_peak_max,
                        alloc_peak_avg,
                        reserved_peak_max,
                        reserved_peak_avg,
                        delta_alloc_peak_max,
                        delta_alloc_peak_avg,
                    )
                )
                if self.rank == 0:
                    logging.info(
                        f"[{self.name} StageProfile] {stage_name}: "
                        f"alloc_peak={alloc_peak_max:.1f}/{alloc_peak_avg:.1f}, "
                        f"reserved_peak={reserved_peak_max:.1f}/{reserved_peak_avg:.1f}, "
                        f"delta_alloc_peak={delta_alloc_peak_max:.1f}/{delta_alloc_peak_avg:.1f}"
                    )

        req_peak_alloc_mb = self._bytes_to_mb(self.request_peak_alloc_bytes)
        req_peak_reserved_mb = self._bytes_to_mb(self.request_peak_reserved_bytes)
        req_peak_alloc_max, req_peak_alloc_avg = self._reduce_scalar(req_peak_alloc_mb)
        req_peak_reserved_max, req_peak_reserved_avg = self._reduce_scalar(req_peak_reserved_mb)
        request_memory_summary = (
            req_peak_alloc_max,
            req_peak_alloc_avg,
            req_peak_reserved_max,
            req_peak_reserved_avg,
        )
        if self.rank == 0:
            logging.info(
                f"[{self.name} StageProfile] request_peak_mem: "
                f"alloc={req_peak_alloc_max:.1f}/{req_peak_alloc_avg:.1f}, "
                f"reserved={req_peak_reserved_max:.1f}/{req_peak_reserved_avg:.1f} MB"
            )

        stage_max_map = {stage_name: max_ms for stage_name, max_ms, _ in stage_summaries}
        request_total_ms = stage_max_map.get("request_total", 0.0)
        denoise_loop_ms = stage_max_map.get("denoise_loop_total", 0.0)
        if request_total_ms > 0.0 and denoise_loop_ms > 0.0:
            perf_summaries.append(("denoise_share_pct", denoise_loop_ms / request_total_ms * 100.0, "pct"))

        if "denoise_step_total" in self.step_metrics_ms:
            max_series, _ = self._reduce_series(self.step_metrics_ms["denoise_step_total"])
            if len(max_series) > 0:
                denoise_step_avg_ms = sum(max_series) / len(max_series)
                perf_summaries.append(("denoise_step_avg_ms", denoise_step_avg_ms, "ms"))
                if denoise_step_avg_ms > 0.0:
                    perf_summaries.append(("denoise_steps_per_s", 1000.0 / denoise_step_avg_ms, "step/s"))

        frame_num = self._to_float(self.run_info.get("frame_num"), default=0.0)
        width = self._to_float(self.run_info.get("width"), default=0.0)
        height = self._to_float(self.run_info.get("height"), default=0.0)
        if request_total_ms > 0.0 and frame_num > 0.0:
            request_total_s = request_total_ms / 1000.0
            perf_summaries.append(("frames_per_s", frame_num / request_total_s, "frame/s"))
            if width > 0.0 and height > 0.0:
                megapixel_total = frame_num * width * height / 1_000_000.0
                perf_summaries.append(("megapixel_per_s", megapixel_total / request_total_s, "MP/s"))

        if request_total_ms > 0.0:
            hot_candidates = [
                (stage_name, max_ms)
                for stage_name, max_ms, _ in stage_summaries
                if stage_name != "request_total"
            ]
            hot_candidates.sort(key=lambda item: item[1], reverse=True)
            for idx, (stage_name, max_ms) in enumerate(hot_candidates[: self.top_k], start=1):
                hotspot_summaries.append((idx, stage_name, max_ms, max_ms / request_total_ms * 100.0))

        if self.rank == 0 and perf_summaries:
            logging.info(f"[{self.name} StageProfile] ==== Derived Perf Metrics (max_rank) ====")
            for metric_name, metric_value, metric_unit in perf_summaries:
                logging.info(
                    f"[{self.name} StageProfile] {metric_name}: {metric_value:.3f} {metric_unit}"
                )

        if self.rank == 0 and hotspot_summaries:
            logging.info(f"[{self.name} StageProfile] ==== Stage Hotspots (max_rank) ====")
            for rank_idx, stage_name, stage_ms, share_pct in hotspot_summaries:
                logging.info(
                    f"[{self.name} StageProfile] top{rank_idx}: "
                    f"{stage_name} {stage_ms:.3f} ms ({share_pct:.2f}%)"
                )

        if not self.step_metrics_ms:
            self._append_report_file(
                stage_summaries,
                step_summaries,
                memory_summaries,
                request_memory_summary,
                perf_summaries,
                hotspot_summaries,
            )
            return

        if self.rank == 0:
            logging.info(f"[{self.name} StageProfile] ==== Per-Step Breakdown (max_rank) ====")
        for metric_name in sorted(self.step_metrics_ms):
            values = self.step_metrics_ms[metric_name]
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
                slowest = sorted(
                    enumerate(max_series), key=lambda x: x[1], reverse=True
                )[: self.top_k]
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
                logging.info(
                    f"[{self.name} StageProfile] {metric_name}: "
                    f"avg={avg_ms:.3f}, p50={p50_ms:.3f}, p90={p90_ms:.3f}, "
                    f"min={min_ms:.3f}, max={max_ms:.3f}, avg_rank_avg={avg_rank_avg_ms:.3f}"
                )

                slowest_str = ", ".join(
                    [f"step{idx}:{val:.3f}" for idx, val in slowest]
                )
                logging.info(
                    f"[{self.name} StageProfile] {metric_name} slowest {self.top_k}: {slowest_str}"
                )

        self._append_report_file(
            stage_summaries,
            step_summaries,
            memory_summaries,
            request_memory_summary,
            perf_summaries,
            hotspot_summaries,
        )

    def _append_report_file(
        self,
        stage_summaries,
        step_summaries,
        memory_summaries,
        request_memory_summary,
        perf_summaries,
        hotspot_summaries,
    ):
        if self.rank != 0 or not self.output_file:
            return
        try:
            output_dir = os.path.dirname(os.path.abspath(self.output_file))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            world_size = dist.get_world_size() if dist.is_initialized() else 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            run_info_fields = []
            for key in sorted(self.run_info.keys()):
                run_info_fields.append(f"{key}={self._safe_kv_value(self.run_info[key])}")
            run_info_suffix = "," + ",".join(run_info_fields) if run_info_fields else ""
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"RUN,{ts},{self.name},world_size={world_size}{run_info_suffix}\n")
                for stage_name, max_ms, avg_ms in stage_summaries:
                    f.write(f"TOTAL,{stage_name},{max_ms:.3f},{avg_ms:.3f}\n")
                for (
                    stage_name,
                    alloc_peak_max,
                    alloc_peak_avg,
                    reserved_peak_max,
                    reserved_peak_avg,
                    delta_alloc_peak_max,
                    delta_alloc_peak_avg,
                ) in memory_summaries:
                    f.write(
                        "MEM,"
                        f"{stage_name},"
                        f"{alloc_peak_max:.3f},{alloc_peak_avg:.3f},"
                        f"{reserved_peak_max:.3f},{reserved_peak_avg:.3f},"
                        f"{delta_alloc_peak_max:.3f},{delta_alloc_peak_avg:.3f}\n"
                    )
                if request_memory_summary is not None:
                    (
                        req_peak_alloc_max,
                        req_peak_alloc_avg,
                        req_peak_reserved_max,
                        req_peak_reserved_avg,
                    ) = request_memory_summary
                    f.write(
                        "MEM_REQUEST,"
                        f"{req_peak_alloc_max:.3f},{req_peak_alloc_avg:.3f},"
                        f"{req_peak_reserved_max:.3f},{req_peak_reserved_avg:.3f}\n"
                    )
                for metric_name, metric_value, metric_unit in perf_summaries:
                    f.write(f"PERF,{metric_name},{metric_value:.6f},{metric_unit}\n")
                for hot_rank, stage_name, stage_ms, stage_share_pct in hotspot_summaries:
                    f.write(
                        "HOT,"
                        f"{hot_rank},{stage_name},{stage_ms:.3f},{stage_share_pct:.3f}\n"
                    )
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
                        "STEP,"
                        f"{metric_name},"
                        f"{avg_ms:.3f},{p50_ms:.3f},{p90_ms:.3f},"
                        f"{min_ms:.3f},{max_ms:.3f},{avg_rank_avg_ms:.3f}\n"
                    )
                    slowest_str = ";".join(
                        [f"step{idx}:{val:.3f}" for idx, val in slowest]
                    )
                    f.write(f"SLOW,{metric_name},{slowest_str}\n")
                f.write("END\n")
        except Exception as exc:
            logging.warning(
                f"[{self.name} StageProfile] failed to append profile file {self.output_file}: {exc}"
            )
