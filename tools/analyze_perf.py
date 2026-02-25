#!/usr/bin/env python3
import argparse
from pathlib import Path


def _parse_run_header(parts):
    meta = {}
    for field in parts[3:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        meta[key] = value
    return meta


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def parse_stage_runs(lines):
    runs = []
    current = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        record = parts[0]
        if record == "RUN":
            if current is not None:
                runs.append(current)
            current = {
                "ts": parts[1] if len(parts) > 1 else "",
                "name": parts[2] if len(parts) > 2 else "",
                "meta": _parse_run_header(parts),
                "total": {},
                "step": {},
                "perf": {},
                "hot": [],
            }
            continue
        if current is None:
            continue
        if record == "TOTAL" and len(parts) >= 4:
            current["total"][parts[1]] = {
                "max_ms": _to_float(parts[2]),
                "avg_ms": _to_float(parts[3]),
            }
        elif record == "STEP" and len(parts) >= 8:
            current["step"][parts[1]] = {
                "avg_ms": _to_float(parts[2]),
                "p50_ms": _to_float(parts[3]),
                "p90_ms": _to_float(parts[4]),
                "min_ms": _to_float(parts[5]),
                "max_ms": _to_float(parts[6]),
                "avg_rank_avg_ms": _to_float(parts[7]),
            }
        elif record == "PERF" and len(parts) >= 4:
            current["perf"][parts[1]] = {
                "value": _to_float(parts[2]),
                "unit": parts[3],
            }
        elif record == "HOT" and len(parts) >= 5:
            current["hot"].append(
                {
                    "rank": int(_to_float(parts[1], 0)),
                    "stage": parts[2],
                    "ms": _to_float(parts[3]),
                    "share_pct": _to_float(parts[4]),
                }
            )
        elif record == "END":
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)
    return runs


def parse_attn_runs(lines):
    runs = []
    current = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        record = parts[0]
        if record == "ATTN_RUN":
            if current is not None:
                runs.append(current)
            current = {
                "ts": parts[1] if len(parts) > 1 else "",
                "name": parts[2] if len(parts) > 2 else "",
                "ratio": {},
                "step": {},
            }
            continue
        if current is None:
            continue
        if record == "ATTN_RATIO" and len(parts) >= 4:
            current["ratio"][parts[1]] = {
                "max": _to_float(parts[2]),
                "avg": _to_float(parts[3]),
            }
        elif record == "ATTN_STEP" and len(parts) >= 8:
            current["step"][parts[1]] = {
                "avg_ms": _to_float(parts[2]),
                "p50_ms": _to_float(parts[3]),
                "p90_ms": _to_float(parts[4]),
                "min_ms": _to_float(parts[5]),
                "max_ms": _to_float(parts[6]),
                "avg_rank_avg_ms": _to_float(parts[7]),
            }
        elif record == "ATTN_END":
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)
    return runs


def parse_block_runs(lines):
    runs = []
    current = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        record = parts[0]
        if record == "BLOCK_RUN":
            if current is not None:
                runs.append(current)
            current = {
                "ts": parts[1] if len(parts) > 1 else "",
                "name": parts[2] if len(parts) > 2 else "",
                "blocks": [],
                "top": [],
            }
            continue
        if current is None:
            continue
        if record == "BLOCK" and len(parts) >= 7:
            current["blocks"].append(
                {
                    "name": parts[1],
                    "total_max_ms": _to_float(parts[2]),
                    "total_avg_ms": _to_float(parts[3]),
                    "call_avg_ms": _to_float(parts[4]),
                    "call_max_ms": _to_float(parts[5]),
                    "calls": int(_to_float(parts[6], 0)),
                }
            )
        elif record == "BLOCK_TOP" and len(parts) >= 5:
            current["top"].append(
                {
                    "rank": int(_to_float(parts[1], 0)),
                    "name": parts[2],
                    "total_max_ms": _to_float(parts[3]),
                    "share": _to_float(parts[4]),
                }
            )
        elif record == "BLOCK_END":
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)
    return runs


def fmt_delta(new_v, old_v):
    if old_v == 0:
        return "n/a"
    delta_pct = (new_v - old_v) / old_v * 100.0
    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.2f}%"


def summarize_run(run, attn_run=None, block_run=None):
    print(f"Run: {run['ts']}  name={run['name']}")
    if run["meta"]:
        meta_text = ", ".join(f"{k}={v}" for k, v in sorted(run["meta"].items()))
        print(f"Meta: {meta_text}")

    request_total = run["total"].get("request_total", {}).get("max_ms", 0.0)
    denoise_total = run["total"].get("denoise_loop_total", {}).get("max_ms", 0.0)
    print(f"request_total(max_rank): {request_total:.3f} ms")
    print(f"denoise_loop_total(max_rank): {denoise_total:.3f} ms")

    print("Top hotspots:")
    hot_list = run["hot"]
    if not hot_list:
        fallback = [
            (name, value["max_ms"])
            for name, value in run["total"].items()
            if name != "request_total"
        ]
        fallback.sort(key=lambda x: x[1], reverse=True)
        hot_list = [
            {"rank": idx + 1, "stage": name, "ms": ms, "share_pct": 0.0}
            for idx, (name, ms) in enumerate(fallback[:5])
        ]
    for item in sorted(hot_list, key=lambda x: x["rank"])[:5]:
        suffix = f", share={item['share_pct']:.2f}%" if item["share_pct"] > 0 else ""
        print(f"  {item['rank']}. {item['stage']}: {item['ms']:.3f} ms{suffix}")

    if run["perf"]:
        print("Derived perf:")
        for key in sorted(run["perf"]):
            metric = run["perf"][key]
            print(f"  {key}: {metric['value']:.6f} {metric['unit']}")

    if attn_run and attn_run.get("step"):
        print("Attention step jitter:")
        for metric_name in ("comm_total", "attn_kernel", "attn_forward_total"):
            metric = attn_run["step"].get(metric_name)
            if not metric:
                continue
            p50_ms = metric.get("p50_ms", 0.0)
            p90_ms = metric.get("p90_ms", 0.0)
            jitter_pct = (p90_ms / p50_ms - 1.0) * 100.0 if p50_ms > 0.0 else 0.0
            print(
                f"  {metric_name}: avg={metric['avg_ms']:.3f} ms, "
                f"p50={p50_ms:.3f} ms, p90={p90_ms:.3f} ms, jitter={jitter_pct:.2f}%"
            )

    if block_run and (block_run.get("top") or block_run.get("blocks")):
        print("Top DiT blocks:")
        block_top = sorted(block_run.get("top", []), key=lambda x: x["rank"])
        if not block_top:
            fallback = sorted(
                block_run.get("blocks", []),
                key=lambda x: x["total_max_ms"],
                reverse=True,
            )[:5]
            block_top = [
                {
                    "rank": idx + 1,
                    "name": item["name"],
                    "total_max_ms": item["total_max_ms"],
                    "share": 0.0,
                }
                for idx, item in enumerate(fallback)
            ]
        for item in block_top[:5]:
            suffix = f", share={item['share']:.2%}" if item["share"] > 0 else ""
            print(f"  {item['rank']}. {item['name']}: {item['total_max_ms']:.3f} ms{suffix}")


def get_metric_value(run, key):
    if key in run["perf"]:
        return run["perf"][key]["value"]
    if key in run["total"]:
        return run["total"][key]["max_ms"]
    return 0.0


def print_comparison(current, prev):
    print("\nComparison vs previous run:")
    for key in ["request_total", "denoise_loop_total", "denoise_steps_per_s", "frames_per_s"]:
        cur_v = get_metric_value(current, key)
        prev_v = get_metric_value(prev, key)
        unit = "ms" if key in ("request_total", "denoise_loop_total") else ""
        print(f"  {key}: {cur_v:.6f}{unit} ({fmt_delta(cur_v, prev_v)})")


def build_suggestions(stage_run, attn_run, block_run):
    suggestions = []
    total = stage_run["total"]
    perf = stage_run["perf"]
    meta = stage_run.get("meta", {})
    request_total = total.get("request_total", {}).get("max_ms", 0.0)
    if request_total <= 0:
        return suggestions

    cfg_fused_forward = str(meta.get("cfg_fused_forward", "0")) == "1"
    dit_cond = total.get("dit_forward_cond", {}).get("max_ms", 0.0)
    dit_uncond = total.get("dit_forward_uncond", {}).get("max_ms", 0.0)
    if (not cfg_fused_forward) and dit_cond > 0 and dit_uncond > 0:
        cond_share = dit_cond / request_total
        uncond_share = dit_uncond / request_total
        if cond_share > 0.2 and uncond_share > 0.2:
            suggestions.append(
                "检测到cond/uncond双前向都是热点，可A/B测试 --cfg_fused_forward 以单次批量前向替代双次前向。"
            )

    cfg_allgather = total.get("cfg_allgather", {}).get("max_ms", 0.0)
    if cfg_allgather / request_total > 0.08:
        suggestions.append(
            "cfg_allgather占比较高，优先检查cfg并行组拓扑与跨机通信；对比开启/关闭serialize_comm与cfg_size=1/2的收益。"
        )

    scheduler_step = total.get("scheduler_step", {}).get("max_ms", 0.0)
    if scheduler_step / request_total > 0.08:
        suggestions.append(
            "scheduler_step占比较高，可A/B对比sample_solver(unipc vs dpm++)与sample_steps，找最优质量/速度拐点。"
        )

    t5_move = total.get("t5_to_device", {}).get("max_ms", 0.0) + total.get("t5_offload_cpu", {}).get("max_ms", 0.0)
    if t5_move / request_total > 0.03:
        suggestions.append(
            "T5搬运成本明显，可在多卡场景固定驻留设备端（关闭T5频繁offload）并复测显存余量。"
        )

    if "denoise_share_pct" in perf and perf["denoise_share_pct"]["value"] > 90.0:
        suggestions.append(
            "总耗时主要在去噪循环，优先优化DiT前向与通信阶段（attn kernel/comm、cfg allgather）。"
        )

    if attn_run:
        comm_share = attn_run["ratio"].get("comm_share", {}).get("max", 0.0)
        kernel_share = attn_run["ratio"].get("kernel_share", {}).get("max", 0.0)
        if comm_share > 0.35:
            suggestions.append("Attention通信占比高，优先调优并行切分(ulysses/ring/tp)与通信重叠。")
        if kernel_share < 0.45:
            suggestions.append("Attention kernel占比偏低，可能受pack/通信或其他开销限制，建议检查qkv打包与数据布局。")

        comm_step = attn_run["step"].get("comm_total")
        kernel_step = attn_run["step"].get("attn_kernel")
        if comm_step and kernel_step:
            comm_p50 = comm_step.get("p50_ms", 0.0)
            comm_p90 = comm_step.get("p90_ms", 0.0)
            kernel_p50 = kernel_step.get("p50_ms", 0.0)
            kernel_p90 = kernel_step.get("p90_ms", 0.0)
            comm_jitter = (comm_p90 / comm_p50 - 1.0) if comm_p50 > 0.0 else 0.0
            kernel_jitter = (kernel_p90 / kernel_p50 - 1.0) if kernel_p50 > 0.0 else 0.0
            if comm_jitter > 0.03 and comm_jitter > kernel_jitter * 1.5:
                suggestions.append(
                    "检测到Attention通信抖动高于kernel抖动，可能存在collective竞争（如allgather/alltoall重叠）；建议对比serialize_comm并补充collective级trace定位冲突窗口。"
                )

    if block_run and block_run.get("top"):
        top_list = sorted(block_run["top"], key=lambda x: x["rank"])
        if len(top_list) >= 2:
            top1_share = top_list[0]["share"]
            top2_share = top_list[1]["share"]
            if top1_share > 0.035 and top1_share > top2_share * 1.20:
                suggestions.append(
                    "少数DiT block占比偏高，建议对top block做细粒度算子分析（LayerNorm/attention/FFN）并尝试算子融合。"
                )

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Analyze Wan stage/attention profiler CSV output.")
    parser.add_argument("profile_file", type=Path, help="Path to perf.csv")
    args = parser.parse_args()

    if not args.profile_file.exists():
        raise SystemExit(f"Profile file not found: {args.profile_file}")

    lines = args.profile_file.read_text(encoding="utf-8").splitlines()
    stage_runs = parse_stage_runs(lines)
    attn_runs = parse_attn_runs(lines)
    block_runs = parse_block_runs(lines)

    if not stage_runs:
        raise SystemExit("No stage profiling runs found.")

    current = stage_runs[-1]
    previous = stage_runs[-2] if len(stage_runs) > 1 else None
    attn_current = attn_runs[-1] if attn_runs else None
    block_current = block_runs[-1] if block_runs else None

    summarize_run(current, attn_current, block_current)
    if previous:
        print_comparison(current, previous)

    suggestions = build_suggestions(current, attn_current, block_current)
    if suggestions:
        print("\nOptimization suggestions:")
        for idx, text in enumerate(suggestions, start=1):
            print(f"  {idx}. {text}")


if __name__ == "__main__":
    main()
