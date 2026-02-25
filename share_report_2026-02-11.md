# Wan2.1 性能优化分享报告（2026-02-11）

## 1. 目标与约束
- 目标：在 **开启 FSDP 控显存** 的前提下，提升端到端推理性能（`request_total` 下降）。
- 约束：不能以显存大幅增加换取速度。
- 判定优先级：
  1. `request_total` 下降。
  2. `reserved_peak` 不高于 FSDP 基线（允许小幅波动）。
  3. 运行稳定。

## 2. 今日工作量（代码与实验）
- 代码提交量：当天累计 **27 个 commit**（`2026-02-11`）。
- 主要新增/改造能力（非纯参数切换）：
  - `WAN_A2A_FENCE`：A2A 栅栏串行策略。
  - `WAN_A2A_EVENT_GATE`：A2A 独立流 + event gate。
  - `WAN_FSDP_ALLGATHER_WAIT_A2A`：将 FSDP allgather 延后到 attention 最后一次 `out_all_to_all` 事件后。
  - `WAN_FSDP_RESHARD_AFTER_FORWARD`：FSDP 重分片开关。
  - `WAN_FSDP_BLOCK_GROUP_SIZE`：block 级 FSDP 分组包装粒度。
  - `WAN_FSDP_WRAP_MODE=ffn_only`：仅 Wan DiT FFN 做 FSDP 包装（T5 回退 block）。
- 工程修复与可观测性增强：
  - 修复 block group 作用域和 forward 签名问题。
  - 修复 grouped block 的 attention cache 挂载问题。
  - 增加更细粒度 profiling 与分析脚本，补全诊断链路。

## 3. 实验范围与方法
- 主验证：统一采用 **20-step**，关注真实优化目标（总体时延 + 显存）。
- 补充分享：纳入早期 **10-step** 框架自带功能（`ALGO=1`、`W8A8`）结果，用于趋势展示。
- 核心指标：`request_total`、`denoise_step_total avg`、`comm_total avg`、`qkv_all_to_all avg`、`out_all_to_all avg`、`reserved_peak`。

## 4. 20-step 结果（核心对比）

### 4.1 原始数据
| 方案 | request_total (ms) | comm_total_avg (ms) | qkv_all_to_all_avg (ms) | out_all_to_all_avg (ms) | reserved_peak (MB) |
|---|---:|---:|---:|---:|---:|
| FSDP 基线 | 124055.328 | 1133.744 | 674.225 | 460.622 | 19792 |
| allgather wait（recent A2A） | 124444.148 | 1147.960 | 677.393 | 472.437 | 19792 |
| allgather wait（last A2A） | 124072.914 | 1141.159 | 678.523 | 462.831 | 19792 |
| A2A gate=0 | 124171.562 | 1134.456 | 672.170 | 465.680 | 19792 |
| A2A gate=1 | 124776.203 | 1159.079 | 679.029 | 481.100 | 20274 |
| 关闭 FSDP | 119702.383 | 988.726 | 576.364 | 420.580 | 42924 |
| FSDP + `ffn_only` | 121249.977 | 996.825 | 580.355 | 418.065 | 44688 |

### 4.2 相对 FSDP 基线变化
| 方案 | request_total | comm_total | qkv_all_to_all | out_all_to_all | reserved_peak |
|---|---:|---:|---:|---:|---:|
| allgather wait（recent A2A） | +0.31% | +1.25% | +0.47% | +2.57% | +0.00% |
| allgather wait（last A2A） | +0.01% | +0.65% | +0.64% | +0.48% | +0.00% |
| A2A gate=0 | +0.09% | +0.06% | -0.30% | +1.10% | +0.00% |
| A2A gate=1 | +0.58% | +2.23% | +0.71% | +4.45% | +2.44% |
| 关闭 FSDP | -3.51% | -12.79% | -14.51% | -8.69% | +116.88% |
| FSDP + `ffn_only` | -2.26% | -12.08% | -13.92% | -9.24% | +125.79% |

## 5. 框架自有功能历史数据（10-step，分享补充）
- 说明：本节用于展示历史趋势，**不与 20-step FSDP 对照做严格横比**。
- 说明：部分 CSV 仅有 stage 汇总，缺少 A2A 子项（`qkv/out_all_to_all`）。

| 方案 | 对应 CSV | request_total (ms) | 相对 no-algo(10-step) |
|---|---|---:|---:|
| no-algo + no-serialize | `perf_noalgo_noserialize.csv` | 62603.539 | 基线 |
| ALGO=1 + serialize=0 | `perf_algo1_noserialize.csv` | 49983.793 | -20.16% |
| ALGO=1 + serialize=1 | `perf_algo1.csv` | 50052.293 | -20.05% |
| ALGO=1 + `cfg_fused_forward=0` | `perf_algo1_ab_base.csv` | 49937.895 | -20.23% |
| ALGO=1 + `cfg_fused_forward=1` | `perf_algo1_ab_fused.csv` | 49448.238 | -21.01% |
| 顺序扫描 baseline | `perf_seq_01_baseline.csv` | 49308.055 | -21.24% |
| 顺序扫描 W8A8/quant | `perf_seq_02_quant.csv` | 42386.715 | -32.29% |
| 顺序扫描 attention cache | `perf_seq_03_attentioncache.csv` | 49424.098 | -21.05% |
| 顺序扫描 rainfusion(no fused) | `perf_seq_04_rainfusion_nofused.csv` | 52583.848 | -16.00% |
| 顺序扫描 quant block | `perf_seq_05_quant_block.csv` | 42741.723 | -31.73% |
| 顺序扫描 rainfusion guard | `perf_seq_07_rainfusion_guard.csv` | 52601.789 | -15.98% |

## 6. 结论（可直接对外同步）
- 现象确认：开启 FSDP 后，通信竞争会抬高 A2A 时间；关闭 FSDP 时 A2A 明显变快。
- 关键事实：单纯做“allgather/A2A 串行化”当前没有形成稳定端到端净收益（收益接近噪声或变慢）。
- trade-off 明确：关闭 FSDP 或 `ffn_only` 可改善通信与时延，但显存显著上升，违背当前硬约束。
- 阶段结论：当前最优仍接近 FSDP 基线配置；需继续寻找“控显存前提下的净收益”路径。

## 7. 后续建议（下一轮）
1. 继续以 20-step 做单变量扫描，避免多变量耦合导致结论失真。
2. 增加“分阶段重叠度”诊断（按 step 记录 allgather 与 A2A 的重叠区间），而不只看均值。
3. 在保持 FSDP 的前提下，优先探索计算侧降耗（kernel/图编译/算子融合）与通信调度协同优化。

