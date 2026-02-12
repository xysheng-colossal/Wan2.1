# Wan2.1 性能优化接力手册（Agent）

## 1. 当前目标（固定）
- 总目标：**总体性能提升**（端到端时延下降），不是单点优化 `all2all`。
- 显存约束：在开启 FSDP 的前提下，`reserved_peak` 不高于当前 FSDP 基线（允许小幅测量波动）。
- 方向说明：`all2all` 优化只是其中一个方向，用于定位通信瓶颈，不是最终目标本身。
- 约束：当前验证统一按 **20-step** 跑，不做 50-step。
- 重点观测：`request_total`、`denoise_step_total avg`、`reserved_peak`、`comm_total avg`、`qkv_all_to_all avg`、`out_all_to_all avg`。

## 1.1 优化判定标准（优先级）
1. `request_total` 下降（主目标）。  
2. `reserved_peak` 不增加（硬约束，优先于子指标好看）。  
3. 运行稳定（无报错/无明显抖动异常）。  
4. 子指标（如 `all2all`、`allgather`）只作为诊断依据，不单独作为成功标准。  

## 2. 固定环境与流程
- 服务器：`ssh npu`
- 容器：`docker exec -i mindie-wan2.1-xysheng bash -lc '...'`
- 仓库目录：`/root/xysheng/Wan2.1`
- 推送分支：本地 `codex/perf-single-commit` -> 远端 `xys/perf`
- 服务器拉取：`git pull xys perf`

## 3. 已落地能力（代码开关）
- `WAN_A2A_FENCE`：A2A 前后同步栅栏（强约束，易降速）
- `WAN_A2A_EVENT_GATE`：A2A 独立流 + event gate（流级依赖）
- `WAN_FSDP_ALLGATHER_WAIT_A2A`：FSDP allgather 等待 A2A 事件
  - 语义已修正为：**等待 attention 最后一次 `out_all_to_all` 事件**
- `WAN_FSDP_RESHARD_AFTER_FORWARD`：控制 FSDP 前后重分片
- `WAN_FSDP_BLOCK_GROUP_SIZE`：block 分组包装粒度
- `WAN_FSDP_WRAP_MODE`：
  - `block`（默认）
  - `ffn_only`（仅 Wan DiT 的 FFN 做 FSDP；T5 自动回退 block）

## 4. 关键历史数据（请以此为对照）

### 4.1 FSDP 基线（对照）
- 日志：`run_agwait0.log`
- 配置要点：`dit_fsdp=True,t5_fsdp=True`，`WAN_FSDP_WRAP_MODE=block`
- 结果：
  - `request_total`: `124055.328 ms`
  - `denoise_step_total avg`: `6095.747 ms`
  - `comm_total avg`: `1133.744 ms`
  - `qkv_all_to_all avg`: `674.225 ms`
  - `out_all_to_all avg`: `460.622 ms`
  - `reserved_peak`: `19792 MB`

### 4.2 关闭 FSDP（速度参考）
- 日志：`run_nofsdp.log`
- 配置要点：`dit_fsdp=False,t5_fsdp=False`
- 结果：
  - `request_total`: `119702.383 ms`（比 FSDP 基线快 `-3.51%`）
  - `comm_total avg`: `988.726 ms`（`-12.79%`）
  - `qkv_all_to_all avg`: `576.364 ms`（`-14.51%`）
  - `out_all_to_all avg`: `420.580 ms`（`-8.69%`）
  - `reserved_peak`: `42924 MB`（显存大增，约 `+116.88%`）

### 4.3 FSDP + ffn_only（最新实验）
- 日志：`run_fsdp_ffnonly.log`
- 配置要点：`WAN_FSDP_WRAP_MODE=ffn_only` + `dit_fsdp=True,t5_fsdp=True`
- 结果：
  - `request_total`: `121249.977 ms`（比 FSDP 基线快 `-2.26%`，但比 no-FSDP 慢 `+1.29%`）
  - `comm_total avg`: `996.825 ms`（比 FSDP 基线好 `-12.08%`）
  - `qkv_all_to_all avg`: `580.355 ms`
  - `out_all_to_all avg`: `418.065 ms`
  - `reserved_peak`: `44688 MB`（显存更高，目标失败）

### 4.4 完整对比表（20-step，含 A2A）
| 方案 | 对应日志 | request_total (ms) | denoise_step_avg (ms) | attn_kernel_avg (ms) | comm_total_avg (ms) | qkv_all_to_all_avg (ms) | out_all_to_all_avg (ms) | reserved_peak (MB) | gen_time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FSDP基线 | `run_agwait0.log` | 124055.328 | 6095.747 | 3278.098 | 1133.744 | 674.225 | 460.622 | 19792 | 124.1431 |
| allgather wait（recent A2A） | `run_agwait1.log` | 124444.148 | 6114.215 | 3302.674 | 1147.960 | 677.393 | 472.437 | 19792 | 124.5290 |
| allgather wait（last A2A） | `run_agwait1_lasta2a.log` | 124072.914 | 6096.835 | 3283.661 | 1141.159 | 678.523 | 462.831 | 19792 | 124.1605 |
| A2A gate=0 | `run_gate0.log` | 124171.562 | 6100.410 | 3285.612 | 1134.456 | 672.170 | 465.680 | 19792 | 124.2586 |
| A2A gate=1 | `run_gate1.log` | 124776.203 | 6130.315 | 3301.764 | 1159.079 | 679.029 | 481.100 | 20274 | 124.8568 |
| 关闭FSDP | `run_nofsdp.log` | 119702.383 | 5883.693 | 3300.980 | 988.726 | 576.364 | 420.580 | 42924 | 119.7899 |
| FSDP + `ffn_only` | `run_fsdp_ffnonly.log` | 121249.977 | 5955.170 | 3282.317 | 996.825 | 580.355 | 418.065 | 44688 | 121.3334 |

### 4.5 相对 FSDP 基线（`run_agwait0.log`）差异
| 方案 | request_total | comm_total | qkv_all_to_all | out_all_to_all | reserved_peak |
|---|---:|---:|---:|---:|---:|
| allgather wait（recent A2A） | +0.31% | +1.25% | +0.47% | +2.57% | +0.00% |
| allgather wait（last A2A） | +0.01% | +0.65% | +0.64% | +0.48% | +0.00% |
| A2A gate=0 | +0.09% | +0.06% | -0.30% | +1.10% | +0.00% |
| A2A gate=1 | +0.58% | +2.23% | +0.71% | +4.45% | +2.44% |
| 关闭FSDP | -3.51% | -12.79% | -14.51% | -8.69% | +116.88% |
| FSDP + `ffn_only` | -2.26% | -12.08% | -13.92% | -9.24% | +125.79% |

### 4.6 框架自有功能历史数据（分享用：ALGO=1 / W8A8）
- 说明：以下均为早期 **10-step** 数据，主要用于分享“框架自有开关”的收益趋势，不与上面 **20-step FSDP 基线**做严格一一对比。
- 数据特征：这些 CSV 多数只有 stage 总时延（如 `request_total`），缺少 `qkv_all_to_all/out_all_to_all/comm_total` 等 attention 子项。

| 方案 | 对应 CSV | sample_steps | request_total (ms) | 相对 no-algo(10-step) |
|---|---|---:|---:|---:|
| no-algo + no-serialize | `perf_noalgo_noserialize.csv` | 10 | 62603.539 | 基线 |
| ALGO=1 + serialize=0 | `perf_algo1_noserialize.csv` | 10 | 49983.793 | -20.16% |
| ALGO=1 + serialize=1 | `perf_algo1.csv` | 10 | 50052.293 | -20.05% |
| ALGO=1 + `cfg_fused_forward=0` | `perf_algo1_ab_base.csv` | 10 | 49937.895 | -20.23% |
| ALGO=1 + `cfg_fused_forward=1` | `perf_algo1_ab_fused.csv` | 10 | 49448.238 | -21.01% |
| 顺序扫描 baseline | `perf_seq_01_baseline.csv` | 10 | 49308.055 | -21.24% |
| 顺序扫描 W8A8/quant | `perf_seq_02_quant.csv` | 10 | 42386.715 | -32.29% |
| 顺序扫描 attention cache | `perf_seq_03_attentioncache.csv` | 10 | 49424.098 | -21.05% |
| 顺序扫描 rainfusion(no fused) | `perf_seq_04_rainfusion_nofused.csv` | 10 | 52583.848 | -16.00% |
| 顺序扫描 quant block | `perf_seq_05_quant_block.csv` | 10 | 42741.723 | -31.73% |
| 顺序扫描 rainfusion guard | `perf_seq_07_rainfusion_guard.csv` | 10 | 52601.789 | -15.98% |

## 5. 结论（当前已确认）
- `all2all` 变慢与 `allgather` 并行竞争相关（关闭 FSDP 时 `all2all` 明显变快）。
- 但“强串行/弱串行”目前都没有实现稳定净收益（或收益接近噪声）。
- `ffn_only` 虽改善 `all2all`，但显存不降反升，不满足“总体性能提升且显存不增加”的目标。

## 6. 常见坑（必须避免）
- `prompt` 含 `'` 时，单引号 shell heredoc 会炸；优先用双引号变量。
- `block group` 不能误作用 T5（曾触发 forward 参数签名错误）。
- `block group` 后 cache 挂载要递归到真实 attention block（否则 `self.cache` 为 `None`）。
- 远端偶发 `Connection closed by UNKNOWN port 65535`，改用交互式 SSH 继续执行。

## 7. 标准运行模板（20-step）
```bash
docker exec -i mindie-wan2.1-xysheng bash -lc '
set -euo pipefail
cd /root/xysheng/Wan2.1
model_base=/apps/sharedstorage/Wan2.1-T2V-14B
WAN_A2A_FENCE=0 \
WAN_A2A_EVENT_GATE=0 \
WAN_FSDP_ALLGATHER_WAIT_A2A=0 \
WAN_FSDP_RESHARD_AFTER_FORWARD=1 \
WAN_FSDP_BLOCK_GROUP_SIZE=1 \
WAN_FSDP_WRAP_MODE=block \
torchrun --nproc_per_node=4 generate.py \
  --task t2v-14B \
  --size 832*480 \
  --ckpt_dir ${model_base} \
  --dit_fsdp \
  --t5_fsdp \
  --frame_num 81 \
  --sample_steps 20 \
  --ulysses_size 4 \
  --vae_parallel \
  --profile_stage \
  --profile_attn \
  --profile_stage_file perf_xxx.csv \
  --serialize_comm \
  --prompt "A boy plays piano." > run_xxx.log 2>&1
grep -E "request_total:|denoise_step_total: avg=|attn_kernel: avg=|comm_total: avg=|qkv_all_to_all: avg=|out_all_to_all: avg=|Generating video used time" run_xxx.log | tail -n 40
'
```

## 8. 新窗口接手 Checklist
1. `git status` 确认工作区干净。  
2. `git log --oneline -n 8` 确认最近提交包含：A2A gate / ALLGATHER_WAIT_A2A / WRAP_MODE。  
3. 服务器 `git pull xys perf`。  
4. 先跑 1 组对照，再改 1 个变量，禁止一次改多个主变量。  
5. 每次记录：配置、`request_total`、`comm_total`、`qkv/out_all_to_all`、`reserved_peak`。  
