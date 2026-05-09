# Wan2.1 算子级优化方案

## 1. 目标与背景

本文档汇总基于 profile 数据可以尝试的算子级优化方案。目标是在不依赖仓库既有性能开关的前提下，针对 Wan2.1 T2V-14B 的真实热点做优化。

共享基线信息：

- 基线分支/提交：`profile` / `6f10d5c`
- 服务器：`ssh npu`
- 容器：`mindie-wan2.1-xysheng`
- 仓库路径：`/root/xysheng/Wan2.1`
- 模型路径：`/apps/sharedstorage/Wan2.1-T2V-14B`
- 任务：`t2v-14B`
- 分辨率：`832*480`
- 推理步数：`20`
- 并行参数：`--dit_fsdp --t5_fsdp --ulysses_size 4 --vae_parallel`
- 共享 20-step 基线生成耗时：约 `123s`
- 新 detailed profile 路径：`/root/xysheng/Wan2.1/result_profile_shared_baseline_ops_20260509`

固定 prompt：

```text
A young boy with short brown hair, dressed in a dark blue t-shirt and red pants, is seen playing a KAWAI upright piano with skill and concentration. The piano's glossy black surface reflects the room's lighting, and its white and black keys are arranged in a standard layout, indicating a scene of musical practice or learning. The boy's hands move over the keys, suggesting he is engaged in playing or practicing a piece.
```

## 2. Profile 结论

本轮 detailed profile 采集于 `2026-05-09`，基于共享基线代码。profile 第一个 20-step 生成耗时为 `122.9333s`，与共享基线一致；第二个 profile step 包含 CANN trace 解析/导出耗时，不作为端到端性能判断依据。

`op_summary` 聚合结果：

| 类别 | Device total | 占比 | 说明 |
|---|---:|---:|---|
| Attention kernel | `253.363s` | `40.81%` | 主要是长序列 self-attention |
| Matmul / linear | `110.797s` | `17.85%` | 主要是 DiT FFN 和投影层 |
| HCCL all-gather | `105.745s` | `17.03%` | 主要是 DiT FSDP 参数 all-gather |
| HCCL all-to-all | `82.880s` | `13.35%` | Ulysses QKV/out 通信 |
| Norm / cast / layout / elementwise / GELU | 单项均较小 | `< 6%` | 暂不作为第一优先级 |

`operator_details.csv` 调用栈细分：

| Bucket | Device total | Rows | 平均耗时 | 代码路径 |
|---|---:|---:|---:|---|
| `attention/self_sp_flash` | `248.310s` | `6400` | `38.798ms` | `wan/modules/attn_layer.py -> xdit_context_parallel.py` |
| `comm/dit_fsdp_allgather_hccl` | `196.255s` | `13120` | `14.958ms` | DiT FSDP pre-forward all-gather |
| `comm/qkv_all_to_all_hccl` | `125.720s` | `38400` | `3.274ms` | `wan/distributed/comm.py(47)` |
| `linear/dit_ffn_fc1_fc2` | `48.766s` | `12800` | `3.810ms` | `wan/modules/model.py(338)` |
| `comm/out_all_to_all_hccl` | `40.041s` | `12800` | `3.128ms` | `wan/distributed/comm.py(81)` |
| `linear/self_qkv_proj` | `21.267s` | `12800` | `1.661ms` | `xdit_context_parallel.py qkv_fn` |
| `linear/cross_attn_proj` | `19.971s` | `25600` | `0.780ms` | `wan/modules/model.py cross_attn` |
| `linear/self_out_proj` | `9.480s` | `6400` | `1.481ms` | `xdit_context_parallel.py(212)` |
| `attention/t2v_cross_text_flash` | `5.025s` | `6400` | `0.785ms` | text cross-attention |

结论：第一优化目标应该是 Ulysses 后的长序列 self-attention，不是 cross-attention。

## 3. P0：固定形状长序列 Self-Attention 专用算子

### 当前路径

```text
wan/distributed/xdit_context_parallel.py::usp_attn_forward
  -> rope_apply(q/k)
  -> wan/modules/attn_layer.py::xFuserLongContextAttention.forward
  -> QKV all_to_all_4D
  -> mindiesd.attention_forward(..., layout="BNSD")
  -> torch_npu npu_fusion_attention / aclnnFlashAttentionScore
```

热点算子：

```text
aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore
```

热点形状：

- attention 前逻辑形状：`[B, S, N, D] = [1, 32760, 10, 128]`
- profiler 中底层 BNSD 形状：`[1, 10, 32760, 128]`
- dtype：`bfloat16`
- attention 类型：full self-attention
- causal：`False`
- dropout：`0`
- Ulysses size：`4`
- 每 rank head 数：`10`
- head dim：`128`

### 建议尝试的算子接口

建议提供一个固定形状/小范围 shape family 的专用 self-attention 算子：

```python
out = wan_ops.npu_wan_self_attention(q, k, v, *, layout="BSND")
```

输入输出约定：

- `q/k/v`：BF16，形状 `[1, 32760, 10, 128]`
- `out`：BF16，形状 `[1, 32760, 10, 128]`
- 数值语义等价于当前 `attention_forward(..., op_type="fused_attn_score", layout="BNSD")`

功能约束：

- 仅要求支持非 causal full attention。
- 不需要 dropout。
- 固定 `head_dim=128`。
- 必须支持 `seq_len=32760`；如果底层需要对齐，可支持就近 padding，但输出需裁回原长度。
- 不匹配形状/dtype 时必须回退旧路径。

优化方向：

- 针对 `S=32760, N=10, D=128` 做专门 tiling。
- 尽量直接消费 BSND，减少当前 `FlashAttentionScore` 周边 transpose/layout 成本。
- 如果算子内部仍使用 BNSD，建议把 layout 转换融合到 load 阶段。
- RoPE 融合可以作为第二阶段：当前 RoPE 总耗时约 `5.6s`，低于 attention 主核，但可通过融合减少读写。

收益预期：

- 该 bucket 占 device time 约 `40%`。
- attention 主核若提升 `5%`，理论 device 侧可贡献约 `2%` 改善。
- 这是当前最明确、最值得优先投入的算子优化点。

## 4. P1：DiT FSDP All-Gather / Matmul 通算融合

### 当前瓶颈

```text
comm/dit_fsdp_allgather_hccl: 196.255s, 13120 calls, avg 14.958ms
```

profile 中观察到的 all-gather size 示例：

```text
87848576 bytes
```

这是 DiT FSDP 参数 all-gather，在 block forward 前触发。它是最大的通信瓶颈，也会和 Ulysses all-to-all 竞争。

### 候选方向

调研 DiT 大 Linear 的 all-gather + matmul 通算融合，优先关注：

- self-attention Q/K/V projection
- self-attention output projection
- FFN `fc1` / `fc2`

可评估的 torch_npu 底层接口：

```python
torch_npu.npu_all_gather_base_mm(input, x2, hcom, world_size, ...)
```

注意事项：

- 当前 FSDP all-gather gather 的是参数，而 `npu_all_gather_base_mm` 更接近 TP 场景中 gather input 后做 matmul。
- 因此它大概率不能直接替换 PyTorch FSDP 的 all-gather。
- 建议先做单 Linear microbench/custom module，验证通算融合可行性后，再决定是否改造成显式 sharded Linear。

建议 microbench 形状：

- FFN fc1：`input [8190, 5120]`，weight `[5120, 13824]`
- FFN fc2：`input [8190, 13824]`，weight `[13824, 5120]`
- QKV projection：`input [8190, 5120]`，weight `[5120, 5120]`

收益预期：

- DiT FSDP all-gather bucket 占 device time 约 `17%`。
- 如果能和大 matmul 做流水/融合，收益上限高。
- 集成风险高于 P0，建议先独立 microbench。

## 5. P2：QKV Grouped All-To-All

### 当前瓶颈

```text
comm/qkv_all_to_all_hccl: 125.720s, 38400 HCCL rows, avg 3.274ms
```

当前逻辑：

- Q/K/V 分别调用 `all_to_all_4D(input, scatter_idx=2, gather_idx=1)`
- HCCL shape 示例：`[4, 8190, 1, 10, 128]`

### 已验证失败的方案

已实现并验证过 Python 层 QKV packed all-to-all：

- 分支：`codex/pack-qkv-a2a`
- 开关：`WAN_PACK_QKV_A2A=1`
- 结果：功能正确，但性能变慢。
- 原因：把 Q/K/V 拼成一个大包后，调用数下降，但单次大包通信尾延迟变大，抵消收益。

因此后续不建议继续做简单 `cat(Q,K,V)` 大包。

### 候选方向

可以评估 HCCL grouped/batched all-to-all 调度：

```python
q_out, k_out, v_out = wan_ops.grouped_all_to_all_4d([q, k, v], group)
```

要求：

- Q/K/V 保持独立 buffer。
- 三路 collective 作为一个 group 提交/调度，减少 launch/scheduling 开销。
- 保留当前每路 tensor 的 chunk 粒度，避免大包尾延迟。
- 输出必须逐元素等价于三次独立 `dist.all_to_all_single`。

收益预期：

- 目标是减少调度开销，而不是改变通信量。
- 风险中等，取决于 HCCL grouped collective 支持能力。

## 6. P3：FFN Matmul + GELU Epilogue

### 当前瓶颈

```text
linear/dit_ffn_fc1_fc2: 48.766s
aclnnGelu: 2.781s
```

热点 FFN 形状：

```text
[8190, 5120] -> [8190, 13824] -> GELU(tanh) -> [8190, 5120]
```

### 已验证失败的方案

通用 `torch_npu.npu_ffn` 已验证：

- 分支：`codex/fused-ffn`
- 开关：`WAN_FUSED_FFN=1`
- 2-step verify 通过。
- 20-step 生成耗时变慢：`127.2100s`，共享基线约 `123s`。

结论：不建议继续把 generic `npu_ffn` 作为主线。

### 候选方向

作为较小的定制优化点，可考虑：

- 只融合 `fc1 + bias + GELU(tanh)` epilogue，减少中间激活写回/读回。
- `fc2` 暂时保留普通 matmul。
- 只针对 BF16、`M=8190, K=5120, N=13824` 特化。

收益预期低于 P0/P1，因为 GELU 自身只占 device time 约 `0.45%`。

## 7. Output Projection Reduce-Scatter 现状

已验证过 output projection 通算融合：

- 分支：`codex/out-proj-comm-fusion`
- 开关：`WAN_OUT_PROJ_RS=1`
- 方案：把 `out_all_to_all + o_proj` 改为 local matmul + `npu_mm_reduce_scatter_base`

结果：

- 2-step 不开 strict verify 时能跑通。
- strict verify 出现 BF16 outlier：约 `573-684 / 41932800` 个元素不匹配，最大绝对误差 `0.25`。
- 20-step 变慢：`125.5892s` vs off-repeat `123.5803s`。

建议：

- 不作为优先方向。
- 除非能提供数值更稳定、layout 更合适的 fused reduce-scatter matmul。

## 8. 集成接口要求

每个新算子必须通过环境变量开关接入，默认关闭，支持安全回滚：

```text
WAN_CUSTOM_SELF_ATTN=0/1
WAN_CUSTOM_SELF_ATTN_VERIFY=0/1
WAN_CUSTOM_GROUPED_A2A=0/1
WAN_CUSTOM_AGMM=0/1
```

verify 模式要求：

1. 同一 forward 内同时计算旧路径和新路径。
2. 使用 `torch.testing.assert_close` 对比。
3. mismatch 时记录 max/mean absolute error。
4. verify 模式不用于性能跑。

性能模式要求：

1. 一次只开启一个新优化。
2. 使用共享基线，不需要每个实验重新跑 off。
3. 记录 `Generating video used time` 和 `REAL_TIME_SECONDS`。
4. 只有 20-step on 跑出收益后，再采 detailed profile。

## 9. 服务器验证命令模板

以下命令用于新 self-attention 算子的 on-only 性能验证。其他算子替换环境变量即可。

```bash
ssh npu
docker exec -i mindie-wan2.1-xysheng bash -lc '
set -euo pipefail
cd /root/xysheng/Wan2.1
git checkout <branch-with-operator-wrapper>

model_base=/apps/sharedstorage/Wan2.1-T2V-14B
prompt_fixed="A young boy with short brown hair, dressed in a dark blue t-shirt and red pants, is seen playing a KAWAI upright piano with skill and concentration. The piano'"'"'s glossy black surface reflects the room'"'"'s lighting, and its white and black keys are arranged in a standard layout, indicating a scene of musical practice or learning. The boy'"'"'s hands move over the keys, suggesting he is engaged in playing or practicing a piece."

start=$(python - <<PY
import time
print(time.time())
PY
)

WAN_CUSTOM_SELF_ATTN=1 \
torchrun --nproc_per_node=4 generate.py \
  --task t2v-14B \
  --size "832*480" \
  --ckpt_dir "${model_base}" \
  --dit_fsdp \
  --t5_fsdp \
  --frame_num 81 \
  --sample_steps 20 \
  --ulysses_size 4 \
  --vae_parallel \
  --prompt "${prompt_fixed}" \
  --save_file custom_self_attn_20.mp4 \
  > run_custom_self_attn_20.log 2>&1

end=$(python - <<PY
import time
print(time.time())
PY
)

python - <<PY > runtime_custom_self_attn_20.txt
print(f"REAL_TIME_SECONDS={float('${end}') - float('${start}'):.3f}")
PY

cat runtime_custom_self_attn_20.txt
grep -E "Generating video used time|Finished\\.|Traceback|RuntimeError|AssertionError" run_custom_self_attn_20.log | tail -n 80
'
```

如果 20-step on 跑出正收益，再加 detailed profile 参数：

```bash
--profile_mode detailed \
--profile_mode_dir result_custom_self_attn_detailed \
--profile_mode_steps 2 \
--profile_mode_wait 0 \
--profile_mode_warmup 1 \
--profile_mode_active 1
```

## 10. 验收标准

优化方案的最低验收标准：

- 2-step verify/smoke 通过，无 deadlock、无 assert。
- 20-step on-only 生成耗时相对共享基线 `~123s` 至少下降 `1%`。
- 新 detailed profile 中目标 bucket 明确下降。
- 其他 bucket 不出现明显回退，尤其是 FSDP all-gather 和 Ulysses all-to-all。
- 显存不应明显超过 FSDP 基线；开启 memory profile 时以 `reserved_peak=19792 MB` 作为参考。
- 新算子必须默认关闭，可通过环境变量回滚。

## 11. 推荐推进顺序

1. 优先实现 `WAN_CUSTOM_SELF_ATTN` wrapper 和单算子 harness，覆盖固定长序列 self-attention 形状。
2. 跑 2-step verify 和 20-step on-only 性能验证。
3. 若 20-step 有收益，采 detailed profile，确认 `attention/self_sp_flash` 下降。
4. 再启动 `WAN_CUSTOM_AGMM` 的 DiT Linear + FSDP 通算 microbench。
5. 最后评估 `WAN_CUSTOM_GROUPED_A2A`，前提是 HCCL 支持 grouped all-to-all 且不需要拼 Q/K/V 大包。

一句话结论：优先优化固定形状长序列 self-attention 算子，这是当前最大、最清晰、归因最干净的算子级优化目标。
