# Wan2.1 算子级优化实验汇总

## 当前状态

- 开发分支：`codex/custom-self-attn-probe`
- 当前提交：`bd747ce feat: add infer attention kernel option`
- 基线：沿用同一份 20-step FSDP baseline，`Generating video used time` 约 `123s`。
- 验证环境：`ssh npu`，容器 `mindie-wan2.1-xysheng`，目录 `/root/xysheng/Wan2.1`。
- 所有新增优化默认关闭，需要通过环境变量显式打开。
- 不使用旧参数 `--serialize_comm` 和 `WAN_A2A_*`。

固定验证命令参数：

```bash
torchrun --nproc_per_node=4 --master_port=<PORT> generate.py \
  --task t2v-14B \
  --size '832*480' \
  --ckpt_dir /apps/sharedstorage/Wan2.1-T2V-14B \
  --dit_fsdp \
  --t5_fsdp \
  --frame_num 81 \
  --sample_steps 20 \
  --ulysses_size 4 \
  --vae_parallel \
  --prompt "<agent.md 固定 prompt>" \
  --save_file <output>.mp4
```

## 已落地代码

### 1. 自定义 self-attention wrapper

相关文件：

- `wan/ops/custom_attention.py`
- `wan/modules/attn_layer.py`
- `tools/bench_custom_self_attn.py`

开关：

- `WAN_CUSTOM_SELF_ATTN=1`
- `WAN_CUSTOM_SELF_ATTN_IMPL=direct_bsnd|mindiesd_bsnd|infer_bnsd`
- `WAN_CUSTOM_SELF_ATTN_VERIFY=1`
- `WAN_CUSTOM_SELF_ATTN_VERIFY_RTOL=<float>`
- `WAN_CUSTOM_SELF_ATTN_VERIFY_ATOL=<float>`
- `WAN_CUSTOM_SELF_ATTN_REQUIRE_SHAPE=0|1`

默认路径仍是：

```python
mindiesd.attention_forward(..., opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
```

### 2. QKV all-to-all 打包

相关文件：

- `wan/distributed/comm.py`
- `wan/modules/attn_layer.py`

新增接口：

```python
all_to_all_4D_qkv_packed(query, key, value, group)
```

开关：

- `WAN_PACK_QKV_A2A=1`
- `WAN_PACK_QKV_A2A_VERIFY=1`

实现思路：

- 对 q/k/v 分别做原 all-to-all 前的 head/sequence 预处理。
- 将三路 packed 到同一个 buffer。
- 执行一次 `dist.all_to_all_single`。
- split 后复用原后处理逻辑。
- copy 优化版避免 `query_t/key_t/value_t + cat` 的中间 staging，直接写入 packed buffer。

### 3. split-head attention 调度

相关文件：

- `wan/modules/attn_layer.py`

开关：

- `WAN_SPLIT_HEAD_ATTN=1`
- `WAN_SPLIT_HEAD_ATTN_VERIFY=1`

实现思路：

- 对原本 `use_all_head=True` 的 10-head attention，改为按 head split 成 10 次单头 `fused_attn_score`，最后 `torch.cat(dim=2)`。
- verify 模式下同时跑原全头 attention，做 bitwise close。

## 实验结果

### 有轻微收益但不稳定

#### `WAN_CUSTOM_SELF_ATTN_IMPL=infer_bnsd`

底层算子：

```python
torch_npu.npu_fused_infer_attention_score(
    q.transpose(1, 2),
    k.transpose(1, 2),
    v.transpose(1, 2),
    num_heads=10,
    input_layout="BNSD",
    scale=0.08838834764831845,
)[0].transpose(1, 2)
```

单算子 microbench：

| 方案 | avg ms | 数值 |
| --- | ---: | --- |
| 默认 `mindiesd_bnsd` | `39.066` | reference |
| `infer_bnsd` | `38.925` | `max diff=0.000244`, `mean diff=0.000003` |

整模型：

| 验证 | 结果 |
| --- | --- |
| 2-step verify | 通过，正常保存视频 |
| verify 误差 | 单层 max diff 峰值到 `0.0625`，mean 约 `1e-6~1e-5` |
| 20-step 第一次 | `121.7145s` |
| 20-step 第二次 | `122.4892s` |

结论：

- 有真实整网收益迹象，但波动较大。
- 相对 `123s` baseline，大致是亚百分级到临界 1%。
- 不是 bitwise 等价，不建议默认打开。
- 可以作为继续深挖 `FlashAttentionScore` / infer attention kernel tiling 的候选入口。

复跑建议：

```bash
WAN_CUSTOM_SELF_ATTN=1 WAN_CUSTOM_SELF_ATTN_IMPL=infer_bnsd \
torchrun --nproc_per_node=4 --master_port=29648 generate.py ...
```

如果要做功能验证：

```bash
WAN_CUSTOM_SELF_ATTN=1 \
WAN_CUSTOM_SELF_ATTN_IMPL=infer_bnsd \
WAN_CUSTOM_SELF_ATTN_VERIFY=1 \
WAN_CUSTOM_SELF_ATTN_VERIFY_RTOL=1e-2 \
WAN_CUSTOM_SELF_ATTN_VERIFY_ATOL=1e-2 \
torchrun --nproc_per_node=4 --master_port=29647 generate.py ... --sample_steps 2
```

### 无有效整网收益

#### `WAN_CUSTOM_SELF_ATTN_IMPL=direct_bsnd`

底层算子：

```python
torch_npu.npu_fusion_attention(q, k, v, head_num=10, input_layout="BSND")
```

结果：

| 验证 | 结果 |
| --- | --- |
| 单算子目标 shape | 默认 `40.013 ms`，direct BSND `40.023 ms` |
| 数值 | `max diff=0`, `mean diff=0` |
| 2-step verify | 通过 |
| 20-step | `124.1452s` |

结论：

- BSND 直连可以避免 mindiesd BNSD wrapper 中的显式 transpose，但目标 shape 上没有收益。
- 整网变慢，no-go。

#### `WAN_CUSTOM_SELF_ATTN_IMPL=mindiesd_bsnd`

结果：

| 验证 | 结果 |
| --- | --- |
| 小 shape | 有收益 |
| 目标 shape | 默认 `38.608 ms`，BSND `40.109 ms` |
| 数值 | `max diff=0`, `mean diff=0` |

结论：

- 小 shape 快不代表主 shape 快。
- 不建议继续。

#### `WAN_PACK_QKV_A2A`

结果：

| 版本 | 2-step verify | 20-step |
| --- | --- | ---: |
| 初版 packed | 通过 | `122.9805s` |
| copy 优化版 | 通过 | `122.6868s` |

结论：

- 功能等价，verify bitwise 通过。
- 相比 baseline 约 `123s` 只有约 `0.25%`，不达 1%。
- 减少 collective 调用数的收益被 pack/split/copy 数据搬运抵消。
- 保留默认关闭，不建议作为主线继续。

#### `WAN_SPLIT_HEAD_ATTN`

结果：

| 验证 | 结果 |
| --- | --- |
| 单算子 | 全 10 头 `39.308 ms`，单头循环后 cat `38.049 ms` |
| 数值 | `max diff=0`, `mean diff=0` |
| 2-step verify | 通过 |
| 20-step | `122.9929s` |

结论：

- 单 attention kernel 有约 3% 改善。
- 整网被 10 次 kernel launch 和 `cat` 抵消。
- no-go。

## 已排查但未接入的低层候选

### `torch_npu.npu_fusion_attention_v2`

结果：

| 方案 | avg ms | 数值 |
| --- | ---: | --- |
| 默认 `fused_attn_score BNSD` | `38.920` | reference |
| `fusion_v2_BNSD` | `40.347` | 0 diff |
| `fusion_v2_BSND` | `39.887` | 0 diff |
| `fusion_v2_BSH` | `40.193` | 0 diff |

结论：数值可靠但更慢，不接入。

### `torch_npu.npu_prompt_flash_attention`

结果：

| 方案 | avg ms | 数值 |
| --- | ---: | --- |
| 默认 `fused_attn_score BNSD` | `39.112` | reference |
| `prompt BNSD` | `40.781` | `max diff=0.000244` |
| `prompt BSND/BSH` | `48-51 ms` | `max diff=0.000244` |

结论：更慢且非 bitwise，不接入。

### `torch_npu.npu_apply_rotary_pos_emb`

目的：

- 尝试一次处理 q/k RoPE，替代当前两次 `rotary_position_embedding`。

结果：

- `BSND` 在 10-head shape 上速度快，但数值不对，diff 很大。
- 40-head 目标 shape 触发 tiling 限制。

结论：语义和当前 `rotated_interleaved` 不直接对齐，不接入。

### `torch_npu.npu_mrope`

目的：

- 尝试 q/k 双路 RoPE 融合。

结果：

- 文档要求的 `cos_sin_cache` 排布和当前 Wan freqs 形态未直接对齐。
- 多种 cache 排布反推都无法对齐现有 `rotated_interleaved` 输出。

结论：当前不接入。若继续，应先独立搞清楚 `cos_sin_cache` 精确布局。

### `torch_npu.npu_ffn`

目的：

- 替代 `Linear -> GELU(tanh) -> Linear`。

结果：

| 方案 | avg ms | 备注 |
| --- | ---: | --- |
| 原始 `linear_gelu_tanh_linear` | `32.093` | reference |
| `npu_ffn_gelu_bias_fp32` | `45.112` | 更慢 |
| `npu_ffn_fastgelu_bias_fp32` | `42.664` | 更慢 |

约束：

- BF16 高精度路径要求 bias 为 FP32。
- activation 与 `GELU(approximate='tanh')` 语义不完全一致。

结论：no-go。

### `torch_npu.npu_linear`

结果：

| shape | 方案 | avg ms |
| --- | --- | ---: |
| `[1,32760,5120] -> 5120` | `F.linear_3d` | `5.330` |
| flatten 后 2D | `npu_linear_2d` | `6.194` |
| `[1,32760,5120] -> 13824` | `F.linear_3d` | `16.253` |
| flatten 后 2D | `npu_linear_2d` | `16.691` |

结论：不如现有 `F.linear/addmm`，不接入。

### QKV packed linear

目的：

- 将 self-attention 的 q/k/v 三个 linear 合成一次大 linear。

结果：

| 方案 | avg ms | 数值 |
| --- | ---: | --- |
| 三次 `F.linear` | `4.103` | reference |
| packed `F.linear(...).chunk(3)` | `4.161` | 0 diff |

结论：不快，不接入。

### `torch.addcmul` 融合残差门控

目的：

- 替代 `x + y * gate`。

结果：

| 方案 | avg ms | 数值 |
| --- | ---: | --- |
| `x + y * g` | `1.4949` | reference |
| `torch.addcmul(x, y, g)` | `1.4901` | `max diff=0.0625`, `mean diff=0.00107` |

结论：收益极小且 BF16 舍入语义不同，不接入。

### GELU 替代

结果：

| 方案 | avg ms | 数值 |
| --- | ---: | --- |
| `F.gelu(approximate='tanh')` | `1.6079` | reference |
| `F.gelu(approximate='none')` | `1.6066` | 0 diff |
| `torch_npu.npu_gelu` | `3.1002` | 更慢 |
| `torch_npu.npu_fast_gelu` | `1.4921` | `max diff=0.03125`, `mean diff=0.00496` |

结论：

- fast_gelu 单算子略快，但激活占比小且误差更大。
- 暂不接入。

## Profile 观察

本地已有 FSDP detailed profile 的 `op_summary` 聚合显示：

| 算子 | 累计耗时 | 调用数 |
| --- | ---: | ---: |
| `aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore` | `247300.994 ms` | `12884` |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | `108254.057 ms` | `51200` |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | `8954.008 ms` | `39028` |
| `RotaryPositionEmbedding2` | `5634.530 ms` | `12800` |
| `aclnnCast_CastAiCore_Cast` | `4664.799 ms` | `50020` |
| `aclnnFlashAttentionScore_TransposeAiCore_Transpose` | `4269.243 ms` | `38400` |
| `aclnnGelu_Gelu_Gelu` | `2768.578 ms` | `6560` |

判断：

- Python 外层组合优化很难再吃到稳定收益。
- `FlashAttentionScore` 是最大头部，且 wrapper 级 BNSD/BSND/BSH 调整已经基本验证无效。
- 后续更应该进入 CANN tiling/kernel 本身，而不是继续在 PyTorch 层拼装。

## 推荐后续路线

### 路线 A：继续验证 `infer_bnsd`

适合目标：

- 接受非 bitwise 但 BF16 量级误差。
- 想快速确认是否能稳定超过 1%。

建议操作：

1. 连续跑 3 到 5 次 20-step，记录 `Generating video used time`。
2. 如果均值稳定低于 `121.8s`，再跑 detailed profile。
3. 对输出视频做人工质量检查。
4. 若质量和均值都通过，再考虑作为可选配置保留。

注意：

- 当前两次 20-step 为 `121.7145s` 和 `122.4892s`，波动较大。
- 不建议默认打开。

### 路线 B：底层优化 `FlashAttentionScore`

适合目标：

- 继续做真正算子级优化。

建议关注：

- 固定 shape：`[B=1, S=32760, N=10, D=128]`。
- dtype：BF16。
- 无 mask、非 causal、全 attention。
- 当前最快稳定参考是 `mindiesd fused_attn_score layout=BNSD`。
- 需要优化的是底层 tiling、内部 transpose、workspace、中间 softmax 输出等。

可复用 microbench 形态：

```python
q = torch.randn((1, 32760, 10, 128), device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = attention_forward(
    q, k, v,
    opt_mode="manual",
    op_type="fused_attn_score",
    layout="BNSD",
)
```

目标：

- 单算子从约 `39 ms` 降到 `38 ms` 以内，并保持整网 20-step 稳定低于 `121.8s`。

### 路线 C：RoPE q/k 双路融合

适合目标：

- 能修改或新增 RoPE kernel。

当前障碍：

- `npu_apply_rotary_pos_emb` 不能直接对齐当前 `rotated_interleaved`。
- `npu_mrope` 的 cache 布局还没反推成功。

建议：

- 先做独立小 shape correctness harness。
- 明确 `cos/sin/cache` 排布，再扩大到 `[1,32760,40,128]` 和 `[1,32760,10,128]`。
- 不建议未对齐语义前接入主链路。

## 继续操作 checklist

1. 从 `codex/custom-self-attn-probe` 拉最新代码。
2. 确认默认无任何 `WAN_*` 开关时仍走原路径。
3. 每次只开启一个实验开关。
4. 功能验证先跑 `--sample_steps 2`。
5. 性能验证跑 `--sample_steps 20`，主看 `Generating video used time`。
6. 达到 1% 后才跑 detailed profile。
7. 不要把 profile 目录、tar 包、`tmp/`、本地未跟踪产物加入提交。

