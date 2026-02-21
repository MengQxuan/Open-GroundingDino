# Open-GroundingDINO 复现与 Query Efficiency 消融实验记录

本项目基于开源仓库 **Open-GroundingDINO**，系统完成了从环境配置、数据集转换、分布式训练到推理测速的完整复现流程，并围绕 **Object Queries 冗余性与效率问题** 展开了系统性实验研究。

重点分析：

* `num_queries` 对精度与速度的影响
* Query Pruning 的可行性
* BERT 冻结策略影响
* 推理效率瓶颈分析

---

## 1. 实验环境

### 1.1 硬件

* GPU：RTX 4090 24GB ×2
* Driver：560+
* CUDA：11.8 / 12.x
* 显存模式：Default

### 1.2 软件

* Python：3.10
* PyTorch：2.1.x (cu118)
* AMP：开启
* 分布式：DDP (`torchrun`)

### 1.3 运行方式

```bash
torchrun --nproc_per_node=1/2 main.py ...
```

---

## 2. 项目目标

* 完整复现 GroundingDINO 训练与评估流程
* 跑通 ODVG 数据格式
* 构建稳定 baseline
* 分析 queries 冗余问题
* 探索轻量化潜力

---

## 3. 数据集与格式

### 3.1 数据集

* COCO 2017

  * train2017
  * val2017

### 3.2 ODVG 转换

```bash
python tools/coco2odvg.py \
  -i instances_train2017.json \
  -o odvg_train2017.jsonl
```

---

## 4. 数据规模设置

### 4.1 主实验规模（10k / 1k）

| 项目    | 数量     |
| ----- | ------ |
| Train | 10,000 |
| Val   | 1,000  |
| Epoch | 5      |
| GPU   | 2×4090 |

该规模兼顾稳定性与实验成本。

---

## 5. 预训练模型

| 模块           | 配置                          |
| ------------ | --------------------------- |
| Backbone     | Swin-T                      |
| Text Encoder | BERT-base-uncased           |
| 权重           | groundingdino_swint_ogc.pth |

---

## 6. Baseline 结果

### 6.1 标准 Baseline（q=900）

| 指标   | 数值    |
| ---- | ----- |
| AP   | 0.552 |
| AP50 | 0.703 |
| AP75 | 0.605 |

---

### 6.2 BERT 冻结实验

| 模型          | AP    |
| ----------- | ----- |
| Normal      | 0.552 |
| Freeze BERT | 0.490 |

结论：

> 冻结文本编码器会显著损伤性能，不适合 ODVG 微调任务。

---

## 7. num_queries 消融实验

### 7.1 精度对比（10k / 1k）

| Queries | AP          | AP50        | AP75        |
| ------- | ----------- | ----------- | ----------- |
| 900     | 0.552       | 0.703       | 0.605       |
| 600     | **0.559**   | **0.713**   | **0.615**   |
| 300     | 0.556–0.558 | 0.710–0.716 | 0.612–0.617 |
| 200     | 0.551       | 0.710       | 0.606       |
| 50      | 0.537       | 0.699       | 0.582       |

### 7.2 稳定性（Seed=42）

| Run  | AP    |
| ---- | ----- |
| Run1 | 0.556 |
| Run2 | 0.558 |

波动 ≤ 0.002

说明结果高度稳定。

---

### 7.3 关键发现

* queries 从 900 → 200：

  * AP 基本不变
  * 几乎无性能损失

* 说明：

  > DETR-style queries 存在严重冗余

---

## 8. Query Pruning 实验

基于训练阶段 Top-K 选择进行裁剪：

| 设置              | AP    |
| --------------- | ----- |
| q300 + prune200 | 0.541 |
| q300 + prune50  | 0.545 |

结论：

> 简单 Top-K Pruning 无法有效提升性能。

原因：

* Matching 阶段仍依赖全 queries
* Transformer 计算仍是主瓶颈

---

## 9. 推理性能 Benchmark

### 9.1 Forward Only 测试（Batch=4）

| Queries | Latency(ms) | FPS   |
| ------- | ----------- | ----- |
| 900     | 203.15      | 19.69 |
| 600     | 203.60      | 19.65 |
| 300     | 202.97      | 19.71 |
| 200     | 203.88      | 19.62 |
| 50      | 202.82      | 19.72 |

### 9.2 结论

> 减少 queries 对推理速度影响极小。

说明：

* 当前瓶颈在 Backbone + Cross-Attention
* 非 queries 数量本身

---

## 10. 综合结论

### 10.1 已验证结论

1️⃣ Pipeline 完整可复现
2️⃣ 微调稳定有效
3️⃣ Queries 存在严重冗余
4️⃣ 冻结 BERT 明显降精度
5️⃣ 简单剪枝收益有限
6️⃣ 推理瓶颈非 queries

---

### 10.2 核心发现

> GroundingDINO 当前架构对 queries 数量不敏感。

原因：

* 多层 Attention 稀释冗余
* Hungarian Matching 过滤
* Decoder 多轮 Refinement


## 11. 结构剪枝与轻量化优化实验（2026.02 更新）

在完成 Query 冗余性分析后，进一步围绕 **Transformer 结构剪枝与跨模态推理加速** 展开系统优化实验。

### 11.1 Transformer 层剪枝

通过裁剪 Encoder / Decoder 层数，降低整体计算复杂度：

| 配置       | Encoder | Decoder | Queries | AP@0.5:0.95  |
| -------- | ------- | ------- | ------- | ------------ |
| Baseline | 6       | 6       | 300     | ~0.556       |
| Pruned   | 4       | 3       | 300     | ~0.519–0.522 |

采用 `--finetune_ignore` 显式忽略被裁剪层参数，实现无冲突加载预训练权重。

结论：

> 适度剪枝可显著减少计算量，精度下降约 3%，处于可接受范围。

---

### 11.2 Cross-Attention 降采样（num_points=2）

在 MSDeformAttn 中减少采样点数量：

```bash
dec_n_points=2
```

实验结果：

| 设置             | AP     |
| -------------- | ------ |
| np=4 (default) | ~0.522 |
| np=2           | ~0.519 |

结论：

> 降采样可减少 Attention 计算量，对精度影响较小。

---

### 11.3 Offset Clamp + Softmax FP32 稳定化

为提升量化与混合精度稳定性，引入：

* Sampling offset 限幅（offset clip）
* Softmax 保持 FP32
* BBox Head 保持 FP32

作用：

> 抑制数值抖动，提高后续量化鲁棒性。

---

### 11.4 DistilBERT 替换文本编码器

将文本编码器从 BERT-base 替换为 DistilBERT：

```bash
text_encoder_type=weights/distilbert-base-uncased
```

并修复：

* 3D attention mask 不兼容问题
* token_type_ids / position_ids 冲突

最终采用统一 TextEncoderShell 封装。

精度与显存表现：

| 模型         | AP     | Max Mem |
| ---------- | ------ | ------- |
| BERT       | ~0.514 | ~10.5GB |
| DistilBERT | ~0.512 | ~9.7GB  |

结论：

> DistilBERT 在保持精度的同时降低显存占用，但端到端加速有限。

---

### 11.5 Caption 长度对推理性能影响

扩展 benchmark 脚本，支持可控 caption 长度测试：

| Caption Len | P50 Latency (ms) | FPS |
| ----------- | ---------------- | --- |
| 4           | ~168             | ~24 |
| 128         | ~231             | ~17 |

结论：

> 长文本显著增加推理延迟，文本塔在长 prompt 场景下成为重要瓶颈。

---

### 11.6 多阶段轻量化优化路线

当前已完成阶段：

| Stage  | 内容            | 状态     |
| ------ | ------------- | ------ |
| Stage1 | Layer Pruning | ✅      |
| Stage2 | np2 Sampling  | ✅      |
| Stage3 | Offset + FP32 | ✅      |
| Stage4 | DistilBERT    | ✅      |
| Stage5 | Text INT8     | 🔄 进行中 |
| Stage6 | Vision QAT    | ⏳ 规划中  |

Stage4 最终配置：

> Pruning + np2 + offset_clip + softmax_fp32 + DistilBERT

达成：

* AP ≈ 0.512
* 显存 < 10GB
* 推理稳定

---

## 12. 当前总体结论

综合 Query 消融与结构轻量化实验，可得：

1. GroundingDINO Queries 冗余严重
2. 主瓶颈在 Backbone + Deformable Attention
3. 层剪枝 + 降采样具备较高性价比
4. 文本塔仅在长 prompt 下影响明显
5. DistilBERT 更适合作为量化前置模型
6. 纯结构优化对端到端加速有限，需结合 INT8 / 部署优化



## Stage5：文本塔 Profiling 与 Caption Length 影响分析

### 实验设置

- 模式：forward_only
- 设备：NVIDIA GPU（CUDA）
- batch_size = 4
- num_queries = 300
- warmup = 50, iters = 200
- AMP：开启
- 数据集：COCO val 子集
- Profiling：启用 profile_split，拆分 tokenize / text encoder / vision+decoder

测试 caption 长度：

\[
L \in \{4, 16, 32, 64, 128\}
\]

---

### 不同 Caption 长度下的延迟统计

单位：毫秒（mean）

| Caption Length | T_tokenize | T_text_enc | T_vision+dec | T_total |
|----------------|------------|------------|-------------|---------|
| 4              | 0.36       | 4.20       | 154.10      | 160.14  |
| 16             | 0.45       | 4.19       | 158.32      | 164.42  |
| 32             | 0.59       | 4.20       | 163.06      | 169.34  |
| 64             | 0.88       | 4.17       | 183.53      | 190.11  |
| 128            | 1.40       | 4.33       | 215.05      | 222.42  |

---

### Caption 长度对各模块延迟的影响

#### 1. Tokenization 阶段

Tokenization 运行于 CPU，其耗时随文本长度近似线性增长：

- 从 0.36 ms 增长至 1.40 ms
- 总体占比低于 1%

表明其不是系统瓶颈。

---

#### 2. 文本编码器阶段（Text Encoder）

Text Encoder 运行于 GPU，其延迟基本保持稳定：

- 始终约为 4 ms
- 与 caption 长度变化几乎无关

说明文本编码主要受固定计算开销主导。

---

#### 3. 视觉与解码阶段（Vision + Decoder）

Vision + Decoder 模块占据主要计算开销：

- 从 154.10 ms 增长至 215.05 ms
- 增量约为 61 ms

该增长与 caption 长度高度相关，说明 Transformer 中的文本-视觉融合模块
（如 Cross-Attention）是主要瓶颈来源。

---

### 延迟增长贡献分析

当 caption 长度从 4 增长至 128 时：

| 模块          | 增量 (ms) | 占比 (%) |
|---------------|-----------|----------|
| Tokenize      | +1.04     | 1.7%      |
| Text Encoder  | +0.13     | 0.2%      |
| Vision+Decoder| +60.95    | 97.9%     |
| Total         | +62.28    | 100%      |

可见，超过 97% 的延迟增长来源于 Vision+Decoder 阶段。

---

### 曲线趋势分析

整体趋势如下：

- T_tokenize 随 caption 长度线性增长
- T_text_enc 基本保持不变
- T_vision+dec 随 caption 长度显著增长
- T_total 主要由 T_vision+dec 决定

（对应折线图如 Fig. X 所示）

---

### Stage5 实验结论

1. 单独优化或量化 Text Encoder 对端到端延迟影响有限；
2 长 Caption 场景下的性能瓶颈主要位于 Transformer 的文本-视觉融合模块；
3. Text Encoder 的 INT8 量化适合作为可行性验证手段，而非主要加速来源；
4. 后续优化应聚焦 Decoder / Cross-Attention 等融合结构。

该结论为 Stage6 中针对 Decoder 的混合精度量化与 QAT 优化提供了直接依据。

---

## Stage6-0：Encoder 内部细分 Profiling（caption_len 扫描）

本节记录对 GroundingDINO **端到端推理**进行分段计时（tokenize / text encoder / backbone / transformer / heads），并进一步对 **Transformer Encoder** 内部进行细分，定位 caption 变长导致延迟上升的真正瓶颈。

- 模式：`forward_only`
- 设备：`cuda`，AMP：`True`
- 固定参数：`batch_size=4, num_queries=300, warmup=50, iters=200, num_batches=8`
- caption_len：`{4, 16, 32, 64, 128}`（caption 内容为重复 token `"a"`）
- 计时统计：均值（mean）/中位数（p50）

---

### 1. 总体结论

随着 `caption_len` 增大，端到端延迟上升几乎**全部来自 Transformer Encoder 内的 fusion 模块（enc_fusion）**；进一步拆分后发现，**增长几乎完全由 fusion attention 中的 `softmax` 与 `context bmm`（以及部分 `scores(QK^T)`）造成**，而文本塔（`text_enc`）、主干 backbone、以及 Deformable Encoder（`enc_msdef`）基本不随文本长度变化。

---

### 2. 端到端与模块级延迟（mean）

| caption_len | T_total | tokenize | text_enc | backbone | transfmr | enc | dec | enc_fusion | enc_text | enc_msdef |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4   | 171.72 | 0.355 | 4.109 | 42.955 | 118.583 | 97.024 | 15.331 | 40.973 | 4.762 | 49.801 |
| 16  | 176.42 | 0.486 | 4.206 | 42.994 | 121.493 | 99.145 | 16.021 | 42.780 | 4.856 | 49.913 |
| 32  | 182.96 | 0.565 | 4.115 | 43.755 | 125.485 | 103.935 | 15.388 | 47.002 | 4.609 | 50.837 |
| 64  | 201.24 | 0.873 | 4.201 | 42.925 | 140.578 | 118.783 | 15.508 | 62.192 | 4.875 | 50.175 |
| 128 | 235.25 | 1.349 | 4.170 | 43.054 | 165.977 | 143.607 | 15.875 | 86.882 | 5.075 | 50.100 |

> 观察：
> - `text_enc` 约 **4ms**，随 `caption_len` 基本不变（不是瓶颈）。
> - `backbone` 约 **43ms**，基本不变。
> - `dec` 约 **15~16ms**，基本不变。
> - `enc_msdef`（Deformable Encoder 主干）约 **50ms**，基本不变。
> - **enc_fusion** 从 **40.97ms → 86.88ms**，随 `caption_len` 显著增长，是主要瓶颈。

---

### 3. caption_len 增长的增量归因（len=4 → 128, mean）

- 总延迟：`171.72 → 235.25`，**+63.53 ms**
- tokenize：`0.355 → 1.349`，**+0.994 ms**
- text_enc：`4.109 → 4.170`，**+0.061 ms（几乎不变）**
- backbone：`42.955 → 43.054`，**+0.099 ms（几乎不变）**
- transformer(enc+dec)：`118.583 → 165.977`，**+47.394 ms**
  - enc：`97.024 → 143.607`，**+46.583 ms**
    - **enc_fusion：`40.973 → 86.882`，+45.909 ms（≈98% 的 encoder 增量）**
    - enc_text：`4.762 → 5.075`，+0.313 ms
    - enc_msdef：`49.801 → 50.100`，+0.299 ms
  - dec：`15.331 → 15.875`，+0.544 ms

> 核心结论：**encoder 的增量几乎全部来自 enc_fusion。**

---

### 4. enc_fusion 内部进一步细分（mean）

enc_fusion 由 `BiAttentionBlock(BiMultiHeadAttention)` 实现，可拆为：
- LN（layer_norm_v/l）
- fusion_attn（BiMultiHeadAttention 总体）
- residual（drop_path + gamma + 残差）

#### 4.1 fusion 三段（mean）

| caption_len | enc_fusion | fusion_ln | fusion_attn | fusion_resid |
|---:|---:|---:|---:|---:|
| 4   | 40.973 | 1.741 | 33.127 | 3.409 |
| 16  | 42.780 | 1.749 | 34.949 | 3.404 |
| 32  | 47.002 | 1.739 | 38.320 | 3.400 |
| 64  | 62.192 | 1.758 | 54.306 | 3.417 |
| 128 | 86.882 | 1.760 | 78.987 | 3.428 |

> 观察：增长几乎全部来自 `fusion_attn`（33.13→78.99），`fusion_ln` 和 `fusion_resid` 基本常数项。

#### 4.2 BiMultiHeadAttention 内部细分（mean）

| caption_len | attn_proj | attn_scores(QK^T) | attn_softmax | attn_ctx(probs×V) | attn_out |
|---:|---:|---:|---:|---:|---:|
| 4   | 17.080 | 1.926 | 1.249 | 5.943 | 2.613 |
| 16  | 17.111 | 2.004 | 2.070 | 6.726 | 2.613 |
| 32  | 17.072 | 2.566 | 3.063 | 8.462 | 2.603 |
| 64  | 17.083 | 4.201 | 8.610 | 15.406 | 2.625 |
| 128 | 17.145 | 7.464 | 18.208 | 24.223 | 2.626 |

> 关键观察：
> - `attn_proj`（线性投影）≈ **17ms**，**基本不随文本长度变化**（常数项）。
> - `attn_out` ≈ **2.6ms**，基本不变（常数项）。
> - 随 `caption_len` 增长最明显的是：
>   - **attn_softmax：1.25 → 18.21（+16.96ms）**
>   - **attn_ctx：5.94 → 24.22（+18.28ms）**
>   - attn_scores：1.93 → 7.46（+5.54ms）
>
> 因此，长 caption 下的主要瓶颈是：**softmax + context bmm + QK^T**（均与 `ntxt` 强相关）。

---

### 5. 对 Stage5/Stage6 的启示

#### Stage5（只量化文本塔）的现实预期
- `text_enc` 始终约 **4ms**，且不随 `caption_len` 增长 → **文本塔不是端到端瓶颈**。
- 仅量化文本塔的价值更偏向：
  - **显存占用下降**
  - **误差/精度敏感性验证（AP 基本不掉）**
  - 端到端速度提升可能有限（取决于整体 pipeline 是否受文本塔影响）

#### Stage6（面向部署的混合精度/量化友好优化）应优先聚焦
- 真正随 `caption_len` 爆炸增长的部分在：`enc_fusion` 的 attention（尤其 `softmax` 与 `ctx bmm`）
- 两类主线方向：
  1. **降低 fusion 有效文本 token 数（token pruning / top-k tokens）**  
     直接削减 `scores/softmax/ctx` 的规模（对长 caption 最有效）。
  2. **对 fusion 的线性层做 INT8/QAT，softmax 保持 FP32**  
     主要收益在 `attn_proj` 等常数项（≈17ms），更稳定、更像工业部署。

---

### 6. 复现命令

```bash
PYTHONPATH=. python tools/benchmark_infer.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --pretrain_model_path outputs/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32/checkpoint_best_regular.pth \
  --device cuda --num_workers 4 --num_queries 300 \
  --warmup 50 --iters 200 --num_batches 8 \
  --amp --forward_only \
  --caption_len 128 \
  --profile_split