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
