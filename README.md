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
