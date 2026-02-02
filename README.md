# Open-GroundingDINO 复现与消融实验记录

本项目基于开源仓库 **Open-GroundingDINO**，完成了从环境配置、数据集转换（COCO → ODVG）、模型加载、训练到评估的**完整复现流程**，并在 COCO 子集与中等规模数据设置上，对 **GroundingDINO 预训练模型的微调行为与关键超参数（num_queries）进行了系统性实验分析**。

实验覆盖 **单卡 / 多卡 RTX 4090** 场景，支持 **AMP 混合精度训练** 与 **分布式训练（DDP）**，并通过 **固定随机种子 + 重复实验** 验证了结果的稳定性。

---

## 1. 实验环境

* GPU：

  * RTX 4090 24GB ×1（早期复现）
  * RTX 4090 24GB ×2（后续消融实验，DDP）
* Driver / CUDA：

  * NVIDIA Driver 560+
  * CUDA 11.8 / 12.x（PyTorch cu118）
* Python：3.8 / 3.10
* PyTorch：2.1.x（支持 AMP）
* 训练方式：

  * `torchrun --nproc_per_node=1/2`
  * Distributed Data Parallel (DDP)
* 操作系统：Linux

---

## 2. 项目目标

* 复现 GroundingDINO 在 COCO 数据集上的完整训练与评估流程
* 跑通 **ODVG（Open-Domain Visual Grounding）** 数据格式
* 在不修改模型结构的前提下：

  * 验证预训练模型的下游迁移能力
  * 验证微调（fine-tuning）是否带来稳定性能提升
* 在此基础上，进一步分析：

  * **num_queries 对性能 / 稳定性 / 训练效率的影响**
* 为后续模型结构或训练策略改进提供**可靠、可复现的基线**

---

## 3. 数据集与格式转换

### 3.1 使用数据集

* **COCO 2017**

  * Train：`train2017`
  * Val：`val2017`

### 3.2 ODVG 格式转换

GroundingDINO 训练阶段采用 **ODVG jsonl** 格式描述检测数据。
本项目使用官方脚本完成 COCO → ODVG 转换：

```bash
python tools/coco2odvg.py \
  -i data/coco/annotations/instances_train2017.json \
  -o data/coco/annotations/odvg_train2017.jsonl \
  --idmap coco2017
```

验证集保持 **COCO 原生 `instances_val2017.json`**，用于标准 COCO evaluation。

---

## 4. 数据规模设置

### 4.1 小规模 Smoke Test（早期验证）

用于快速验证 pipeline 正确性：

* 训练集：COCO train2017 抽样 **5k**
* 验证集：COCO val2017 抽样 **500**
* Epoch：3–5
* 目的：

  * 验证训练 / 评估流程
  * 对比预训练 vs 微调效果

### 4.2 中等规模设置（主实验）

用于系统性消融分析：

* **训练集：10,000 images**
* **验证集：1,000 images**
* Epoch：5
* 训练方式：DDP（2× RTX 4090）
* 该设置在可控时间成本下，能够稳定反映性能趋势

---

## 5. 预训练模型

* 权重：`groundingdino_swint_ogc.pth`
* Backbone：Swin-T
* Text Encoder：BERT-base-uncased
* 加载方式：`--pretrain_model_path`

---

## 6. 实验结果

> **AP（Average Precision）** 为 COCO 官方目标检测指标，综合衡量不同 IoU 阈值下的检测性能。

---

### 6.1 小规模设置：预训练 vs 微调

**COCO 子集（5k / 500）**

| Model      | AP@[0.5:0.95] | AP@0.50 | AP@0.75 |
| ---------- | ------------- | ------- | ------- |
| Pretrained | 0.519         | 0.675   | 0.566   |
| Fine-tuned | 0.542         | 0.690   | 0.593   |

* **AP 提升：+0.023（≈ +4.4%）**
* 说明：

  * pipeline 完整可用
  * 微调带来稳定、可观的性能提升

---

### 6.2 中等规模设置：num_queries 消融实验（10k / 1k）

在固定模型结构、训练轮数与优化策略的条件下，系统比较不同 `num_queries`：

| num_queries | AP@[0.5:0.95] | AP@0.50     | AP@0.75     | Training Time (2×4090) |
| ----------- | ------------- | ----------- | ----------- | ---------------------- |
| 900         | 0.552         | 0.703       | 0.605       | ~1h53m                 |
| 600         | **0.559**     | **0.713**   | **0.615**   | ~2h01m                 |
| 300         | 0.556–0.558   | 0.710–0.716 | 0.612–0.617 | ~1h58m                 |

#### 关键观察

* 将 `num_queries` 从 **900 降至 300**：

  * **未导致性能下降**
  * AP 保持在同一水平，甚至略有提升
* 表明在 COCO 规模检测任务中：

  * **Object queries 存在显著冗余**
  * 减少 queries 可在几乎不损失精度的情况下，降低计算开销

---

### 6.3 稳定性验证（固定 Seed + 重复实验）

* 固定随机种子（`seed=42`）
* 在 `num_queries=300` 下重复运行多次

结果示例：

* Run A：AP@[0.5:0.95] = 0.556
* Run B：AP@[0.5:0.95] = 0.558

**波动 ≤ 0.002**

说明：

* 结果波动处于随机噪声范围
* query resize（900 → 300）策略在当前实现中是**稳定且可靠的**

---

## 7. 当前结论

* 成功复现 GroundingDINO 在 COCO 上的训练与评估流程
* ODVG 数据格式转换正确，与 COCO eval 完全兼容
* 预训练模型具备较强基线性能
* 微调在不同数据规模下均能带来稳定性能提升
* **num_queries 可显著减少而不影响检测精度**
* 在固定 seed 条件下，实验结果高度稳定
* 当前 pipeline 可作为后续结构与训练策略研究的可靠基线
