
# Open-GroundingDINO 复现与微调实验记录

本项目基于开源仓库 **Open-GroundingDINO**，完成了从环境配置、数据集转换（COCO → ODVG）、模型加载、训练到评估的**完整复现流程**，并在 COCO 子集上验证了 **GroundingDINO 预训练模型的微调有效性**。  
实验在 **单卡 RTX 4090（24GB）** 环境下完成，支持 AMP 混合精度训练。

---

## 1. 实验环境

- GPU：RTX 4090 24GB（单卡）
- CUDA：12.x
- Python：3.8
- PyTorch：支持 AMP
- 训练方式：`torchrun` 单进程
- 操作系统：Linux

---

## 2. 项目目标

- 复现 GroundingDINO 在 COCO 数据集上的训练与评估流程
- 理解并跑通 **ODVG（Open-Domain Visual Grounding）** 数据格式
- 在不修改模型结构的前提下，验证：
  - 预训练模型在下游数据集上的表现
  - 微调（fine-tuning）是否能带来稳定性能提升
- 为后续改进模型结构 / 训练策略提供可靠基线

---

## 3. 数据集与格式转换

### 3.1 使用数据集

- **COCO 2017**
  - Train：`train2017`
  - Val：`val2017`

### 3.2 ODVG 格式转换

GroundingDINO 训练阶段使用 **ODVG jsonl** 格式描述检测数据。  
本项目使用仓库自带脚本完成转换：

```bash
python tools/coco2odvg.py \
  -i data/coco/annotations/instances_train2017.json \
  -o data/coco/annotations/odvg_train2017.jsonl \
  --idmap coco2017
````

验证集保持 COCO 原生 `instances_val2017.json`，用于标准 COCO eval。

---

## 4. 子集（Smoke Test）设置

由于 COCO 全量训练耗时较长（单 epoch 约 6h+），为了快速验证复现正确性，构建了一个 **子集验证配置**：

* 训练集：COCO train2017 抽样 **5000 张**
* 验证集：COCO val2017 抽样 **500 张**
* 输入分辨率：

  * `data_aug_scales = [480]`
  * `data_aug_max_size = 800`
* 训练 epoch：3～5
* 目的：

  * 快速验证 pipeline 是否正确
  * 对比预训练与微调效果
  * 支持高频实验迭代

---

## 5. 预训练模型

使用官方提供的预训练权重：

* `groundingdino_swint_ogc.pth`
* Backbone：Swin-T
* Text Encoder：BERT-base-uncased

权重通过 `--pretrain_model_path` 加载。

---

## 6. 实验结果

```
AP（Average Precision）是 COCO 官方目标检测评估指标，综合衡量模型在不同 IoU 阈值下的检测精度与召回能力。
```

### 6.1 预训练模型（0 训练，直接评估）

在 **COCO 子集验证集（500 张）** 上直接评估预训练模型：

```
AP@[IoU=0.50:0.95] = 0.519
AP@0.50            = 0.675
AP@0.75            = 0.566
```

评估耗时约 **67 秒**。

---

### 6.2 微调后模型（Fine-tuning）

在相同子集设置下进行微调训练后，评估结果为：

```
AP@[IoU=0.50:0.95] = 0.542
AP@0.50            = 0.690
AP@0.75            = 0.593
```

相较于预训练模型：

* **AP 提升：+0.023**
* 相对提升约 **4.4%**
* 训练时间：约 **1 小时 28 分钟**

该结果表明：

* 训练与评估流程完全跑通
* 微调在子集上带来了稳定、可观的性能提升

---

## 7. 当前结论

* 成功复现了 GroundingDINO 在 COCO 数据集上的训练与评估流程
* ODVG 数据格式转换正确，训练与 COCO eval 兼容
* 预训练模型具备较强基线性能
* 微调在子集上能够稳定提升检测精度
* 当前 pipeline 可作为后续模型与训练策略改进的可靠基线

---

## 8. 后续计划

* 冻结 / 解冻 Text Encoder（BERT）的消融实验
* `num_queries` 数量对性能与训练速度的影响分析
* 不同输入分辨率与数据增强策略的对比
* 推理可视化与 open-vocabulary grounding demo
* 尝试在更大数据集或混合数据集上进行训练

---