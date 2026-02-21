# Open-GroundingDINO 复现与轻量化优化实验

> 本项目基于开源仓库 **Open-GroundingDINO**，完整复现了 GroundingDINO 的训练与评估流程，并围绕 **Object Queries 冗余性与结构轻量化** 展开了系统性消融实验研究。
>
> 核心原理与模型架构详见 [principles.md](./principles.md)。

## 目录

- [项目简介](#项目简介)
- [实验环境](#实验环境)
- [快速开始](#快速开始)
  - [环境安装](#环境安装)
  - [模型权重](#模型权重)
  - [数据集准备](#数据集准备)
- [训练](#训练)
- [推理](#推理)
- [数据集格式（ODVG）](#数据集格式odvg)
- [实验结果](#实验结果)
  - [Baseline 结果](#baseline-结果)
  - [BERT 冻结实验](#bert-冻结实验)
  - [num_queries 消融实验](#num_queries-消融实验)
  - [Query Pruning 实验](#query-pruning-实验)
  - [推理性能 Benchmark](#推理性能-benchmark)
  - [结构轻量化优化实验](#结构轻量化优化实验)
  - [Caption 长度对推理性能影响](#caption-长度对推理性能影响)
  - [Encoder 内部细分 Profiling](#encoder-内部细分-profiling)
- [综合结论](#综合结论)

---

## 项目简介

本项目基于开源仓库 **Open-GroundingDINO**，系统完成了从环境配置、数据集转换、分布式训练到推理测速的完整复现流程，并围绕 **Object Queries 冗余性与效率问题** 展开了系统性实验研究。

**主要研究方向：**

* `num_queries` 对精度与速度的影响
* Query Pruning 的可行性验证
* BERT 冻结策略对性能的影响
* Transformer 层剪枝与跨模态推理加速
* DistilBERT 替换文本编码器
* Caption 长度对推理延迟的影响分析

**项目目标：**

* 完整复现 GroundingDINO 训练与评估流程
* 跑通 ODVG 数据格式的完整 pipeline
* 构建稳定 baseline，分析 queries 冗余问题
* 探索模型轻量化潜力

---

## 实验环境

### 硬件

| 配置项  | 参数              |
| ------- | ----------------- |
| GPU     | RTX 4090 24GB × 2 |
| Driver  | 560+              |
| CUDA    | 11.8 / 12.x       |
| 显存模式 | Default           |

### 软件

| 配置项      | 参数                   |
| ----------- | ---------------------- |
| Python      | 3.10                   |
| PyTorch     | 2.1.x (cu118)          |
| AMP         | 开启                   |
| 分布式      | DDP (`torchrun`)       |

---

## 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/MengQxuan/Open-GroundingDino.git
cd Open-GroundingDino

# 安装依赖
pip install -r requirements.txt
```

### 模型权重

下载预训练权重并放置于 `weights/` 目录：

| 模块           | 配置                          |
| -------------- | ----------------------------- |
| Backbone       | Swin-T                        |
| Text Encoder   | BERT-base-uncased             |
| 预训练权重     | `groundingdino_swint_ogc.pth` |

```bash
mkdir -p weights
# 下载预训练权重至 weights/ 目录
# 下载 bert-base-uncased 至 weights/bert-base-uncased/
# 可选：下载 distilbert-base-uncased 至 weights/distilbert-base-uncased/
```

### 数据集准备

以 COCO 2017 为例：

```bash
# 数据集结构
datasets/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

将 COCO 标注转换为 ODVG 格式：

```bash
python tools/coco2odvg.py \
  -i annotations/instances_train2017.json \
  -o annotations/odvg_train2017.jsonl
```

数据规模设置（10k/1k 兼顾稳定性与实验成本）：

| 项目  | 数量   |
| ----- | ------ |
| Train | 10,000 |
| Val   | 1,000  |
| Epoch | 5      |
| GPU   | 2×4090 |

---

## 训练

### 基础训练（Baseline，q=900）

```bash
torchrun --nproc_per_node=2 --master_port=29501 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_q900_ddp2_baseline \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 \
  --num_workers 8 \
  --amp \
  --save_log
```

### 消融实验：调整 num_queries

```bash
# q=600
torchrun --nproc_per_node=2 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_q600 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 num_queries=600 \
  --num_workers 8 --amp --save_log

# q=300（同理可设置 num_queries=300/200/50）
torchrun --nproc_per_node=2 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_q300 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 num_queries=300 \
  --num_workers 8 --amp --save_log
```

### 固定随机种子

```bash
torchrun --nproc_per_node=2 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_q300_seed42_run1 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 num_queries=300 \
  --seed 42 \
  --num_workers 8 --amp --save_log
```

### Query Pruning 实验

```bash
torchrun --nproc_per_node=2 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_q300_prune200 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 \
             num_queries=300 query_prune_topk=200 \
  --seed 42 --num_workers 8 --amp --save_log
```

### 结构剪枝训练（Enc=4, Dec=3）

```bash
torchrun --nproc_per_node=2 --master_port=29501 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/prune_enc4_dec3_q300_bert \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --finetune_ignore \
    transformer.encoder.layers.4 \
    transformer.encoder.layers.5 \
    transformer.encoder.text_layers.4 \
    transformer.encoder.text_layers.5 \
    transformer.encoder.fusion_layers.4 \
    transformer.encoder.fusion_layers.5 \
    transformer.decoder.layers.3 \
    transformer.decoder.layers.4 \
    transformer.decoder.layers.5 \
    transformer.decoder.bbox_embed.3 \
    transformer.decoder.bbox_embed.4 \
    transformer.decoder.bbox_embed.5 \
  --options \
    text_encoder_type=weights/bert-base-uncased \
    epochs=20 num_queries=300 enc_layers=4 dec_layers=3 \
  --num_workers 8 --amp --save_log
```

### Stage4：全量轻量化配置（剪枝 + np2 + offset_clip + softmax_fp32 + DistilBERT）

```bash
torchrun --nproc_per_node=2 --master_port=29501 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --finetune_ignore \
    transformer.encoder.layers.4 transformer.encoder.layers.5 \
    transformer.encoder.text_layers.4 transformer.encoder.text_layers.5 \
    transformer.encoder.fusion_layers.4 transformer.encoder.fusion_layers.5 \
    transformer.decoder.layers.3 transformer.decoder.layers.4 transformer.decoder.layers.5 \
    transformer.decoder.bbox_embed.3 transformer.decoder.bbox_embed.4 transformer.decoder.bbox_embed.5 \
    transformer.decoder.layers.0.cross_attn.sampling_offsets.weight \
    transformer.decoder.layers.0.cross_attn.sampling_offsets.bias \
    transformer.decoder.layers.0.cross_attn.attention_weights.weight \
    transformer.decoder.layers.0.cross_attn.attention_weights.bias \
    transformer.decoder.layers.1.cross_attn.sampling_offsets.weight \
    transformer.decoder.layers.1.cross_attn.sampling_offsets.bias \
    transformer.decoder.layers.1.cross_attn.attention_weights.weight \
    transformer.decoder.layers.1.cross_attn.attention_weights.bias \
    transformer.decoder.layers.2.cross_attn.sampling_offsets.weight \
    transformer.decoder.layers.2.cross_attn.sampling_offsets.bias \
    transformer.decoder.layers.2.cross_attn.attention_weights.weight \
    transformer.decoder.layers.2.cross_attn.attention_weights.bias \
  --options \
    text_encoder_type=weights/distilbert-base-uncased \
    epochs=20 num_queries=300 enc_layers=4 dec_layers=3 \
    dec_n_points=2 offset_clip=8 softmax_fp32=True \
  --num_workers 8 --amp --save_log
```

---

## 推理

### 单张图片推理

```bash
python tools/inference_on_a_image.py \
  --config_file tools/GroundingDINO_SwinT_OGC.py \
  --checkpoint_path weights/groundingdino_swint_ogc.pth \
  --image_path test.png \
  --text_prompt "person, car, dog" \
  --output_dir outputs/inference_test \
  --box_threshold 0.3 \
  --text_threshold 0.25
```

### 推理速度测试

```bash
python tools/benchmark_infer.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --device cuda \
  --num_workers 4 \
  --num_queries 300 \
  --warmup 50 \
  --iters 200 \
  --amp \
  --forward_only
```

### Caption 长度 Profiling 测试

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
```

---

## 数据集格式（ODVG）

ODVG 格式采用 JSONL 文件，每行一个 JSON 对象：

* **目标检测**数据集使用 `detection` 字段，并需要额外的 `label_map` 文件。
* **视觉定位**数据集使用 `grounding` 字段。

```json
{
  "filename": "image.jpg",
  "height": 693,
  "width": 1024,
  "detection": {
    "instances": [
      { "bbox": [262,210,323,338], "label": 0, "category": "dog" },
      { "bbox": [164,263,252,371], "label": 1, "category": "cat" }
    ]
  },
  "grounding": {
    "caption": "a wire hanger with a paper cover",
    "regions": [
      { "bbox": [20,215,985,665], "phrase": "a paper cover" },
      { "bbox": [19,19,982,671],  "phrase": "a wire hanger" }
    ]
  }
}
```

### label_map 格式

OD 数据需提供 label_map，索引从 `"0"` 开始：

```json
{"0": "person", "1": "bicycle", "2": "car", ...}
```

### 数据集配置文件

```json
{
  "train": [
    {
      "root": "path/coco_2017/train2017/",
      "anno": "path/coco_2017/annotations/coco2017_train_odvg.jsonl",
      "label_map": "path/coco_2017/annotations/coco2017_label_map.json",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "path/coco_2017/val2017",
      "anno": "config/instances_val2017.json",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
```

---

## 实验结果

> 所有实验均在 COCO 2017 数据集（10k train / 1k val）上进行，使用 2×RTX 4090，开启 AMP，DDP 分布式训练。

---

### Baseline 结果

**标准 Baseline（q=900，5 epochs）**

| 指标   | 数值  |
| ------ | ----- |
| AP     | 0.552 |
| AP50   | 0.703 |
| AP75   | 0.605 |
| AP_S   | 0.389 |
| AP_M   | 0.587 |
| AP_L   | 0.702 |
| AR@1   | 0.418 |
| AR@10  | 0.708 |
| AR@100 | 0.780 |

Max GPU 显存：12,154 MB

---

### BERT 冻结实验

| 模型               | AP    | AP50  | AP75  |
| ------------------ | ----- | ----- | ----- |
| Normal（BERT 正常） | 0.552 | 0.703 | 0.605 |
| Freeze BERT         | 0.490 | 0.632 | 0.541 |

**结论：**

> 冻结文本编码器会显著损伤性能（-0.062 AP），且几乎没有带来训练速度提升，因此不适合 ODVG 微调任务。BERT 侧对目标类别的分类对齐贡献极大，不能冻结。

---

### num_queries 消融实验

#### 精度对比（10k/1k，5 epochs）

| Queries | AP          | AP50        | AP75        | Max Mem (MB) |
| ------- | ----------- | ----------- | ----------- | ------------ |
| 900     | 0.552       | 0.703       | 0.605       | 12,154       |
| 600     | **0.559**   | **0.713**   | **0.615**   | —            |
| 300     | 0.556–0.558 | 0.710–0.716 | 0.612–0.617 | 11,560       |
| 200     | 0.551       | 0.710       | 0.606       | —            |
| 50      | 0.537       | 0.699       | 0.582       | —            |

#### 稳定性验证（Seed=42，q=300）

| 运行 | AP    | AP50  | AP75  |
| ---- | ----- | ----- | ----- |
| Run1 | 0.556 | 0.710 | 0.612 |
| Run2 | 0.558 | 0.716 | 0.617 |

波动 ≤ 0.002，结果高度稳定。

**核心发现：**

> * queries 从 900 → 200：AP 基本不变，几乎无性能损失。
> * queries 降至 50 时才出现明显精度下降（-0.015 AP）。
> * 说明 DETR-style 的 Object Queries 存在严重冗余。

---

### Query Pruning 实验

基于训练阶段 Top-K 选择进行裁剪：

| 设置              | AP    | AP50  | AP75  |
| ----------------- | ----- | ----- | ----- |
| q=300（基准）      | 0.556 | 0.710 | 0.612 |
| q300 + prune200   | 0.541 | 0.701 | 0.596 |
| q300 + prune50    | 0.545 | 0.702 | 0.601 |

**结论：**

> 简单 Top-K Pruning 无法有效提升性能，反而略有下降。原因在于 Hungarian Matching 阶段仍依赖全量 queries，Transformer 计算仍是主瓶颈，简单裁剪无法绕过该开销。

---

### 推理性能 Benchmark

**Forward Only 测试（batch=4，warmup=50，iters=200）**

| Queries | Latency mean (ms) | Latency p50 (ms) | Latency p90 (ms) | FPS (mean) | FPS (p50) |
| ------- | ----------------- | ---------------- | ---------------- | ---------- | --------- |
| 900     | 203.15            | 188.90           | 287.57           | 19.69      | 21.17     |
| 600     | 203.60            | 187.04           | 289.29           | 19.65      | 21.39     |
| 300     | 202.97            | 179.67           | 277.32           | 19.71      | 22.26     |
| 200     | 203.88            | 185.00           | 293.18           | 19.62      | 21.62     |
| 50      | 202.82            | 187.66           | 279.52           | 19.72      | 21.31     |

**结论：**

> 减少 queries 数量对推理速度影响**极小**（各组延迟几乎相同），说明推理瓶颈**不在 queries 数量本身**，而在 Backbone 与 Deformable Attention 等计算模块。

---

### 结构轻量化优化实验

#### Stage1：Transformer 层剪枝（Enc=4, Dec=3）

通过裁剪 Encoder / Decoder 层数降低计算复杂度，采用 `--finetune_ignore` 忽略被裁剪层参数，实现无冲突加载预训练权重。

| 配置     | Encoder | Decoder | Queries | AP          | Max Mem (MB) |
| -------- | ------- | ------- | ------- | ----------- | ------------ |
| Baseline | 6       | 6       | 300     | ~0.556      | 11,560       |
| Pruned   | 4       | 3       | 300     | ~0.519–0.533 | 10,742      |

> 适度剪枝可显著减少显存和计算量，精度下降约 3%，处于可接受范围。

#### Stage2：Cross-Attention 降采样（dec_n_points=2）

在 MSDeformAttn 中减少采样点数量：

| 设置            | AP     | Max Mem (MB) |
| --------------- | ------ | ------------ |
| np=4（default） | ~0.519 | 10,742       |
| np=2            | ~0.519 | 10,727       |

> 降采样可减少 Attention 计算量，对精度影响极小。

#### Stage3：Offset Clamp + Softmax FP32 数值稳定化

引入以下数值稳定措施：

* Sampling offset 限幅（offset_clip=8）
* Softmax 保持 FP32
* BBox Head 保持 FP32

| 设置                          | AP     | Max Mem (MB) |
| ----------------------------- | ------ | ------------ |
| 剪枝 + np2 + offset_clip      | ~0.510 | 10,368       |
| 剪枝 + np2 + offset + FP32    | ~0.511 | 10,730       |

> 数值稳定化措施抑制了混合精度下的数值抖动，提高后续量化鲁棒性。

#### Stage4：DistilBERT 替换文本编码器

将文本编码器从 BERT-base 替换为 DistilBERT，并修复 3D attention mask 不兼容、token_type_ids 冲突等问题，最终采用统一 TextEncoderShell 封装。

| 配置                              | AP     | Max Mem (MB) |
| --------------------------------- | ------ | ------------ |
| 剪枝 + np2 + offset + FP32（BERT） | ~0.511 | 10,730       |
| + DistilBERT（Stage4 最终）        | ~0.512 | 9,764        |

**Stage4 最终配置（Pruning + np2 + offset_clip + softmax_fp32 + DistilBERT）：**

| 指标     | 数值    |
| -------- | ------- |
| AP       | 0.512   |
| AP50     | 0.679   |
| AP75     | 0.559   |
| Max Mem  | < 10 GB |

> DistilBERT 在保持精度的同时降低显存占用约 ~1 GB，但端到端速度提升有限。

#### 轻量化优化阶段汇总

| Stage  | 内容                        | AP     | 状态     |
| ------ | --------------------------- | ------ | -------- |
| Baseline | 标准 q=300 + BERT           | ~0.556 | ✅       |
| Stage1 | Layer Pruning（enc4, dec3） | ~0.533 | ✅       |
| Stage2 | np2 降采样                  | ~0.519 | ✅       |
| Stage3 | Offset + FP32 稳定化        | ~0.511 | ✅       |
| Stage4 | DistilBERT 替换             | ~0.512 | ✅       |
| Stage5 | Text INT8 量化              | —      | 🔄 进行中 |
| Stage6 | Vision QAT（Decoder）       | —      | ⏳ 规划中 |

---

### Caption 长度对推理性能影响

**测试设置**：batch=4，num_queries=300，warmup=50，iters=200，AMP 开启，forward_only 模式，启用 `--profile_split`。

#### 各模块延迟随 Caption 长度变化（均值，ms）

| Caption Length | T_tokenize | T_text_enc | T_vision+dec | T_total |
| -------------- | ---------- | ---------- | ------------ | ------- |
| 4              | 0.36       | 4.20       | 154.10       | 160.14  |
| 16             | 0.45       | 4.19       | 158.32       | 164.42  |
| 32             | 0.59       | 4.20       | 163.06       | 169.34  |
| 64             | 0.88       | 4.17       | 183.53       | 190.11  |
| 128            | 1.40       | 4.33       | 215.05       | 222.42  |

#### Caption 长度增长的延迟贡献（len=4 → 128）

| 模块           | 增量 (ms) | 占比 (%) |
| -------------- | --------- | -------- |
| Tokenize       | +1.04     | 1.7%     |
| Text Encoder   | +0.13     | 0.2%     |
| Vision+Decoder | +60.95    | 97.9%    |
| Total          | +62.28    | 100%     |

**结论：**

> * Text Encoder（约 4ms）基本不随 caption 长度变化，**文本塔不是端到端性能瓶颈**。
> * 超过 97% 的延迟增长来源于 Vision+Decoder 阶段（准确说是 Transformer 中的文本-视觉融合模块）。
> * 仅量化文本塔对端到端速度提升有限，主要价值在于降低显存占用。

---

### Encoder 内部细分 Profiling

**测试设置**：固定 batch=4，num_queries=300，caption_len ∈ {4, 16, 32, 64, 128}，重复 token "a"。

#### 端到端与模块级延迟（mean，ms）

| caption_len | T_total | text_enc | backbone | enc    | dec    | enc_fusion | enc_msdef |
| ----------- | ------- | -------- | -------- | ------ | ------ | ---------- | --------- |
| 4           | 171.72  | 4.11     | 42.96    | 97.02  | 15.33  | 40.97      | 49.80     |
| 16          | 176.42  | 4.21     | 42.99    | 99.15  | 16.02  | 42.78      | 49.91     |
| 32          | 182.96  | 4.12     | 43.76    | 103.94 | 15.39  | 47.00      | 50.84     |
| 64          | 201.24  | 4.20     | 42.93    | 118.78 | 15.51  | 62.19      | 50.18     |
| 128         | 235.25  | 4.17     | 43.05    | 143.61 | 15.88  | 86.88      | 50.10     |

#### enc_fusion 内部细分（fusion_attn 各子操作，mean，ms）

| caption_len | attn_proj | attn_scores(QK^T) | attn_softmax | attn_ctx(probs×V) | attn_out |
| ----------- | --------- | ----------------- | ------------ | ----------------- | -------- |
| 4           | 17.08     | 1.93              | 1.25         | 5.94              | 2.61     |
| 16          | 17.11     | 2.00              | 2.07         | 6.73              | 2.61     |
| 32          | 17.07     | 2.57              | 3.06         | 8.46              | 2.60     |
| 64          | 17.08     | 4.20              | 8.61         | 15.41             | 2.63     |
| 128         | 17.15     | 7.46              | 18.21        | 24.22             | 2.63     |

**关键发现：**

> * `text_enc`（约 4ms）、`backbone`（约 43ms）、`enc_msdef`（约 50ms）基本不随 caption 长度变化。
> * **`enc_fusion`（BiAttentionBlock）从 40.97ms 增长至 86.88ms，是主要瓶颈**，占 encoder 增量的约 98%。
> * 在 `enc_fusion` 内部，增长几乎全部来自 `attn_softmax`（+16.96ms）和 `attn_ctx`（+18.28ms）——均与文本 token 数 `ntxt` 成二次方关系。
> * `attn_proj`（线性投影，约 17ms）和 `attn_out`（约 2.6ms）基本为常数项。

---

## 综合结论

### 已验证结论

1. ✅ **Pipeline 完整可复现**：从环境配置到 COCO 评估全流程跑通
2. ✅ **微调稳定有效**：固定种子后 AP 波动 ≤ 0.002
3. ✅ **Queries 存在严重冗余**：900 → 200 精度基本不变
4. ✅ **冻结 BERT 明显降精度**：-0.062 AP，且加速收益极小
5. ✅ **简单剪枝收益有限**：Top-K Pruning 反而略有下降
6. ✅ **推理瓶颈非 queries**：减少 queries 对速度几乎无影响

### 核心结论

> **GroundingDINO 当前架构对 queries 数量不敏感。**
>
> 原因：多层 Attention 稀释冗余、Hungarian Matching 过滤、Decoder 多轮 Refinement。

> **性能瓶颈位于 Transformer Encoder 内的 enc_fusion（跨模态融合模块）。**
>
> 长 caption 下，`BiAttentionBlock` 中的 softmax 与 context bmm 随文本 token 数呈二次方增长，是端到端延迟上升的根本原因。

### 后续优化方向

基于以上实验分析，推荐以下两类主线优化方向（Stage6）：

1. **降低 fusion 有效文本 token 数（Token Pruning / Top-K tokens）**
   直接削减 `scores/softmax/ctx` 的规模，对长 caption 场景最有效。

2. **对 fusion 线性层做 INT8/QAT，softmax 保持 FP32**
   主要收益在 `attn_proj` 等常数项（约 17ms），稳定且接近工业部署实践。

### 综合指标对比

| 模型配置                        | AP    | Max Mem (MB) | 备注             |
| ------------------------------- | ----- | ------------ | ---------------- |
| Baseline（q=900, BERT）          | 0.552 | 12,154       | 标准基准         |
| q=300（最优 queries 数量）       | 0.558 | 11,560       | 精度最优         |
| Stage1（层剪枝）                 | 0.533 | 10,742       | 精度损失约 3%    |
| Stage4（全量轻量化 + DistilBERT） | 0.512 | 9,764        | 显存 < 10 GB     |

---

*更多技术细节、模型架构原理及关键模块说明，请参见 [principles.md](./principles.md)。*
ENDOFFILE
<command with shellId: 0 is still running after 10 seconds. The command is still running but hasn't produced output yet. It may be waiting for input or still processing. Use read_bash to continue waiting, write_bash if you think it needs input, or stop_bash to stop it.>