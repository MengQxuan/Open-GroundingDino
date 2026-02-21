# Open-GroundingDINO 中文文档

## 项目简介

本项目基于开源仓库 **Open-GroundingDINO**，系统完成了从环境配置、数据集转换、分布式训练到推理测速的完整复现流程，并围绕 **Object Queries 冗余性与效率问题** 展开了系统性实验研究，进一步探索了 **Transformer 结构轻量化** 的可行路径。

**主要研究方向：**

- `num_queries` 对精度与速度的影响
- Query Pruning 的可行性
- BERT 冻结策略影响
- 推理效率瓶颈分析
- Transformer 层剪枝与跨模态推理加速
- 文本编码器替换（DistilBERT）

> 详细的核心原理、模型架构与技术说明，请参阅 [docs/PRINCIPLES.md](docs/PRINCIPLES.md)。

---

## 目录

- [实验环境](#实验环境)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [预训练权重](#预训练权重)
- [训练](#训练)
- [评估](#评估)
- [推理](#推理)
- [实验结果](#实验结果)
- [轻量化优化实验](#轻量化优化实验)
- [推理性能分析](#推理性能分析)
- [综合结论](#综合结论)

---

## 实验环境

### 硬件

| 项目   | 配置              |
| ------ | ----------------- |
| GPU    | RTX 4090 24GB ×2  |
| Driver | 560+              |
| CUDA   | 11.8 / 12.x       |
| 显存模式 | Default           |

### 软件

| 项目       | 版本             |
| ---------- | ---------------- |
| Python     | 3.10             |
| PyTorch    | 2.1.x (cu118)    |
| AMP        | 开启             |
| 分布式     | DDP (`torchrun`) |

---

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/MengQxuan/Open-GroundingDino.git
cd Open-GroundingDino
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 编译 CUDA 算子（必需）

```bash
cd groundingdino/models/GroundingDINO/ops
python setup.py build install
cd ../../../../
```

---

## 数据集准备

### COCO 2017

下载 [COCO 2017](https://cocodataset.org/) 数据集，包含：

- `train2017/`（训练集图像）
- `val2017/`（验证集图像）
- `annotations/instances_train2017.json`
- `annotations/instances_val2017.json`

### ODVG 格式转换

本项目训练需要将数据转换为 ODVG 格式（`.jsonl`）：

```bash
python tools/coco2odvg.py \
  -i /path/to/annotations/instances_train2017.json \
  -o /path/to/annotations/odvg_train2017.jsonl
```

ODVG 格式示例：

```json
{
  "filename": "image.jpg",
  "height": 693,
  "width": 1024,
  "detection": {
    "instances": [
      {"bbox": [262, 210, 323, 338], "label": 0, "category": "dog"},
      {"bbox": [164, 263, 252, 371], "label": 1, "category": "cat"}
    ]
  }
}
```

完整格式说明请参阅 [data_format.md](data_format.md)。

### 实验规模设置（10k / 1k）

本实验使用 COCO 2017 的子集，兼顾稳定性与实验成本：

| 项目  | 数量   |
| ----- | ------ |
| Train | 10,000 |
| Val   | 1,000  |
| Epoch | 5      |
| GPU   | 2×4090 |

配置文件：[config/datasets_coco_10k1k.json](config/datasets_coco_10k1k.json)

---

## 预训练权重

| 模块         | 配置                         |
| ------------ | ---------------------------- |
| Backbone     | Swin-T                       |
| Text Encoder | BERT-base-uncased            |
| 权重文件     | `groundingdino_swint_ogc.pth` |

下载预训练权重后放置于 `weights/` 目录，文本编码器亦放置于 `weights/bert-base-uncased/`。

---

## 训练

### 单 GPU 训练（Baseline）

```bash
torchrun --nproc_per_node=1 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_baseline \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 \
  --num_workers 8 \
  --amp \
  --save_log
```

### 双 GPU DDP 训练（推荐）

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

### 调整 num_queries

```bash
# q=600
torchrun --nproc_per_node=2 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_q600 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 num_queries=600 \
  --num_workers 8 --amp --save_log

# q=300
torchrun --nproc_per_node=2 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_q300 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 num_queries=300 \
  --num_workers 8 --amp --save_log
```

### 固定随机种子（可复现性验证）

```bash
torchrun --nproc_per_node=2 --master_port=29501 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_q300_seed42_run1 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 num_queries=300 \
  --seed 42 \
  --num_workers 8 --amp --save_log
```

### 冻结 BERT 训练

```bash
torchrun --nproc_per_node=2 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --output_dir outputs/10k1k_baseline_freeze_bert_encoder_ddp \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=weights/bert-base-uncased epochs=5 \
  --num_workers 4 --amp --save_log
```

### Transformer 层剪枝训练（enc=4, dec=3）

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
    epochs=20 \
    num_queries=300 \
    enc_layers=4 \
    dec_layers=3 \
  --num_workers 8 --amp --save_log
```

### 完整轻量化配置（Stage4：剪枝 + np2 + offset_clip + softmax_fp32 + DistilBERT）

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
    dec_n_points=2 \
    offset_clip=8 \
    softmax_fp32=True \
  --num_workers 8 --amp --save_log
```

---

## 评估

训练完成后，模型会自动在每个 epoch 结束时进行评估并输出 COCO AP 指标。最优模型保存为 `checkpoint_best_regular.pth`。

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

### 推理性能 Benchmark

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

### Caption 长度影响测速（profile_split 模式）

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

## 实验结果

### Baseline 结果（q=900，10k/1k，5 epoch）

| 指标   | 数值  |
| ------ | ----- |
| AP     | 0.552 |
| AP50   | 0.703 |
| AP75   | 0.605 |

训练时间：约 1:52:57（双卡）

---

### BERT 冻结实验

| 模型          | AP    | 训练时间  |
| ------------- | ----- | --------- |
| Normal        | 0.550 | 3:07:14   |
| Freeze BERT   | 0.454 | 2:53:43   |
| Normal (DDP2) | 0.552 | 1:52:57   |
| Freeze BERT (DDP2) | 0.490 | 未记录 |

**结论：** 冻结文本编码器会显著损伤性能（-0.096 AP），加速收益仅约 7%，后续实验不采用该策略。

---

### num_queries 消融实验（精度对比，10k/1k）

| Queries | AP            | AP50          | AP75          | 训练时间  |
| ------- | ------------- | ------------- | ------------- | --------- |
| 900     | 0.552         | 0.703         | 0.605         | 1:52:57   |
| 600     | **0.559**     | **0.713**     | **0.615**     | 2:01:19   |
| 300     | 0.556–0.558   | 0.710–0.716   | 0.612–0.617   | 1:57–1:59 |
| 200     | 0.551         | 0.710         | 0.606         | 1:57:31   |
| 50      | 0.537         | 0.699         | 0.582         | 1:53:19   |

**可复现性验证（Seed=42，q=300）：**

| Run  | AP    | 波动  |
| ---- | ----- | ----- |
| Run1 | 0.556 | —     |
| Run2 | 0.558 | ≤0.002 |

**关键发现：** queries 从 900 降至 200，AP 基本不变，说明 DETR-style queries 存在严重冗余。

---

### Query Pruning 实验

基于训练阶段 Top-K 选择进行裁剪：

| 设置              | AP    |
| ----------------- | ----- |
| q300 + prune200   | 0.541 |
| q300 + prune50    | 0.545 |

**结论：** 简单 Top-K Pruning 无法有效提升性能，原因是 Matching 阶段仍依赖全 queries，Transformer 计算仍是主瓶颈。

---

### 推理性能 Benchmark（Forward Only，Batch=4）

| Queries | Latency (ms) mean/p50/p90    | FPS mean/p50 |
| ------- | ----------------------------- | ------------ |
| 900     | 203.15 / 188.90 / 287.57      | 19.69 / 21.17 |
| 600     | 203.60 / 187.04 / 289.29      | 19.65 / 21.39 |
| 300     | 202.97 / 179.67 / 277.32      | 19.71 / 22.26 |
| 200     | 203.88 / 185.00 / 293.18      | 19.62 / 21.62 |
| 50      | 202.82 / 187.66 / 279.52      | 19.72 / 21.31 |

**结论：** 减少 queries 对推理速度影响极小，当前瓶颈在 Backbone + Cross-Attention，而非 queries 数量本身。

---

### 5K/0.5K 子集实验

使用 COCO 2017 的 5k/500 子集进行训练验证：

| 指标   | AP    | AP50  | AP75  |
| ------ | ----- | ----- | ----- |
| Run1   | 0.546 | 0.688 | 0.602 |
| Run2   | 0.547 | 0.695 | 0.597 |
| Run3   | 0.542 | 0.690 | 0.593 |

训练时间约 1:28:34。

---

## 轻量化优化实验

### 多阶段轻量化优化路线

| Stage  | 内容                              | 状态      |
| ------ | --------------------------------- | --------- |
| Stage1 | Layer Pruning（enc=4, dec=3）     | ✅ 完成   |
| Stage2 | np2 Sampling（dec_n_points=2）    | ✅ 完成   |
| Stage3 | Offset Clamp + Softmax FP32       | ✅ 完成   |
| Stage4 | DistilBERT 替换                   | ✅ 完成   |
| Stage5 | Text Encoder INT8 量化            | 🔄 进行中 |
| Stage6 | Vision/Decoder QAT                | ⏳ 规划中 |

---

### Stage1：Transformer 层剪枝

通过裁剪 Encoder / Decoder 层数（enc=4, dec=3），配合 `--finetune_ignore` 忽略被裁剪层参数：

| 配置       | Encoder | Decoder | AP@0.5:0.95   | Max Mem  |
| ---------- | ------- | ------- | ------------- | -------- |
| Baseline   | 6       | 6       | ~0.552–0.559  | ~12.2GB  |
| Pruned     | 4       | 3       | 0.519–0.533   | ~10.7GB  |

**结论：** 适度剪枝可显著减少计算量，精度下降约 3%，处于可接受范围。

详细结果（Epoch 15 最优）：

| 指标  | AP    | AP50  | AP75  | 显存     |
| ----- | ----- | ----- | ----- | -------- |
| 剪枝后 | 0.533 | 0.697 | 0.583 | ~10.7GB  |

---

### Stage2：Cross-Attention 降采样（dec_n_points=2）

在 MSDeformAttn 中减少采样点数量：

| 设置             | AP     | 延迟 (ms) | FPS     |
| ---------------- | ------ | --------- | ------- |
| np=4 (default)   | ~0.533 | 187.39    | 21.35   |
| np=2             | ~0.519 | 186.33    | 21.47   |

**结论：** 降采样可减少 Attention 计算量，对精度影响较小（约 -0.014 AP），推理延迟基本不变。

---

### Stage3：Offset Clamp + Softmax FP32 稳定化

引入以下优化以提升量化与混合精度稳定性：

- Sampling offset 限幅（`offset_clip=8`）
- Softmax 保持 FP32（`softmax_fp32=True`）
- BBox Head 保持 FP32

| 设置                       | AP     | 显存     |
| -------------------------- | ------ | -------- |
| 剪枝 + np2                 | ~0.519 | 10.7GB   |
| 剪枝 + np2 + offset_clip   | ~0.510 | 10.4GB   |
| 剪枝 + np2 + offset + fp32 | ~0.511 | 10.7GB   |

**作用：** 抑制数值抖动，提高后续量化鲁棒性。

---

### Stage4：DistilBERT 替换文本编码器

将文本编码器从 BERT-base 替换为 DistilBERT：

```bash
text_encoder_type=weights/distilbert-base-uncased
```

修复内容：
- 3D attention mask 不兼容问题
- `token_type_ids` / `position_ids` 冲突
- 统一 TextEncoderShell 封装

精度与显存表现：

| 模型       | AP     | Max Mem |
| ---------- | ------ | ------- |
| BERT       | ~0.511 | ~10.7GB |
| DistilBERT | ~0.512 | ~9.8GB  |

Stage4 最终配置：`Pruning + np2 + offset_clip + softmax_fp32 + DistilBERT`

**达成指标：**
- AP ≈ 0.512
- 显存 < 10GB
- 推理稳定（延迟约 187ms，FPS 约 21.3）

---

## 推理性能分析

### Caption 长度对推理性能影响（Stage5 Profiling）

测试设置：`batch_size=4, num_queries=300, warmup=50, iters=200`

| Caption Length | T_tokenize (ms) | T_text_enc (ms) | T_vision+dec (ms) | T_total (ms) |
| -------------- | --------------- | --------------- | ----------------- | ------------ |
| 4              | 0.36            | 4.20            | 154.10            | 160.14       |
| 16             | 0.45            | 4.19            | 158.32            | 164.42       |
| 32             | 0.59            | 4.20            | 163.06            | 169.34       |
| 64             | 0.88            | 4.17            | 183.53            | 190.11       |
| 128            | 1.40            | 4.33            | 215.05            | 222.42       |

**Caption 长度增长的延迟归因（len=4 → 128）：**

| 模块           | 增量 (ms) | 占比 (%) |
| -------------- | --------- | -------- |
| Tokenize       | +1.04     | 1.7%     |
| Text Encoder   | +0.13     | 0.2%     |
| Vision+Decoder | +60.95    | 97.9%    |
| **Total**      | +62.28    | 100%     |

---

### Stage6-0：Encoder 内部细分 Profiling

细分各模块的延迟随 caption_len 的变化：

| caption_len | T_total | enc_fusion | enc_text | enc_msdef | dec   |
| ----------- | ------- | ---------- | -------- | --------- | ----- |
| 4           | 171.72  | 40.97      | 4.76     | 49.80     | 15.33 |
| 16          | 176.42  | 42.78      | 4.86     | 49.91     | 16.02 |
| 32          | 182.96  | 47.00      | 4.61     | 50.84     | 15.39 |
| 64          | 201.24  | 62.19      | 4.88     | 50.18     | 15.51 |
| 128         | 235.25  | 86.88      | 5.08     | 50.10     | 15.88 |

**核心结论：** encoder 的增量几乎全部来自 `enc_fusion`（+45.9 ms，占 ~98%），`enc_msdef` 和 `dec` 基本不随 caption 长度变化。

**BiMultiHeadAttention 内部细分（caption 4 → 128）：**

| 算子          | cap=4 (ms) | cap=128 (ms) | 增量   |
| ------------- | ---------- | ------------ | ------ |
| attn_proj     | 17.08      | 17.15        | +0.07  |
| attn_scores   | 1.93       | 7.46         | +5.54  |
| attn_softmax  | 1.25       | 18.21        | +16.96 |
| attn_ctx      | 5.94       | 24.22        | +18.28 |
| attn_out      | 2.61       | 2.63         | +0.02  |

**关键发现：** 长 caption 下的主要瓶颈是 `softmax + context bmm + QK^T`（均与文本 token 数 ntxt 强相关），线性投影（attn_proj ≈ 17ms）为常数项。

---

## 综合结论

### 已验证结论

| 编号 | 结论                                              |
| ---- | ------------------------------------------------- |
| 1    | Pipeline 完整可复现                               |
| 2    | 微调稳定有效，5 epoch 可获得良好性能              |
| 3    | Queries 存在严重冗余（900→200 精度基本不变）       |
| 4    | 冻结 BERT 明显降精度（-0.096 AP），加速收益有限   |
| 5    | 简单 Top-K Pruning 收益有限                       |
| 6    | 推理瓶颈非 queries 数量，在于 Backbone + Attention |
| 7    | 层剪枝 + 降采样具备较高性价比（AP 仅降约 3%）     |
| 8    | 长 caption 瓶颈在 enc_fusion 的 attention 计算   |
| 9    | DistilBERT 可在保持精度同时降低显存约 10%         |

### 核心发现

> GroundingDINO 当前架构对 queries 数量不敏感，主要瓶颈在 Backbone + Deformable Attention + Cross-modal Fusion。

**原因分析：**

- 多层 Attention 稀释冗余
- Hungarian Matching 过滤低质量 queries
- Decoder 多轮 Refinement 提升质量

### 后续优化方向

1. **Stage5（文本塔 INT8）：** 主要收益在显存，端到端速度提升有限
2. **Stage6（Decoder QAT / Mixed Precision）：** 应聚焦 enc_fusion 的 attention 结构：
   - **Token Pruning：** 减少有效文本 token 数（直接削减 scores/softmax/ctx 规模）
   - **线性层 INT8：** 对 attn_proj（≈17ms 常数项）做 INT8/QAT，softmax 保持 FP32

---

## 数据格式

详见 [data_format.md](data_format.md)。

## 项目原理

详见 [docs/PRINCIPLES.md](docs/PRINCIPLES.md)。

## 许可证

本项目遵循原始仓库的许可证协议，详见 [LICENSE](LICENSE)。
