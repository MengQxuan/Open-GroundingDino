# Open-GroundingDINO 核心原理、模型架构与实验流程
本文档详细介绍 GroundingDINO 的核心思想、模型架构、训练与推理流程，以及本项目各阶段轻量化优化的技术原理。
## 目录
- [1. GroundingDINO 概述](#1-groundingdino-概述)
- [2. 模型架构](#2-模型架构)
  - [2.1 整体结构](#21-整体结构)
  - [2.2 视觉主干（Backbone）](#22-视觉主干backbone)
  - [2.3 文本编码器（Text Encoder）](#23-文本编码器text-encoder)
  - [2.4 特征融合编码器（Feature Enhancer）](#24-特征融合编码器feature-enhancer)
  - [2.5 语言引导查询选择（Language-Guided Query Selection）](#25-语言引导查询选择language-guided-query-selection)
  - [2.6 跨模态解码器（Cross-Modality Decoder）](#26-跨模态解码器cross-modality-decoder)
  - [2.7 输出头（Prediction Heads）](#27-输出头prediction-heads)
- [3. 关键技术机制](#3-关键技术机制)
  - [3.1 Object Queries 与 DETR 继承](#31-object-queries-与-detr-继承)
  - [3.2 Deformable Attention（可变形注意力）](#32-deformable-attentionmsdeformattn)
  - [3.3 BiMultiHeadAttention（双向多头注意力）](#33-bimultiheadattention双向多头注意力)
  - [3.4 Hungarian Matching（匈牙利匹配）](#34-hungarian-matching匈牙利匹配)
  - [3.5 ODVG 数据格式](#35-odvg-数据格式)
- [4. 训练流程](#4-训练流程)
  - [4.1 数据预处理](#41-数据预处理)
  - [4.2 损失函数](#42-损失函数)
  - [4.3 分布式训练配置](#43-分布式训练配置)
- [5. 推理流程](#5-推理流程)
- [6. 实验关键步骤详解](#6-实验关键步骤详解)
  - [6.1 num_queries 消融](#61-num_queries-消融)
  - [6.2 Query Pruning 机制](#62-query-pruning-机制)
  - [6.3 Transformer 层剪枝](#63-transformer-层剪枝)
  - [6.4 MSDeformAttn 采样点降采样](#64-msdeformattn-采样点降采样)
  - [6.5 数值稳定化（Offset Clamp + Softmax FP32）](#65-数值稳定化offset-clamp--softmax-fp32)
  - [6.6 DistilBERT 替换 BERT](#66-distilbert-替换-bert)
  - [6.7 enc_fusion 瓶颈分析](#67-enc_fusion-瓶颈分析)
---
## 1. GroundingDINO 概述
**GroundingDINO** 是一种开放集目标检测（Open-Set Object Detection）模型，能够根据任意自然语言描述检测图像中的目标，突破了传统闭集检测器只能识别固定类别的限制。
核心创新：
* **文本引导检测**：输入文本 prompt（如 "person . car . dog"），模型输出与 prompt 匹配的边界框及置信度。
* **跨模态融合**：在 Transformer Encoder 阶段将视觉特征与文本特征深度融合（Feature Enhancer），实现语义对齐。
* **DETR-style 端到端检测**：无需 NMS，通过 Hungarian Matching 实现端到端训练。
与传统检测器的比较：

| 特性       | 传统检测器（YOLO/Faster-RCNN） | GroundingDINO      |
| :--------- | :----------------------------- | :----------------- |
| 类别数量   | 固定闭集                       | 开放集，任意类别   |
| 输入模态   | 仅图像                         | 图像 + 文本        |
| NMS 后处理 | 需要                           | 不需要             |
| 跨模态融合 | 无                             | Encoder 深度融合   |
---
## 2. 模型架构
### 2.1 整体结构
```
输入图像 ──────────────────────────────────────────────────────────┐
                                                                   │
文本 Prompt ──► Text Tokenizer ──► Text Encoder（BERT/DistilBERT）──► 文本特征（L×d）
                                                                   │
图像 ──► Visual Backbone（Swin-T） ──► 多尺度特征图（C3/C4/C5）       │
         │                                                         │
         └──► Feature Enhancer（Transformer Encoder）               │
                 ├── MSDeformAttn（视觉自注意力）                    │
                 ├── BiAttentionBlock（视觉-文本双向交叉注意力）◄─────┘
                 └── FeedForward                                   │
                          │                                        │
                          ▼                                        │
              Language-Guided Query Selection（Top-K）             │
                          │                                        │
                          ▼                                        │
              Cross-Modality Decoder                               │
                 ├── Self-Attention（queries 间）                   │
                 ├── Cross-Attention（queries ◄─► 视觉特征）        │
                 ├── Cross-Attention（queries ◄─► 文本特征）◄────────┘
                 └── FeedForward                                   │
                          │                                        │
                          ▼                                        │
              Prediction Heads                                     │
                 ├── BBox Head（MLP，预测边界框）                    │
                 └── Grounding Head（与文本对齐，预测类别概率）       │
```
### 2.2 视觉主干（Backbone）
本项目采用 **Swin-T（Swin Transformer Tiny）** 作为视觉主干：
* 输入：图像（H×W×3）
* 输出：多尺度特征图（C3、C4、C5），通道数分别为 96/192/384
* 特点：
  * Shifted Window Attention，兼顾局部与全局感知
  * 层次化特征提取，适合多尺度目标检测
  * 与 Deformable DETR 中的多尺度特征图格式对齐
多尺度特征经 Feature Projection 映射为统一维度（d_model=256），然后进入 Encoder。
### 2.3 文本编码器（Text Encoder）
**BERT-base-uncased（或 DistilBERT-base-uncased）**：
* 输入：文本 prompt（如 "person . car . dog"），经 Tokenizer 转换为 token 序列
* 输出：token-level 特征序列（L×d_bert），d_bert=768（BERT）或 768（DistilBERT）
* 作用：将文本语义编码为高维特征向量，为后续视觉-文本融合提供语义锚点
**关键实现细节：**
```
文本 Prompt ──► BertTokenizer ──► input_ids（L）
                              ──► attention_mask（L）
                              ──► token_type_ids（BERT 专用）
```
* BERT 使用 12 层 Transformer，参数量约 110M
* DistilBERT 使用 6 层 Transformer，参数量约 66M（BERT 的 60%），通过知识蒸馏保持约 97% 的 BERT 性能
* 文本特征维度（768）通过线性层投影为 d_model=256，与视觉特征维度对齐
**为何不能冻结 BERT：**
> 在 ODVG 微调场景下，BERT 需要根据新类别语义动态调整文本特征的分布，以与视觉特征对齐。冻结后，文本特征固定，视觉侧无法充分学习类别对齐，导致 AP 显著下降（实测 -0.062）。
### 2.4 特征融合编码器（Feature Enhancer）
**Feature Enhancer** 是 GroundingDINO 的核心创新模块，通过双向跨模态注意力（BiAttentionBlock）将视觉特征与文本特征深度融合。
结构（每层交替堆叠）：
```
for each Encoder Layer:
    视觉特征 ──► MSDeformAttn（视觉自注意力，捕捉视觉上下文）
                 │
                 ▼
    视觉特征 ──► BiAttentionBlock（视觉 ◄──► 文本 双向融合）◄── 文本特征
                 │
                 ▼
    视觉特征 ──► FFN（FeedForward Network）
```
本项目 Baseline 配置：6 层 Encoder（`enc_layers=6`）  
轻量化配置：4 层 Encoder（`enc_layers=4`）
**BiAttentionBlock 详解（见 Section 3.3）**
### 2.5 语言引导查询选择（Language-Guided Query Selection）
在 Decoder 之前，从 Encoder 输出的视觉特征中选取 Top-K（`num_queries`）个候选位置作为初始 queries：
```
Encoder 输出（N×d）──► 与文本特征计算相关性分数
                  ──► Top-K 选择（k = num_queries）
                  ──► 初始化 Query Position（Anchor Points）
                  ──► 初始化 Query Content（Decoder 输入）
```
**为何 queries 可以大幅削减：**
> * Language-Guided 选择已经利用文本语义过滤掉无关候选位置，保留与 prompt 相关的 Top-K 区域
> * 即使 num_queries 从 900 降至 200，选出的高质量候选仍足以覆盖大多数目标
> * 因此 AP 几乎不变（实测从 0.552 降至 0.551）
### 2.6 跨模态解码器（Cross-Modality Decoder）
**Cross-Modality Decoder** 基于 Deformable DETR Decoder，但增加了与文本特征的交叉注意力：
```
for each Decoder Layer:
    queries ──► Self-Attention（queries 间相互感知）
              │
              ▼
    queries ──► Cross-Attention（queries ◄──► 视觉特征，MSDeformAttn）
              │
              ▼
    queries ──► Cross-Attention（queries ◄──► 文本特征，传统 Cross-Attn）
              │
              ▼
    queries ──► FFN
              │
              ▼
    queries ──► Bbox Refinement（迭代精修边界框）
```
本项目 Baseline 配置：6 层 Decoder（`dec_layers=6`）  
轻量化配置：3 层 Decoder（`dec_layers=3`）
**Decoder 延迟分析：**
Decoder 延迟约 15ms，且基本不随 caption 长度变化（因为文本侧 Cross-Attention 以文本特征长度为 Key，维度较小）。
### 2.7 输出头（Prediction Heads）
每个 query 对应两个输出头：
1. **BBox Head（MLP）**：预测归一化边界框坐标 (cx, cy, w, h)
   * 采用 Sigmoid 激活，输出 [0,1] 范围内的归一化坐标
   * 多层 Decoder 迭代精修（Iterative Refinement）
2. **Grounding Head（与文本对齐）**：
   * 计算每个 query 特征与每个文本 token 的相似度
   * 通过 max-pooling 得到按类别词的分类概率
   * 训练时与 ground-truth 文本 span 对齐，推理时输出置信度
**为何不需要 NMS：**
> Hungarian Matching 保证每个 query 最多匹配一个 ground-truth，且训练目标约束不同 queries 预测不同位置，因此预测结果天然无重复，无需后处理 NMS。
---
## 3. 关键技术机制
### 3.1 Object Queries 与 DETR 继承
GroundingDINO 继承了 DETR 系列的核心设计：**Object Queries**。
* **Queries 的本质**：一组可学习的位置编码（Anchor Points）+ 内容嵌入（Content Embedding），每个 query 代表模型"关注"图像中某个位置的潜在目标
* **冗余性根源**：
  * 多层 Decoder Attention 在训练过程中自发地将 queries 分配到不同区域
  * 对于稀疏目标场景（如 COCO，图像平均目标数 < 10），900 个 queries 中绝大多数是空置的
  * Hungarian Matching 每次训练只为真正有目标的 queries 分配监督信号，其余 queries 以"no-object"类惩罚
**实测冗余程度（本项目消融）：**
| num_queries | AP    | 与 q=900 差异 |
| ----------- | ----- | ------------- |
| 900         | 0.552 | —             |
| 600         | 0.559 | +0.007（更好） |
| 300         | 0.558 | +0.006        |
| 200         | 0.551 | -0.001        |
| 50          | 0.537 | -0.015        |
> 结论：q=200 时几乎无损，说明 DETR-style queries 存在严重冗余。
### 3.2 Deformable Attention（MSDeformAttn）
**可变形注意力**是 Deformable DETR 引入的高效注意力机制：
* **问题**：传统自注意力对 N 个特征点计算两两相关性，复杂度 O(N²)
* **解决方案**：每个 query 只关注图像中少量（`n_points`）动态采样位置，复杂度降为 O(N × n_points)
```python
# MSDeformAttn 伪代码
for each query q:
    # 预测采样偏移量（相对于参考点的偏移）
    offsets = linear(q)  # shape: [n_heads, n_levels, n_points, 2]
    offsets = clamp(offsets, -offset_clip, +offset_clip)  # 可选：限幅稳定化
    
    # 对多尺度特征图进行双线性插值采样
    sampled_feats = grid_sample(feature_maps, ref_points + offsets)
    
    # 计算注意力权重并加权求和
    attn_weights = softmax(linear(q))  # shape: [n_heads, n_levels, n_points]
    output = sum(attn_weights * sampled_feats)
```
**`dec_n_points`（本项目优化 Stage2）：**
* 默认 `dec_n_points=4`，即每个 query 在每个尺度每个头采样 4 个点
* 设置 `dec_n_points=2` 可降低约 50% 的采样计算量
* 实测精度损失极小（AP 从 0.519 到 0.519），性价比高
**`offset_clip`（本项目优化 Stage3）：**
* 限制采样偏移量的绝对值，防止采样超出特征图边界过远
* 引入局部稀疏先验，提高混合精度和后续量化的数值稳定性
### 3.3 BiMultiHeadAttention（双向多头注意力）
**BiMultiHeadAttention** 是 Feature Enhancer 中的核心跨模态融合操作，实现视觉特征与文本特征的双向交叉注意力：
```
Vision Features (V): shape [B, N_vis, d]     ← 图像所有特征点
Text Features   (T): shape [B, N_txt, d]     ← 文本所有 token
# 视觉 ← 文本：图像特征关注相关文本 token
V' = V + CrossAttn(Q=V, K=T, V=T)
# 文本 ← 视觉：文本特征关注相关视觉区域
T' = T + CrossAttn(Q=T, K=V, V=V)
```
**计算复杂度分析：**
* `attn_scores (QK^T)`：O(N_vis × N_txt × d)
* `attn_softmax`：O(N_vis × N_txt)（沿 N_txt 维度）
* `attn_ctx (probs×V)`：O(N_vis × N_txt × d)
> 当 N_txt（caption 长度）增大时，以上三项均随之增长，这正是 enc_fusion 成为长 caption 场景主要瓶颈的原因（见 Section 6.7）。
**enc_fusion 结构分解（实测延迟，caption_len=4）：**
```
enc_fusion（40.97ms）
├── fusion_ln（LayerNorm）：1.74ms
├── fusion_attn（BiMultiHeadAttention）：33.13ms
│   ├── attn_proj（线性投影 QKV）：17.08ms  ← 常数项
│   ├── attn_scores（QK^T）：1.93ms
│   ├── attn_softmax：1.25ms
│   ├── attn_ctx（probs×V）：5.94ms
│   └── attn_out（输出投影）：2.61ms  ← 常数项
└── fusion_resid（残差 + drop_path）：3.41ms
```
### 3.4 Hungarian Matching（匈牙利匹配）
GroundingDINO（继承自 DETR）使用 Hungarian Algorithm（匈牙利算法）进行训练时的目标分配：
**目标**：在 N 个预测 queries 与 M 个 ground-truth 目标之间，寻找最优的一一对应关系（N >> M）。
**匹配成本**：
```
Cost(query_i, gt_j) = λ₁ × L_cls(pred_i, gt_j) 
                    + λ₂ × L_bbox_L1(pred_i, gt_j)
                    + λ₃ × L_bbox_GIoU(pred_i, gt_j)
```
**训练损失**：
```
L_total = Σ_{matched pairs} [L_cls + L_bbox_L1 + L_bbox_GIoU]
        + Σ_{unmatched queries} L_no_object
```
**关键性质：**
* 每个 query 最多匹配一个 GT，每个 GT 最多被一个 query 匹配
* 未匹配的 queries 被分配"no-object"类，以 BCE 损失惩罚
* 保证预测结果无重复，无需 NMS
**这也是 Query Pruning 不奏效的根本原因：**
> 在 Forward Pass 中即使只保留 Top-K queries 进行 NMS，Hungarian Matching 在训练时仍需要全量 queries 参与分配，Transformer 的计算量不变，因此简单 Top-K Pruning 无法减少训练成本。
### 3.5 ODVG 数据格式
**ODVG（Object Detection + Visual Grounding）** 是本项目使用的统一多任务数据格式：
```
OD 任务（detection 字段）：
  输入  = 图像 + label_map 中所有类别名称拼接的 prompt
  输出  = 各类别目标的边界框
VG 任务（grounding 字段）：
  输入  = 图像 + 自然语言描述 caption
  输出  = caption 中各 phrase 对应的边界框
```
统一格式使得 OD 与 VG 任务可以在同一模型、同一损失函数下联合训练，充分利用不同来源的标注数据。
---
## 4. 训练流程
### 4.1 数据预处理
```
原始 COCO JSON ──► tools/coco2odvg.py ──► ODVG JSONL 格式
                                         ├── 图像路径
                                         ├── 目标边界框
                                         └── 类别标签 → label_map 映射 → 文本 prompt
数据增强：
  ├── RandomResize（随机缩放）
  ├── RandomCrop（随机裁剪）
  ├── ColorJitter（颜色抖动）
  └── RandomHorizontalFlip（随机水平翻转）
```
### 4.2 损失函数
| 损失项              | 说明                                     |
| ------------------- | ---------------------------------------- |
| `L_cls`（分类）     | 文本-视觉对齐的 Focal Loss（BCE 变体）    |
| `L_bbox_L1`（L1）   | 边界框坐标回归的 L1 Loss                  |
| `L_bbox_GIoU`（GIoU）| Generalized IoU Loss，处理非重叠框       |
| `L_no_object`       | 未匹配 queries 的背景分类损失            |
加权组合：
```
L_total = λ_cls × L_cls + λ_L1 × L_bbox_L1 + λ_GIoU × L_bbox_GIoU + λ_bg × L_no_object
```
### 4.3 分布式训练配置
```bash
# DDP 分布式训练（2×GPU）
torchrun --nproc_per_node=2 --master_port=29501 main.py \
  --config_file config/cfg_coco.py \
  --datasets config/datasets_coco_10k1k.json \
  --amp \        # 混合精度训练（AMP）
  --save_log
```
| 配置项       | 参数                          |
| ------------ | ----------------------------- |
| 优化器       | AdamW                         |
| 学习率       | 1e-4（backbone）/ 1e-4（其余）|
| Scheduler    | MultiStep LR                  |
| Batch Size   | 4 per GPU（总 batch=8）       |
| AMP          | 开启（FP16 + FP32 混合）      |
| Gradient Clip| 0.1                           |
**预训练权重加载（finetune_ignore）：**
> 使用 `--finetune_ignore` 参数可显式忽略指定层的预训练权重，使这些层从随机初始化开始训练。在层剪枝场景中，通过忽略"多余层"的权重，可以无冲突地将 6 层预训练模型加载到 4 层架构中。
---
## 5. 推理流程
```
输入：图像 + 文本 Prompt（如 "person . car . dog"）
Step 1：文本处理
  prompt ──► BertTokenizer ──► input_ids, attention_mask
  Text Encoder（BERT/DistilBERT）──► 文本特征（L×768 ──► L×256）
Step 2：视觉特征提取
  图像 ──► Swin-T Backbone ──► 多尺度特征图（C3/C4/C5, 均映射至 256-dim）
Step 3：跨模态特征融合（Feature Enhancer）
  for each Encoder Layer（共 4 或 6 层）:
      视觉特征 ──► MSDeformAttn ──► BiAttentionBlock（与文本融合）──► FFN
Step 4：语言引导 Query 选择
  融合后视觉特征 ──► 与文本计算相关性 ──► Top-K 选择 ──► 初始化 queries
Step 5：跨模态解码（Decoder）
  for each Decoder Layer（共 3 或 6 层）:
      queries ──► Self-Attn ──► Cross-Attn（视觉）──► Cross-Attn（文本）──► FFN
      ──► Iterative BBox Refinement
Step 6：预测输出
  最终 queries ──► BBox Head ──► 边界框坐标（cx, cy, w, h）
                ──► Grounding Head ──► 与文本 token 相似度 ──► 类别置信度
Step 7：后处理
  按置信度阈值（box_threshold）过滤
  按文本阈值（text_threshold）过滤
  无需 NMS
```
---
## 6. 实验关键步骤详解
### 6.1 num_queries 消融
**实验目的**：验证 Object Queries 的冗余程度。
**实验方法**：保持其他参数不变，仅改变 `num_queries`（900 / 600 / 300 / 200 / 50），在 COCO 10k/1k 上训练 5 epochs，记录最终 AP。
**为何 queries 冗余：**
1. **Language-Guided Selection** 已从视觉特征中选出与文本最相关的 Top-K 位置，大量 queries 在该阶段已被过滤
2. **多层 Decoder Refinement** 使少量 queries 也能逐步精修到正确位置
3. **Focal Loss** 对背景 queries 使用较小权重，减弱了冗余 queries 的负面影响
4. **COCO 目标密度低**：平均每张图 < 10 个目标，200 个 queries 完全够用
### 6.2 Query Pruning 机制
**实验方法**：在 `num_queries=300` 的模型中，Decoder 前使用 Top-K 再次筛选（`query_prune_topk=200` 或 `50`），仅保留分数最高的 queries 进行 Decoder 前向。
**为何效果有限：**
* Hungarian Matching 在计算匹配成本时仍需所有 queries 的预测结果
* Transformer Encoder（Feature Enhancer）计算量不随 query 数量变化
* 主要计算瓶颈（Backbone + Encoder）无法通过 query 层面的 pruning 减少
**结论**：Query Pruning 只能减少少量 Decoder 计算，无法解决根本性的推理瓶颈。
### 6.3 Transformer 层剪枝
**目标**：降低 Encoder/Decoder 层数以减少整体计算量。
**关键技术：`--finetune_ignore`**
```bash
# 显式忽略第 4、5 层（index from 0）的预训练权重
--finetune_ignore \
  transformer.encoder.layers.4 \
  transformer.encoder.layers.5 \
  transformer.encoder.text_layers.4 \
  transformer.encoder.text_layers.5 \
  transformer.encoder.fusion_layers.4 \
  transformer.encoder.fusion_layers.5 \
  transformer.decoder.layers.3 \
  ... （对应 BBox Head 层也需忽略）
```
加载流程：
```
预训练权重（6 层）──► 过滤 finetune_ignore 中的参数 ──► 剩余权重加载到新模型（4 层）
新模型中被忽略的层 ──► 随机初始化 ──► 从头训练（epochs=20）
```
**精度与显存权衡（Stage1）：**
| 配置     | AP    | 显存 (MB) | 训练时间/epoch |
| -------- | ----- | --------- | -------------- |
| 6+6 层   | 0.556 | 11,560    | ~22min         |
| 4+3 层   | 0.533 | 10,742    | ~16min         |
> 层剪枝使显存降低约 700MB，训练加速约 27%，AP 损失约 0.023。
### 6.4 MSDeformAttn 采样点降采样
**原理**：Deformable Attention 中，每个 query 在每个特征层每个注意力头采样 `dec_n_points` 个位置：
```
采样计算量 ∝ n_heads × n_levels × dec_n_points
```
设置 `dec_n_points=2`（默认为 4），可减少约 50% 的采样和加权求和计算。
**实现方式**：通过 `--finetune_ignore` 忽略 `cross_attn.sampling_offsets` 和 `cross_attn.attention_weights` 的预训练权重，重新初始化适配 `n_points=2` 的参数维度。
### 6.5 数值稳定化（Offset Clamp + Softmax FP32）
**背景**：混合精度（AMP，FP16/BF16）训练中，以下操作容易出现数值不稳定：
* 采样偏移量过大导致采样位置超出特征图范围
* Softmax 在 FP16 下对大数值输入溢出（inf）或梯度爆炸
**Offset Clamp（`offset_clip=8`）：**
```python
offsets = offsets.clamp(-offset_clip, +offset_clip)
```
* 限制采样偏移量绝对值 ≤ 8 个像素（相对单位），引入局部稀疏先验
* 防止采样点飘移至离参考点过远的区域，提高注意力的局部性
**Softmax FP32（`softmax_fp32=True`）：**
```python
attn_weights = F.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
```
* Softmax 强制在 FP32 精度下计算，再转回 FP16 输出
* 避免 FP16 下大 logit 值导致的 softmax 溢出
* 对量化后的 INT8 模型尤为重要（量化精度范围更小）
### 6.6 DistilBERT 替换 BERT
**动机**：DistilBERT 是 BERT 的知识蒸馏版本，参数量约为 BERT 的 60%，在 GLUE benchmark 上保留约 97% 的性能。
**技术挑战与修复：**
| 问题                         | 原因                                                  | 修复方案                              |
| ---------------------------- | ----------------------------------------------------- | ------------------------------------- |
| 3D attention mask 不兼容      | GroundingDINO 使用 3D mask（B×L×L），DistilBERT 期望 2D | 在 TextEncoderShell 中统一 mask 维度  |
| `token_type_ids` 冲突         | DistilBERT 不使用 `token_type_ids`，传入会报错         | 在 Shell 中过滤掉 `token_type_ids`    |
| `position_ids` 冲突           | 同上，DistilBERT 内部自动生成 position_ids             | 同上                                  |
**TextEncoderShell 封装方案：**
```python
class TextEncoderShell(nn.Module):
    def __init__(self, encoder_type):
        # 统一接口，内部自动处理 BERT / DistilBERT 的差异
        ...
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # 自动过滤 DistilBERT 不支持的参数
        # 统一输出格式 last_hidden_state: [B, L, d]
        ...
```
**性能对比（Stage4 最终配置）：**
| 模型       | AP    | Max Mem (MB) |
| ---------- | ----- | ------------ |
| BERT       | 0.511 | 10,730       |
| DistilBERT | 0.512 | 9,764        |
> DistilBERT 显存节省约 1GB，精度基本持平，推理延迟相当（文本塔本身只占总延迟的约 2%）。
### 6.7 enc_fusion 瓶颈分析
**分析工具**：`--profile_split` 参数，结合 `torch.cuda.Event` 精确计时各子模块。
**核心结论（caption_len 从 4 增长至 128）：**
```
总延迟增量：+63.53 ms
增量归因：
├── tokenize：+0.994 ms（1.6%）
├── text_enc：+0.061 ms（0.1%）
├── backbone：+0.099 ms（0.2%）
├── decoder：+0.544 ms（0.9%）
└── enc_fusion：+45.909 ms（72.3%）← 主要瓶颈
    └── fusion_attn（BiMultiHeadAttention）：
        ├── attn_softmax：+16.96 ms（二次方增长 w.r.t. N_txt）
        ├── attn_ctx：+18.28 ms（二次方增长 w.r.t. N_txt）
        └── attn_scores：+5.54 ms（二次方增长 w.r.t. N_txt）
```
**数学原理**：`BiAttentionBlock` 中视觉-文本的交叉注意力计算复杂度为：
```
O(N_vis × N_txt × d)
```
当 `N_txt`（caption 长度）增大时，`scores`、`softmax`、`ctx_bmm` 均线性增长（总体对 `N_txt` 二次方），导致 `enc_fusion` 随 caption 长度显著增长。
**优化方向：**
1. **Token Pruning**（最有效）：在 `enc_fusion` 之前，对文本 token 进行稀疏选择，减小有效 `N_txt`
2. **Flash Attention**：利用 IO-aware 融合算子减少 `softmax + ctx_bmm` 的显存带宽开销
3. **INT8/QAT（次优先）**：对 `attn_proj`（常数项 ~17ms）做量化，稳定且接近工业实践
