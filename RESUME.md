# 简历项目经历

---

## 基于 GroundingDINO 的开放词汇目标检测模型结构压缩与推理加速研究　　2026.01 – 至今

### 项目背景

GroundingDINO 是当前开放集目标检测领域的代表性工作，能够根据任意自然语言描述在图像中定位目标，突破了传统检测器只能识别固定类别的限制。然而，其标准配置（BERT-base + 6 层 Encoder/Decoder + 900 Object Queries）参数量大、推理延迟高（约 200ms/batch）、显存占用超过 12 GB，在边缘部署和实时推理场景中存在明显瓶颈。本项目在此背景下，围绕**模型结构冗余分析与系统性轻量化**展开研究，旨在探索精度-速度-显存之间的最优权衡。

### 主要内容

- **完整复现训练与评估流程**：基于开源仓库 Open-GroundingDINO，在 COCO 2017（10k/1k）上完成从数据预处理（ODVG 格式转换）、2×RTX 4090 DDP 分布式训练（AMP 混合精度）到 COCO AP 评估的完整 pipeline，建立稳定 Baseline（AP = 0.552，AP50 = 0.703）。

- **系统性消融实验**：设计并执行多组控制变量实验，涵盖 num_queries（900 / 600 / 300 / 200 / 50）消融、BERT 冻结策略对比、Query Pruning 可行性验证，量化分析各因素对精度与速度的影响，得出 **queries 存在严重冗余**（900→200 时 AP 几乎不变）、推理瓶颈并非 queries 数量等核心结论。

- **多阶段结构轻量化**：分四个阶段递进压缩模型：
  - **Stage 1**：利用 `--finetune_ignore` 机制对 Transformer 进行层剪枝（Encoder 6→4 层，Decoder 6→3 层），实现无冲突加载预训练权重；
  - **Stage 2**：对 MSDeformAttn 的采样点进行降采样（dec_n_points 4→2），降低 Deformable Attention 计算量；
  - **Stage 3**：引入 Offset Clamp（offset_clip=8）+ Softmax FP32 双重数值稳定化策略，抑制混合精度下的溢出与梯度抖动，为后续量化奠定基础；
  - **Stage 4**：将文本编码器从 BERT-base（110M）替换为 DistilBERT（66M），解决 3D attention mask 不兼容与 token_type_ids 冲突，设计统一 TextEncoderShell 封装层。

- **精细化 Profiling 与瓶颈定位**：基于 `torch.cuda.Event` 实现模块级精确计时体系（`--profile_split`），对 tokenize / text_enc / backbone / enc_fusion / enc_msdef / decoder 等子模块进行分层计时，将推理延迟随 caption 长度增长的来源精准定位至 **Feature Enhancer 中的 BiAttentionBlock（enc_fusion）**，揭示其 softmax 与 context bmm 随文本 token 数呈 **O(N_vis × N_txt)** 二次方增长的规律。

### 性能提升

与标准 Baseline（BERT + 6+6 层 + 900 queries，AP = 0.552）相比，最终轻量化配置（DistilBERT + 4+3 层 + 300 queries + np2 + Offset Clamp + Softmax FP32）取得以下提升：

| 指标 | Baseline | 轻量化最终配置 | 提升幅度 |
| :--- | :--- | :--- | :--- |
| AP（COCO 2017 val） | 0.552 | 0.512 | 精度损失约 7%（换取以下三项提升） |
| 训练时间/epoch | 0:21:14 | 0:15:03 | **提速约 29%** |
| 显存峰值 | 12,154 MB | 9,764 MB | **节省约 2.4 GB（↓20%）** |
| 显存是否低于单卡 10 GB | 否 | **是** | 可部署至更多 GPU 型号 |

此外，通过 Profiling 实验量化证明：文本编码器仅占端到端延迟约 2%，而 enc_fusion 在 caption 长度从 4 增长至 128 时延迟增量占总增量的 **97.9%**，为后续 Token Pruning 和 Flash Attention 优化方向提供了明确的数据支撑。

### 困难与解决方案

1. **DistilBERT 接口不兼容**：GroundingDINO 内部使用 3D attention mask（B×L×L），而 DistilBERT 仅接受 2D mask；同时 DistilBERT 不支持 `token_type_ids` 和 `position_ids` 显式传入。  
   → **方案**：设计 `TextEncoderShell` 统一封装层，在 forward 中自动检测编码器类型，按需裁剪不支持的参数并统一 mask 维度，实现 BERT / DistilBERT 无缝切换，无需修改模型主体代码。

2. **层剪枝时预训练权重维度不匹配**：直接将 6 层预训练权重加载至 4 层模型会引发参数形状冲突。  
   → **方案**：利用 `--finetune_ignore` 机制，在权重加载时显式过滤被裁剪层的参数，使剩余层继承预训练权重，被裁剪层从随机初始化重新训练，完全避免冲突。

3. **混合精度（AMP）下训练不稳定**：在 FP16/BF16 下，MSDeformAttn 的采样偏移量容易漂移至特征图边界之外，softmax 对大 logit 值容易溢出（inf），导致梯度爆炸。  
   → **方案**：组合使用 Offset Clamp（将采样偏移量绝对值限制在 ±8 个像素单位内）与 Softmax FP32（强制在 FP32 精度下计算 softmax 后转回 FP16），有效抑制数值抖动并提升量化鲁棒性。

4. **推理瓶颈难以定位**：减少 queries 数量后推理速度几乎不变，与直觉不符，无法判断瓶颈位置。  
   → **方案**：自主构建基于 `torch.cuda.Event` 的模块级 Profiling 体系，对 Encoder 内部 enc_fusion / enc_msdef / enc_text 等子模块分别精确计时，最终将瓶颈定位至 BiAttentionBlock 内的 softmax 与 context bmm 操作，并以 caption 长度实验定量验证二次方增长规律。

### 核心价值

- **工程能力**：独立完成从多 GPU 分布式训练配置、ODVG 数据格式转换、推理 Benchmark 到细粒度 Profiling 的完整工程链路，代码健壮、实验可复现（固定种子后 AP 波动 ≤ 0.002）。
- **系统性思维**：通过消融实验建立"queries 冗余验证 → 层剪枝 → 注意力压缩 → 编码器替换"的递进优化框架，每一步均有量化实验支撑，逻辑严谨。
- **问题定位能力**：面对"减少 queries 却不提速"的反直觉现象，不依赖经验猜测，而是自主搭建 Profiling 工具将瓶颈精准定位至 BiAttentionBlock，形成可复用的性能分析方法论。
- **量化落地意识**：实验设计（Offset Clamp + Softmax FP32）从一开始就考虑量化友好性，实验结论（enc_fusion 为瓶颈）直接指向 Token Pruning 和 INT8/QAT 等工业落地优化方向，研究具有实际工程价值。
