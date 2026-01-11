# DeepFashion-Captioner: 基于 ResNet-LSTM-Attention 的服饰图像描述系统

本项目是一个针对服饰领域的专业图像描述生成系统（Image Captioning）。它结合了经典且稳健的 **ResNet-101 (CNN Encoder)** 与 **Dual-Layer LSTM (RNN Decoder)** 架构，并通过 **Adaptive Attention** 机制建立视觉与文本的有效对齐。

为了追求极致的生成质量，项目引入了 **Qwen2-VL** 对原始数据进行深度重构增强，并针对长句生成、数值稳定性及高分辨率特征提取进行了多项硬核优化。

---

## 🚀 核心亮点 (Core Highlights)

- **大模型驱动的数据增强 (V3)**:
  利用顶级多模态模型 **Qwen2-VL-7B** 对 DeepFashion 原始短语标注进行了“长句化”改写。生成的描述涵盖了材质、剪裁、风格及穿着效果，使生成文本从“词组”进化为“自然语言”。
- **高分辨率特征捕捉**:
  针对服饰细节，将输入分辨率提升至 **518x518**（采用 LANCZOS 高保真插值），相比传统的 224 尺寸能多保留约 5 倍的像素细节。
- **卓越系统的稳定性**:
  - **Attention Logits Clamp**: 对注意力得分进行 $[-10, 10]$ 的强力钳制，杜绝了模型在长序列解码时因溢出引发的 `NaN` 风险。
  - **Label Smoothing**: 引入 $0.05$ 的平滑系数，有效提升了模型在细分款式上的泛化性能。
- **全生命周期可视化**:
  支持 **Attention Map 实时生成**。模型每产出一个单词，均可可视化其在原图上的“关注”落点，实现了预测逻辑透明化。

---

## 🏗️ 架构详述 (Technical Architecture)

### 1. 编码器 (CNN Encoder)
- **骨干网**: 预训练 `ResNet-101`。
- **空间一致性**: 移除了最后的 AvgPool 和 FC 层，通过 `AdaptiveAvgPool2d` 强制输出 $14 \times 14 \times 2048$ 的特征矩阵。
- **微调策略**: 采用分层学习率，仅对高层语义 Block 进行微调。

### 2. 注意力机制 (Adaptive Attention)
- **原理**: 计算当前 Decoder 隐藏状态与 $196$ 个像素区域的相关度分布。
- **稳定性增强**: 加入 Logits 钳制层，确保在长序列生成时梯度流平稳。

### 3. 解码器 (RNN Decoder)
- **单元**: 双层堆叠式 LSTM。
- **输入融合**: 每个时间步将 Word Embedding 与经注意力算法加权的视觉特征进行串联。

---

## 📂 项目文件清单

### 模型与推理 (Core & Inference)
- **[models.py](models.py)**: 模型定义（Encoder, Attention, Decoder）。
- **[app.py](app.py)**: **Web 交互界面**。基于 Gradio 打造，支持 Beam Search 调节。
- **[caption.py](caption.py)**: **推理与可视化脚本**。可生成单图描述及注意力热力图。

### 数据流水线 (Data Pipeline)
- **[json_aug.py](json_aug.py)**: **增强核心**。调用 Qwen2-VL 批量处理图片的脚本（支持并发读取）。
- **[karpathy_aug.py](karpathy_aug.py)**: 建立增强文本与数据集划分 (Train/Val/Test) 的映射关系。
- **[create_input_files.py](create_input_files.py)**: 最终预处理。生成模型专用的二进制/索引文件。

### 训练与评测 (Train & Eval)
- **[train_aug.py](train_aug.py)**: **主训练脚本**。集成 Label Smoothing 与高分辨率支持。
- **[eval.py](eval.py)**: 自动化指标测量 (BLEU-4, METEOR, ROUGE-L)。

---

## 📊 关键资产 (Assets)

| 资产文件 | 描述 |
| :--- | :--- |
| `13_checkpoint_...pth.tar` | **V3 权重**: 包含 13 轮微调后的最优参数。 |
| `WORDMAP_...json` | **全局字典**: 针对服饰增强预料优化的专业词表。 |
| `captions_aug.json` | **增强语料源**: Qwen2-VL 生成的原始文本数据库。 |

---

## 🛠️ 快速运行指南

### 1. 环境准备
```bash
# 建议环境：Python 3.10+ / CUDA 12.0+
pip install -r requirements.txt
```

### 2. 启动 Web 展示界面
```bash
python app.py
```

### 3. 一键可视化热力图
```bash
python caption.py --img "your_image.jpg" --beam_size 1
```

### 4. 自动化对比评测
```bash
python eval.py
```

---

## 📈 训练核心超参数 reference
- **输入尺寸**: $518 \times 518$ (LANCZOS 高保真缩放)
- **优化器**: Adam ($LR_{Encoder}=5e-5, LR_{Decoder}=4e-4$)
- **学习率衰减**: 连续 8 个 Epoch 指标不提升则缩小学习率。
- **正则化**: Label Smoothing ($0.05$), Dropout ($0.5$)
- **硬件标准**: NVIDIA RTX 5090 (32GB)，建议推理显存 $\ge$ 12GB。

---

## 💡 开发者备注 (Dev Notes)
- **数值保护**: 已经预置了 Logits Clamp，不需要担心 `exp(x)` 带来的 `inf` 溢出。
- **训练建议**: 若 GPU 显存较小（< 24G），请在 `train_aug.py` 中适当降低 `batch_size`。
- **Windows 用户**: 务必将 `num_workers` 设为 0 以避免路径与并发冲突。
