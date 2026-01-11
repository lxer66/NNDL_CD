# DeepFashion-Captioning-Plus: 工业级服饰图像描述生成系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5-orange.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

本项目是一个专注于服饰垂直领域的双架构图像描述生成系统（Image Captioning）。它涵盖了从经典 **ResNet-LSTM** 架构到尖端 **DINOv2 + Flan-T5 + LoRA** 架构的完整实现，旨在为时尚电商提供高精度的服饰细节描述、面料分析及风格总结。

---

## 🌟 核心特性

- **双重前沿架构**: 同时支持经典经典的轻量化 Baseline 和基于视觉大模型的 SOTA 方案。
- **大模型驱动的数据增强 (Gen-AI Augmented)**:
  - **文本增强**: 利用 **Qwen2-VL-7B** 将原始 DeepFashion 的短语标注重写为包含面料、版型、细节的自然语言长句。
  - **背景增强**: 使用 **BiRefNet** 抠图配合 **LCM-LoRA** 进行实景背景合成，提升模型在复杂环境下的鲁棒性。
- **工业级训练稳定性**: 针对 Transformer 解码中的 `NaN` 问题引入了 **VC-BDR Loss (Visually-Constrained Batch Diversity Regularization)** 和 **Logits Clamping**。
- **一体化 GUI 展示**: 提供基于 Gradio 的交互式前端，支持双模型实时推理对比及 Attention Map 热力图可视化。

---

## 🏗️ 项目架构

项目由两个核心子工程组成：

### 1. Advanced Architecture: DINOv2-T5
位于目录 `dinov2_t5/`，代表了当前 LLM 时代的视觉对齐方案。
- **Encoder**: DINOv2-Base @ 518px（无监督预训练视觉骨干）。
- **Decoder**: Flan-T5-Base（具有强大的自然语言指令遵循能力）。
- **Connector**: 采用 **SwiGLU** 门控线性单元进行模态对齐。
- **Adaptation**: **LoRA (Rank 32)** 作用于视觉与文本两端，极大地减少了训练参数量。

### 2. Classic Architecture: ResNet-LSTM-Attention
位于目录 `resnet_lstm_attention/`，追求极速推理与稳定性。
- **Encoder**: ResNet-101 特征提取。
- **Attention**: 自适应空间注意力机制，每一步生成均能定位到图像特定部位。
- **Decoder**: 双层 LSTM 循环神经网路。

---

## 📂 目录导航

```bash
NNDL_CD/
├── dinov2_t5/                       # DINOv2-T5 进阶项目
│   ├── Dino-518-Captioning/src/    # 核心源代码 (Modeling, Train, Inference)
│   ├── weights/                    # 存储 LoRA 适配器权重
│   └── models/                     # 本地预训练模型库
├── resnet_lstm_attention/           # ResNet-Attention 经典项目
│   ├── app.py                      # ResNet 版独立 Demo
│   ├── models.py                   # 模型结构定义
│   └── train_aug.py                # 增强版训练脚本
├── requirements.txt                 # 项目统一依赖环境
└── README.md                        # 主项目文档
```

---

## 🛠️ 快速开始

### 1. 环境安装
推荐使用 Python 3.10 和 CUDA 12.1。
```bash
# 克隆仓库
git clone https://github.com/your-username/DeepFashion-Captioning-Plus.git
cd DeepFashion-Captioning-Plus

# 安装依赖
pip install -r requirements.txt
```

### 2. 推理演示
我们提供了一个统一的 Gradio 前端，可以同时调用两个子项目：
```bash
# 请确保权重文件已按目录要求放置
python dinov2_t5/Dino-518-Captioning/src/frontend.py
```

### 3. 数据处理与数据增强
若需执行全自动数据流水线，可参考子目录文档：
- **抠图与背景合成**: 运行 `dinov2_t5/Dino-518-Captioning/src/augment_data_background.py`
- **Qwen2-VL 语义改写**: 运行 `resnet_lstm_attention/json_aug.py`

---

## 📈 性能表现与评估

我们针对两个模型在增强后的数据上进行了全面评估。注意：分数的变化反映了模型从“模版背诵”向“自然语言理解”的演进。

### 1. 定量指标对比 (On Test Set)

| 模型架构 | 数据版本 | BLEU-4 | METEOR | ROUGE-L | 主要特征 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-101 + LSTM** | Baseline (V1) | 0.1299 | 0.3390 | 0.2159 | 倾向于背诵模版，存在性别误判 |
| **ResNet-101 + LSTM** | **增强版 (V3)** | 0.0981 | 0.2909 | **0.2978** | **语义覆盖更高**，识别格纹/领口等细节 |
| **DINOv2 + Flan-T5** | **增强版 (SOTA)** | **0.3807** | **0.6588** | **0.6309** | **质的飞跃**，叙事性强，逻辑严密 |

> **指标深度解析**：
> - **BLEU-4 回落现象**：在 ResNet 模型中，V3 版的分数低于 V1，这是因为 V3 抛弃了死板的模版句（如 "The person wears..."），转向更多样化的表达（如 "A man models / has on..."）。虽然字面匹配度下降，但 **ROUGE-L (语义召回率)** 提升了约 8%，说明实际捕捉到的服饰信息点更密集。
> - **DINOv2-T5 的领先性**：凭借 518px 高分辨率特征与 T5 的语言理解力，进阶模型在所有指标上实现了相对于基线的倍数级超越。

---

## 💡 技术亮点

### 1. 视文双重增强 (Visual-Textual Dual Augmentation)
- **视觉端**: 针对 DeepFashion 原图多为白底的缺陷，利用 **BiRefNet** 提取主体 + **SD 1.5 & LCM-LoRA** 进行实景重绘（街拍、现代家居等），仅需 8 步推理。
- **文本端**: 利用 **Qwen2-VL-7B** 对 40,000+ 标注进行重写，强制分配 80% 权重于服饰细节（材质、剪裁），20% 权重于环境描述。

### 2. VC-BDR Loss (防止后验崩塌)
在微调 T5 时，模型容易忽略图像而产生“万能模版句”。我们设计了 **基于视觉约束的批次多样性正则化**：
$$L_{VC-BDR} = \max(0, \text{Margin} - (1 - \text{Sim}_{visual}) \times \text{Sim}_{text})$$
该 Loss 惩罚那些“视觉特征迥异但文本输出雷同”的样本对，强迫模型关注像素级的差异。

### 3. 稳健的训练策略
- **Blind Test (盲人测试)**: 每个 Epoch 后输入全黑图片，确保模型不会在无视觉输入的情况下凭空臆造。
- **Logits Clamping**: 在 Attention 层引入 $[-10, 10]$ 限制，彻底解决高分辨率特征导致的 Softmax 溢出 (`NaN`) 问题。

---

## 📜 许可说明
本项目遵循 MIT 许可证。

## 🤝 贡献
欢迎提交 Issue 或 Pull Request 来优化模型在特定服装面料或配饰上的表现！
