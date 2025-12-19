"""
Global Configuration for DINOv2-Base (518px) + Flan-T5-Base Captioning
Hardware: RTX 5090 (32GB VRAM)
"""

import os
from pathlib import Path

# ============================================================
# 1. 路径配置 (Path Configuration)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # dinov2_t5/
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 数据源
CAPTIONS_PATH = DATA_DIR / "captions_aug.json"
IMAGES_DIR = DATA_DIR / "aug_images"

# 预训练模型路径
VISION_MODEL_PATH = str(MODELS_DIR / "dinov2-base")
TEXT_MODEL_PATH = str(MODELS_DIR / "flan-t5-base")  # 切换到 Flan-T5-Base

# 输出路径
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

# 确保输出目录存在
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 图像处理配置 (Image Processing)
# ============================================================
# DINOv2 原生黄金分辨率: Patch Size = 14, 518 = 14 × 37
IMAGE_SIZE = 518

# ============================================================
# 3. 训练超参数 (Training Hyperparameters)
# ============================================================
# 批次大小 - RTX 5090 32GB 显存优化（关闭梯度累积，直接物理批次=32）
BATCH_SIZE = 32              # Physical batch size
GRAD_ACCUM_STEPS = 1         # No gradient accumulation; effective batch = 32

# 数据加载
NUM_WORKERS = 16

# 学习率配置 (Differential Learning Rates)
LR_MLP = 1e-3           # MLP Connector 学习率 (高, 随机初始化)
LR_LORA = 2e-4          # LoRA 学习率 (低, 预训练权重)

# 训练轮数
EPOCHS = 15

# 混合精度训练 (RTX 5090 标配)
BF16 = True

# 文本序列最大长度
MAX_LENGTH = 200

# ============================================================
# 4. 模型架构配置 (Model Architecture)
# ============================================================
# LoRA 配置
LORA_R = 32                                    # LoRA rank
LORA_ALPHA = 64                                # LoRA alpha
LORA_DROPOUT = 0.1                             # LoRA dropout

# Vision Encoder (DINOv2) LoRA targets
VISION_LORA_TARGETS = ["query", "key", "value", "dense"]

# Language Model (T5) LoRA targets  
TEXT_LORA_TARGETS = ["q", "v"]

# MLP Connector 配置
DINO_HIDDEN_SIZE = 768    # DINOv2-Base 输出维度
T5_HIDDEN_SIZE = 768      # Flan-T5-Base 输入维度 (base 模型为 768)
MLP_HIDDEN_SIZE = 2048    # SwiGLU MLP 中间层维度 (2.67x embedding dim)

# VC-BDR Loss 配置 (Visually-Constrained Batch Diversity Regularization)
VCBDR_START_EPOCH = 2     # 从 Epoch 3 (index=2) 开始启用辅助损失
VCBDR_WEIGHT = 5        # VC-BDR 损失权重 (lambda)

# ============================================================
# 5. 生成配置 (Generation Config)
# ============================================================
NUM_BEAMS = 5
REPETITION_PENALTY = 1.2
MAX_NEW_TOKENS = MAX_LENGTH

# ============================================================
# 6. 日志与检查点配置 (Logging & Checkpointing)
# ============================================================
LOG_INTERVAL = 10          # 每10个step记录一次loss(大批次训练)
SAVE_INTERVAL = 1          # 每个epoch保存一次检查点
EVAL_INTERVAL = 1          # 每个epoch评估一次

# TensorBoard 日志目录
TENSORBOARD_DIR = str(LOGS_DIR / "tensorboard")

# ============================================================
# 7. 环境配置 (Environment)
# ============================================================
# CUDA 显存优化 (RTX 5090)
# 修正：使用新的环境变量名
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  # 改用新名称

# 禁用tokenizers的并行警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 设备配置
DEVICE = "cuda"

# ============================================================
# 8. 随机种子 (Random Seed)
# ============================================================
SEED = 42

# ============================================================
# 配置验证
# ============================================================
def validate_config():
    """验证配置有效性"""
    assert IMAGE_SIZE % 14 == 0, f"IMAGE_SIZE must be divisible by 14 (DINOv2 patch size), got {IMAGE_SIZE}"
    assert IMAGE_SIZE == 518, f"Expected IMAGE_SIZE=518 for optimal DINOv2 performance, got {IMAGE_SIZE}"
    assert os.path.exists(VISION_MODEL_PATH), f"Vision model not found at {VISION_MODEL_PATH}"
    assert os.path.exists(TEXT_MODEL_PATH), f"Text model not found at {TEXT_MODEL_PATH}"
    assert CAPTIONS_PATH.exists(), f"Captions file not found at {CAPTIONS_PATH}"
    assert IMAGES_DIR.exists(), f"Images directory not found at {IMAGES_DIR}"
    print("✓ Configuration validated successfully")
    print(f"  - Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - BF16: {BF16}")

if __name__ == "__main__":
    validate_config()