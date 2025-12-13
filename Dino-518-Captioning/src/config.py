"""
Global Configuration for DINOv2-Base (518px) + T5-Small Captioning
Hardware: RTX 5090 (32GB VRAM)
"""

import os
from pathlib import Path

# ============================================================
# 1. 路径配置 (Path Configuration)
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 预训练模型路径
VISION_MODEL_PATH = str(MODELS_DIR / "dinov2-base")
TEXT_MODEL_PATH = str(MODELS_DIR / "t5-small")

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
# 批次大小 - RTX 5090 32GB 显存优化,使用大批次加速训练
BATCH_SIZE = 20

# 数据加载
NUM_WORKERS = 16

# 学习率配置
LR_LORA = 3e-4          # LoRA 学习率 (Stage 2)

# MLP 学习率策略
LR_MLP_PRETRAIN = 3e-3  # Stage 1 (Epoch 0): 高学习率，快速对齐
LR_MLP_FINETUNE = 3e-4  # Stage 2 (Epoch 1+): 低学习率 (1/10)，防止破坏 LoRA

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
T5_HIDDEN_SIZE = 512      # T5-Small 输入维度
MLP_HIDDEN_SIZE = 2048    # MLP 中间层维度

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
    print("✓ Configuration validated successfully")
    print(f"  - Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - BF16: {BF16}")

if __name__ == "__main__":
    validate_config()