#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# RTX 5090 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "DINOv2-Base (518px) + T5-Small + MLP"
echo "Batch Size: 128 (Ultra Fast)"
echo "GPU: RTX 5090"
echo "========================================"

cd /media/lxer/D/STUDY/course/5/NNDL_CD/dinov2_t5/Dino-518-Captioning

# 使用 accelerate 启动
accelerate launch --mixed_precision=bf16 src/train.py

# 使用方法
# 1. 给脚本执行权限
#chmod +x /media/lxer/D/STUDY/course/5/NNDL_CD/dinov2_t5/scripts/train.sh

# 2. 启动训练
#cd /media/lxer/D/STUDY/course/5/NNDL_CD/dinov2_t5
#./scripts/train.sh

# 或者直接运行
# cd /media/lxer/D/STUDY/course/5/NNDL_CD/dinov2_t5/Dino-518-Captioning
# python src/train.py