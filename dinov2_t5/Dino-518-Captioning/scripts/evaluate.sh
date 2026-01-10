#!/bin/bash

# 在测试集上评估
python src/evaluate.py \
    --split test \
    --checkpoint output/checkpoints/best_model \
    --output evaluation_results.json \
    --save_predictions

# 或者
# python src/evaluate.py --split test --save_predictions