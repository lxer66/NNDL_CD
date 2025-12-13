#!/bin/bash

# 单张图片推理
python src/inference.py \
    --image data/images/example.jpg \
    --checkpoint output/checkpoints/best_model \
    --num_beams 5 \
    --repetition_penalty 1.2 \
    --output inference_result.json

# 或者
# cd Dino-518-Captioning
# python src/inference.py --image path/to/image.jpg