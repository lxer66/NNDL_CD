"""
Inference Script for DINOv2-Base + Flan-T5-Base Image Captioning
支持单张图片推理和批量推理
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
import argparse
from typing import Union, List
import json

from config import (
    VISION_MODEL_PATH,
    TEXT_MODEL_PATH,
    IMAGE_SIZE,
    NUM_BEAMS,
    REPETITION_PENALTY,
    MAX_NEW_TOKENS,
    CHECKPOINTS_DIR
)
from modeling import DinoT5LoRAModel


class ImageCaptioner:
    """图像描述生成器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            checkpoint_path: 检查点目录路径
            device: 运行设备
        """
        self.device = device
        print(f"Loading model from {checkpoint_path}...")
        print(f"Device: {self.device}")
        
        # 1. 初始化模型
        self.model = DinoT5LoRAModel(use_gradient_checkpointing=False)
        
        # 2. 加载训练好的权重
        self.model.load_trainable(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully")
        
        # 3. 加载处理器和分词器
        self.image_processor = AutoImageProcessor.from_pretrained(
            VISION_MODEL_PATH,
            use_fast=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        
        print("✓ Processors loaded")
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        预处理单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            pixel_values: [1, 3, 518, 518]
        """
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # 处理图片
        pixel_values = self.image_processor(
            images=image,
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
            do_center_crop=False,
            return_tensors="pt"
        ).pixel_values
        
        return pixel_values.to(self.device)
    
    @torch.no_grad()
    def generate_caption(
        self,
        image_path: Union[str, Path],
        num_beams: int = NUM_BEAMS,
        repetition_penalty: float = REPETITION_PENALTY,
        max_length: int = MAX_NEW_TOKENS,
        return_scores: bool = False
    ) -> Union[str, tuple]:
        """
        为单张图片生成描述
        
        Args:
            image_path: 图片路径
            num_beams: Beam search 宽度
            repetition_penalty: 重复惩罚系数
            max_length: 最大生成长度
            return_scores: 是否返回生成分数
            
        Returns:
            caption: 生成的描述文本
            或 (caption, score) 如果 return_scores=True
        """
        # 预处理
        pixel_values = self.preprocess_image(image_path)
        
        # 生成
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=return_scores,
            output_scores=return_scores
        )
        
        # 解码
        if return_scores:
            caption = self.tokenizer.decode(
                generated_ids.sequences[0],
                skip_special_tokens=True
            )
            # 计算平均分数
            scores = generated_ids.sequences_scores[0].item()
            return caption, scores
        else:
            caption = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            return caption
    
    @torch.no_grad()
    def batch_generate(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8,
        num_workers: int = 0,
        **kwargs
    ) -> List[str]:
        """
        批量生成描述
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批次大小
            
        Returns:
            captions: 描述文本列表
        """
        captions = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # 批量预处理（可选多线程加速 I/O 和 CPU 解码）
            if num_workers and num_workers > 0:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=num_workers) as ex:
                    pixel_values_list = list(ex.map(self.preprocess_image, batch_paths))
            else:
                pixel_values_list = [self.preprocess_image(path) for path in batch_paths]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            
            # 批量生成
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                **kwargs
            )
            
            # 批量解码
            batch_captions = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            captions.extend(batch_captions)
        
        return captions


def main():
    """命令行推理接口"""
    parser = argparse.ArgumentParser(
        description="Image Captioning Inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "best_model"),
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=NUM_BEAMS,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=REPETITION_PENALTY,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional)"
    )
    
    args = parser.parse_args()
    
    # 检查图片是否存在
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    # 检查检查点是否存在
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    print("=" * 80)
    print("🖼️  Image Captioning Inference")
    print("=" * 80)
    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num Beams: {args.num_beams}")
    print(f"Repetition Penalty: {args.repetition_penalty}")
    print("=" * 80 + "\n")
    
    # 初始化推理器
    captioner = ImageCaptioner(checkpoint_path=args.checkpoint)
    
    # 生成描述
    print("Generating caption...")
    caption, score = captioner.generate_caption(
        image_path=args.image,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        return_scores=True
    )
    
    # 打印结果
    print("\n" + "=" * 80)
    print("📝 Generated Caption:")
    print("=" * 80)
    print(f"{caption}")
    print(f"\nScore: {score:.4f}")
    print("=" * 80)
    
    # 保存结果
    if args.output:
        result = {
            "image": args.image,
            "caption": caption,
            "score": float(score),
            "config": {
                "num_beams": args.num_beams,
                "repetition_penalty": args.repetition_penalty,
                "max_length": args.max_length
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Result saved to {args.output}")


if __name__ == "__main__":
    main()