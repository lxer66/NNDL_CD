"""
Dataset for DINOv2-Base (518px) + Flan-T5-Base Image Captioning
处理已经resize到518×518的图像和JSONL格式的标注
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoTokenizer
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    VISION_MODEL_PATH, 
    TEXT_MODEL_PATH, 
    IMAGE_SIZE, 
    MAX_LENGTH,
    DATA_DIR,
    IMAGES_DIR,
    BATCH_SIZE,
    NUM_WORKERS
)


class CaptioningDataset(Dataset):
    """
    Image Captioning Dataset
    读取已经resize到518×518的图片和JSONL格式的标注
    """
    
    def __init__(
        self,
        split: str = "train",
        data_root: Optional[str] = None
    ):
        """
        Args:
            split: 数据集划分 ('train', 'val', 'test')
            data_root: 数据根目录路径
        """
        super().__init__()
        self.split = split
        self.data_root = Path(data_root) if data_root else DATA_DIR
        # 允许自定义 data_root，但图片目录始终指向新的增广图片子目录
        images_subdir = Path(IMAGES_DIR).name
        self.images_dir = (self.data_root / images_subdir)
        
        # 加载标注数据 (JSONL格式)
        annotations_path = self.data_root / f"{split}.jsonl"
        self.annotations = []
        with open(annotations_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.annotations.append(json.loads(line.strip()))
        
        # 初始化 DINOv2 Image Processor
        self.image_processor = AutoImageProcessor.from_pretrained(
            VISION_MODEL_PATH,
            use_fast=True  # 添加这个参数，使用快速处理器
        )
        
        # 初始化 T5 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        
        print(f"✓ Loaded {len(self.annotations)} samples for split '{split}'")
        print(f"  - Data root: {self.data_root}")
        print(f"  - Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
        print(f"  - Max Text Length: {MAX_LENGTH}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回处理后的样本
        
        Returns:
            dict: {
                'pixel_values': torch.Tensor [3, 518, 518],
                'labels': torch.Tensor [max_length]
            }
        """
        sample = self.annotations[idx]
        
        # 1. 加载图像 (已经是518×518)
        image_path = self.images_dir / sample['image']
        image = Image.open(image_path).convert('RGB')
        
        # 2. 图像处理 - 关键修正!!!
        # 即使图片已经是518×518,仍然需要显式指定size和do_center_crop
        # 否则processor可能使用默认配置
        pixel_values = self.image_processor(
            images=image,
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
            do_center_crop=False,  # 不裁剪,图片已经是目标尺寸
            return_tensors="pt"
        ).pixel_values.squeeze(0)  # [3, 518, 518]
        
        # 3. 文本处理 - 获取 Caption
        caption = sample['caption']
        
        # Tokenize Caption
        encoded = self.tokenizer(
            caption,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取 labels (input_ids)
        labels = encoded.input_ids.squeeze(0)  # [max_length]
        
        # 4. Padding Fix - 关键步骤!!!
        # 将 Padding Token ID (0 for T5) 替换为 -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,  # [3, 518, 518]
            'labels': labels                # [max_length], padding位置为-100
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """获取样本的元信息 (用于调试和可视化)"""
        sample = self.annotations[idx]
        return {
            'image_path': str(self.images_dir / sample['image']),
            'caption': sample['caption'],
            'image_id': sample.get('image_id', sample['image'])
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """自定义批次整理函数"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'pixel_values': pixel_values,  # [batch_size, 3, 518, 518]
        'labels': labels                # [batch_size, max_length]
    }


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    splits: List[str] = ['train', 'val']
) -> Dict[str, DataLoader]:
    """创建数据加载器"""
    dataloaders = {}
    
    for split in splits:
        dataset = CaptioningDataset(split=split)
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=collate_fn
        )
        
        print(f"✓ Created DataLoader for '{split}' split")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Num batches: {len(dataloaders[split])}")
    
    return dataloaders


if __name__ == "__main__":
    """测试数据集加载"""
    print("=" * 60)
    print("Testing CaptioningDataset")
    print("=" * 60)
    
    try:
        dataset = CaptioningDataset(split='train')
        sample = dataset[0]
        info = dataset.get_sample_info(0)
        
        print(f"\n{'='*60}")
        print("✓ Sample 0 loaded successfully:")
        print(f"{'='*60}")
        print(f"  - Image Path: {info['image_path']}")
        print(f"  - Caption: {info['caption']}")
        print(f"  - Pixel Values Shape: {sample['pixel_values'].shape}")
        print(f"  - Pixel Values Range: [{sample['pixel_values'].min():.3f}, {sample['pixel_values'].max():.3f}]")
        print(f"  - Labels Shape: {sample['labels'].shape}")
        print(f"  - Labels (first 20 tokens): {sample['labels'][:20].tolist()}")
        print(f"  - Num of -100 in labels: {(sample['labels'] == -100).sum().item()}")
        print(f"  - Num of valid tokens: {(sample['labels'] != -100).sum().item()}")
        
        # 测试 DataLoader
        print(f"\n{'='*60}")
        print("Testing DataLoader:")
        print(f"{'='*60}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        batch = next(iter(dataloader))
        print(f"✓ Batch loaded successfully:")
        print(f"  - Batch Pixel Values Shape: {batch['pixel_values'].shape}")
        print(f"  - Batch Labels Shape: {batch['labels'].shape}")
        print(f"  - Memory usage: ~{batch['pixel_values'].element_size() * batch['pixel_values'].nelement() / 1024**2:.2f} MB")
        
        # 验证关键配置
        print(f"\n{'='*60}")
        print("Verifying Key Configurations:")
        print(f"{'='*60}")
        assert sample['pixel_values'].shape == (3, IMAGE_SIZE, IMAGE_SIZE), \
            f"Expected shape (3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {sample['pixel_values'].shape}"
        print(f"✓ Image size verified: {IMAGE_SIZE}×{IMAGE_SIZE}")
        
        assert (sample['labels'] == -100).any(), \
            "Expected -100 for padding tokens"
        print(f"✓ Padding fix verified: {(sample['labels'] == -100).sum().item()} padding tokens")
        
        assert sample['labels'].shape[0] == MAX_LENGTH, \
            f"Expected max_length {MAX_LENGTH}, got {sample['labels'].shape[0]}"
        print(f"✓ Max length verified: {MAX_LENGTH}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure your data directory structure is:")
        print("  data/")
        print("  ├── train.jsonl")
        print("  ├── val.jsonl")
        print("  ├── test.jsonl")
        print("  └── images/")
        print("\nJSONL format: {'image': 'xxx.jpg', 'caption': '...'}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)