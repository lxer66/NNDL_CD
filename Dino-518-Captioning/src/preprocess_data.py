"""
数据预处理脚本 - DINOv2-T5 数据集清洗与划分
功能:
1. 验证图片存在性
2. 将图片resize到518×518并替换原图
3. 过滤文本过短的样本 (< 5 words)
4. 划分训练集/验证集/测试集 (80%/10%/10%)
5. 保存为 jsonl 格式
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# 使用全局配置的路径，确保与服务器目录一致
from config import CAPTIONS_PATH, IMAGES_DIR, DATA_DIR, IMAGE_SIZE


class DataPreprocessor:
    """DINOv2-T5 数据预处理器"""
    
    def __init__(
        self,
        captions_path: str,
        images_dir: str,
        output_dir: str,
        image_size: int = 518,
        min_words: int = 5,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            captions_path: captions.json 文件路径
            images_dir: 图片文件夹路径
            output_dir: 输出目录 (保存 train/val/test.jsonl)
            image_size: resize后的图片尺寸
            min_words: 最小单词数阈值
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
        """
        self.captions_path = Path(captions_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.min_words = min_words
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = defaultdict(int)
    
    def load_captions(self) -> Dict[str, str]:
        """加载 captions.json"""
        print(f"📖 Loading captions from: {self.captions_path}")
        with open(self.captions_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        print(f"✅ Loaded {len(captions)} image-caption pairs")
        return captions
    
    def resize_image(self, image_path: Path) -> bool:
        """
        将图片resize到指定尺寸并替换原图
        
        Args:
            image_path: 图片路径
            
        Returns:
            bool: 是否成功
        """
        try:
            img = Image.open(image_path).convert('RGB')
            # 直接resize到518×518,不保持长宽比
            img_resized = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            # 替换原图
            img_resized.save(image_path, quality=100)
            return True
        except Exception as e:
            print(f"  ⚠️ Error resizing {image_path}: {e}")
            return False
    
    def validate_sample(self, image_name: str, caption: str) -> Tuple[bool, str]:
        """
        验证单个样本
        
        Returns:
            (is_valid, reason): 是否有效及原因
        """
        # 1. 检查图片是否存在
        image_path = self.images_dir / image_name
        if not image_path.exists():
            return False, "image_not_found"
        
        # 2. 检查文本单词数
        words = caption.strip().split()
        if len(words) < self.min_words:
            return False, "text_too_short"
        
        return True, "valid"
    
    def clean_and_resize_data(self, captions: Dict[str, str]) -> List[Dict]:
        """
        数据清洗 + Resize图片
        
        Returns:
            List of {"image": image_name, "caption": caption}
        """
        print(f"\n🧹 Cleaning data (min_words={self.min_words})...")
        print(f"🖼️  Resizing images to {self.image_size}×{self.image_size}...")
        
        valid_samples = []
        
        for image_name, caption in tqdm(captions.items(), desc="Processing"):
            is_valid, reason = self.validate_sample(image_name, caption)
            
            if is_valid:
                # Resize图片
                image_path = self.images_dir / image_name
                if self.resize_image(image_path):
                    valid_samples.append({
                        "image": image_name,
                        "caption": caption.strip()
                    })
                    self.stats['valid'] += 1
                else:
                    self.stats['resize_failed'] += 1
            else:
                self.stats[reason] += 1
        
        # 打印统计信息
        print("\n📊 Cleaning Statistics:")
        print(f"  ✅ Valid samples: {self.stats['valid']}")
        print(f"  ❌ Image not found: {self.stats['image_not_found']}")
        print(f"  ❌ Text too short (< {self.min_words} words): {self.stats['text_too_short']}")
        print(f"  ❌ Resize failed: {self.stats['resize_failed']}")
        print(f"  📈 Valid rate: {self.stats['valid'] / len(captions) * 100:.2f}%")
        
        return valid_samples
    
    def split_data(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        划分训练集/验证集/测试集
        
        Returns:
            (train_samples, val_samples, test_samples)
        """
        print(f"\n✂️ Splitting data (train/val/test = {self.train_ratio}/{self.val_ratio}/{self.test_ratio})...")
        
        # 设置随机种子
        random.seed(self.seed)
        random.shuffle(samples)
        
        # 计算划分点
        total = len(samples)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        print(f"  📦 Train: {len(train_samples)} samples ({len(train_samples)/total*100:.1f}%)")
        print(f"  📦 Val:   {len(val_samples)} samples ({len(val_samples)/total*100:.1f}%)")
        print(f"  📦 Test:  {len(test_samples)} samples ({len(test_samples)/total*100:.1f}%)")
        
        return train_samples, val_samples, test_samples
    
    def save_jsonl(self, samples: List[Dict], split: str):
        """
        保存为 jsonl 格式 (每行一个 JSON 对象)
        
        Args:
            samples: 样本列表
            split: "train" / "val" / "test"
        """
        output_path = self.output_dir / f"{split}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  💾 Saved {split} set to: {output_path}")
    
    def run(self):
        """执行完整的预处理流程"""
        print("=" * 60)
        print("🚀 Starting Data Preprocessing Pipeline")
        print("=" * 60)
        
        # 1. 加载数据
        captions = self.load_captions()
        
        # 2. 清洗数据 + Resize图片
        valid_samples = self.clean_and_resize_data(captions)
        
        if len(valid_samples) == 0:
            print("\n❌ No valid samples found! Please check your data.")
            return
        
        # 3. 划分数据集
        train_samples, val_samples, test_samples = self.split_data(valid_samples)
        
        # 4. 保存文件
        print("\n💾 Saving processed data...")
        self.save_jsonl(train_samples, "train")
        self.save_jsonl(val_samples, "val")
        self.save_jsonl(test_samples, "test")
        
        print("\n" + "=" * 60)
        print("✅ Preprocessing Complete!")
        print("=" * 60)
        print(f"\n📁 Output files:")
        print(f"  - {self.output_dir}/train.jsonl")
        print(f"  - {self.output_dir}/val.jsonl")
        print(f"  - {self.output_dir}/test.jsonl")
        print(f"\n⚠️ Note: Original images have been resized to {self.image_size}×{self.image_size}")


def main():
    """主函数 - 配置路径并运行预处理"""
    
    # ========== 配置区域 ==========
    # 使用配置中的新数据源 (增广后的 captions 和图片)
    CAPTIONS_FILE = CAPTIONS_PATH
    IMAGES_FOLDER = IMAGES_DIR
    OUTPUT_DIR = DATA_DIR  # 产出 train/val/test.jsonl 到 data/

    IMAGE_SIZE_CFG = IMAGE_SIZE                      # DINOv2 原生分辨率
    MIN_WORDS = 5                                    # 最小单词数
    TRAIN_RATIO = 0.8                                # 80% 训练集
    VAL_RATIO = 0.1                                  # 10% 验证集
    SEED = 42                                        # 随机种子
    # ==============================
    
    # 实例化并运行
    preprocessor = DataPreprocessor(
        captions_path=str(CAPTIONS_FILE),
        images_dir=str(IMAGES_FOLDER),
        output_dir=str(OUTPUT_DIR),
        image_size=IMAGE_SIZE_CFG,
        min_words=MIN_WORDS,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()