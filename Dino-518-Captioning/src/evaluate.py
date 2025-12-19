"""
Evaluation Script for Image Captioning
计算 METEOR, ROUGE-L, BLEU, CIDEr 等指标
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import argparse
import json
from tqdm import tqdm
from typing import Dict, List
import numpy as np

# 评估指标库
try:
    from nltk.translate.meteor_score import meteor_score
    from nltk import word_tokenize
    import nltk
    # 下载必要的数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    print("⚠️ Warning: NLTK not available, METEOR will be skipped")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️ Warning: rouge-score not available, ROUGE will be skipped")

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice  # <--- 添加 SPICE 导入
    COCO_METRICS_AVAILABLE = True
except ImportError:
    COCO_METRICS_AVAILABLE = False
    print("⚠️ Warning: pycocoevalcap not available, BLEU/CIDEr/SPICE will be skipped")

from config import (
    DATA_DIR,
    CHECKPOINTS_DIR,
    NUM_BEAMS,
    REPETITION_PENALTY,
    MAX_NEW_TOKENS
)
from dataset import CaptioningDataset
from inference import ImageCaptioner


class CaptioningEvaluator:
    """图像描述评估器"""
    
    def __init__(self, checkpoint_path: str):
        """
        Args:
            checkpoint_path: 检查点目录路径
        """
        self.captioner = ImageCaptioner(checkpoint_path=checkpoint_path)
        
        # 初始化评估器
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rougeL'],
                use_stemmer=True
            )
    
    def compute_meteor(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """计算 METEOR 分数"""
        if not METEOR_AVAILABLE:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            # METEOR 需要分词
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)
        
        return np.mean(scores)
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """计算 ROUGE-L 分数"""
        if not ROUGE_AVAILABLE:
            return {"rougeL": 0.0}
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            scores.append(score['rougeL'].fmeasure)
        
        return {"rougeL": np.mean(scores)}
    
    def compute_coco_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """计算 BLEU, CIDEr 和 SPICE 分数"""
        if not COCO_METRICS_AVAILABLE:
            return {"BLEU-4": 0.0, "CIDEr": 0.0, "SPICE": 0.0}
        
        # 转换为 COCO 格式
        # gts: {image_id: [ref1, ref2, ...]}
        # res: {image_id: [pred]}
        # 注意：SPICE 需要 image_id 是整数或字符串，这里使用索引作为 ID
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        
        metrics = {}

        # 1. 计算 BLEU
        try:
            bleu_scorer = Bleu(4)
            bleu_scores, _ = bleu_scorer.compute_score(gts, res)
            metrics["BLEU-1"] = bleu_scores[0]
            metrics["BLEU-2"] = bleu_scores[1]
            metrics["BLEU-3"] = bleu_scores[2]
            metrics["BLEU-4"] = bleu_scores[3]
        except Exception as e:
            print(f"⚠️ BLEU calculation failed: {e}")

        # 2. 计算 CIDEr
        try:
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            metrics["CIDEr"] = cider_score
        except Exception as e:
            print(f"⚠️ CIDEr calculation failed: {e}")

        # 3. 计算 SPICE (新增)
        try:
            spice_scorer = Spice()
            spice_score, _ = spice_scorer.compute_score(gts, res)
            metrics["SPICE"] = spice_score
        except Exception as e:
            print(f"⚠️ SPICE calculation failed (Java required): {e}")
            metrics["SPICE"] = 0.0
        
        return metrics
    
    def evaluate_dataset(
        self,
        split: str = "test",
        num_samples: int = None,
        batch_size: int = 32,
        num_workers: int = 16
    ) -> Dict[str, float]:
        """
        在数据集上评估
        
        Args:
            split: 数据集划分 ('train', 'val', 'test')
            num_samples: 评估样本数量 (None表示全部)
            batch_size: 批次大小
            
        Returns:
            metrics: 评估指标字典
        """
        print(f"\n{'='*80}")
        print(f"Evaluating on {split} set")
        print(f"{'='*80}\n")
        
        # 加载数据集
        dataset = CaptioningDataset(split=split)
        
        if num_samples:
            dataset.annotations = dataset.annotations[:num_samples]
        
        print(f"Total samples: {len(dataset)}")
        
        # 收集预测和参考（批量推理）
        predictions = []
        references = []
        
        print("\nGenerating captions in batches...")
        for start in tqdm(range(0, len(dataset), batch_size)):
            end = min(start + batch_size, len(dataset))
            batch_infos = [dataset.get_sample_info(i) for i in range(start, end)]
            batch_paths = [info['image_path'] for info in batch_infos]
            batch_refs = [info['caption'] for info in batch_infos]

            batch_preds = self.captioner.batch_generate(
                image_paths=batch_paths,
                batch_size=len(batch_paths),
                num_workers=num_workers,
                num_beams=NUM_BEAMS,
                repetition_penalty=REPETITION_PENALTY,
                max_length=MAX_NEW_TOKENS
            )

            predictions.extend(batch_preds)
            references.extend(batch_refs)
        
        # 计算指标
        print("\nComputing metrics...")
        metrics = {}
        
        # METEOR
        if METEOR_AVAILABLE:
            meteor = self.compute_meteor(predictions, references)
            metrics['METEOR'] = meteor
            print(f"  METEOR: {meteor:.4f}")
        
        # ROUGE
        if ROUGE_AVAILABLE:
            rouge_scores = self.compute_rouge(predictions, references)
            metrics.update(rouge_scores)
            print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        # BLEU & CIDEr & SPICE
        if COCO_METRICS_AVAILABLE:
            coco_metrics = self.compute_coco_metrics(predictions, references)
            metrics.update(coco_metrics)
            print(f"  BLEU-4: {coco_metrics.get('BLEU-4', 0.0):.4f}")
            print(f"  CIDEr: {coco_metrics.get('CIDEr', 0.0):.4f}")
            print(f"  SPICE: {coco_metrics.get('SPICE', 0.0):.4f}")  # <--- 打印 SPICE
        
        print(f"\n{'='*80}")
        print("Evaluation Complete")
        print(f"{'='*80}\n")
        
        return metrics, predictions, references


def main():
    """命令行评估接口"""
    parser = argparse.ArgumentParser(
        description="Image Captioning Evaluation"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "best_model"),
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of worker threads for image preprocessing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 Image Captioning Evaluation")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num Samples: {args.num_samples or 'All'}")
    print("=" * 80)
    
    # 创建评估器
    evaluator = CaptioningEvaluator(checkpoint_path=args.checkpoint)
    
    # 运行评估
    metrics, predictions, references = evaluator.evaluate_dataset(
        split=args.split,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 保存结果
    if args.output:
        results = {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "num_samples": len(predictions),
            "metrics": metrics
        }
        
        if args.save_predictions:
            results["predictions"] = predictions
            results["references"] = references
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()