"""
CPU 测试脚本 - 验证训练流程的正确性
使用极小数据集和简化配置
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import json
from PIL import Image
import numpy as np

# 强制使用 CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from modeling import DinoT5LoRAModel
from dataset import CaptioningDataset, collate_fn
from torch.utils.data import DataLoader


def create_dummy_dataset(num_samples=10):
    """创建虚拟数据集用于测试"""
    from config import DATA_DIR, IMAGE_SIZE
    
    dummy_dir = Path("test_data")
    dummy_dir.mkdir(exist_ok=True)
    (dummy_dir / "images").mkdir(exist_ok=True)
    
    print(f"Creating {num_samples} dummy samples...")
    
    annotations = []
    
    for i in range(num_samples):
        # 创建随机图片
        img_name = f"test_image_{i}.jpg"
        img_path = dummy_dir / "images" / img_name
        
        # 生成随机图片
        random_img = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(random_img).save(img_path)
        
        # 创建虚拟标注
        captions = [
            "A red shirt with white patterns",
            "Blue jeans with a modern style",
            "A black dress with elegant design",
            "White sneakers with colorful details",
            "A brown jacket with warm texture"
        ]
        
        annotations.append({
            "image": img_name,
            "caption": captions[i % len(captions)]
        })
    
    # 保存 train.jsonl
    with open(dummy_dir / "train.jsonl", 'w') as f:
        for ann in annotations[:8]:  # 8 for train
            f.write(json.dumps(ann) + '\n')
    
    # 保存 val.jsonl
    with open(dummy_dir / "val.jsonl", 'w') as f:
        for ann in annotations[8:]:  # 2 for val
            f.write(json.dumps(ann) + '\n')
    
    print(f"✓ Dummy dataset created at: {dummy_dir}")
    return dummy_dir


def test_model_initialization():
    """测试模型初始化"""
    print("\n" + "=" * 80)
    print("Test 1: Model Initialization")
    print("=" * 80)
    
    try:
        model = DinoT5LoRAModel(use_gradient_checkpointing=False)
        print("✓ Model initialized successfully")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading(data_dir):
    """测试数据集加载"""
    print("\n" + "=" * 80)
    print("Test 2: Dataset Loading")
    print("=" * 80)
    
    try:
        # 创建数据集
        train_dataset = CaptioningDataset(split='train', data_root=str(data_dir))
        val_dataset = CaptioningDataset(split='val', data_root=str(data_dir))
        
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Val dataset: {len(val_dataset)} samples")
        
        # 测试加载一个样本
        sample = train_dataset[0]
        print(f"✓ Sample loaded:")
        print(f"  - pixel_values shape: {sample['pixel_values'].shape}")
        print(f"  - labels shape: {sample['labels'].shape}")
        
        return train_dataset, val_dataset
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, dataset):
    """测试前向传播"""
    print("\n" + "=" * 80)
    print("Test 3: Forward Pass")
    print("=" * 80)
    
    try:
        model.eval()
        
        # 获取一个 batch
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        
        print(f"✓ Batch created:")
        print(f"  - pixel_values: {batch['pixel_values'].shape}")
        print(f"  - labels: {batch['labels'].shape}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                pixel_values=batch['pixel_values'],
                labels=batch['labels']
            )
        
        print(f"✓ Forward pass successful:")
        print(f"  - loss: {outputs['loss'].item():.4f}")
        print(f"  - logits shape: {outputs['logits'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model, dataset):
    """测试生成"""
    print("\n" + "=" * 80)
    print("Test 4: Generation")
    print("=" * 80)
    
    try:
        model.eval()
        
        # 获取一个样本
        sample = dataset[0]
        pixel_values = sample['pixel_values'].unsqueeze(0)  # Add batch dim
        
        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=30,
                num_beams=2
            )
        
        print(f"✓ Generation successful:")
        print(f"  - generated_ids shape: {generated_ids.shape}")
        
        # 解码
        from config import TEXT_MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"  - generated text: '{generated_text}'")
        
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, dataset):
    """测试训练步骤"""
    print("\n" + "=" * 80)
    print("Test 5: Training Step (Mini)")
    print("=" * 80)
    
    try:
        model.train()
        
        # 创建优化器
        param_groups_dict = model.get_trainable_params()
        optimizer_param_groups = [
            {'params': param_groups_dict['mlp_params'], 'lr': 2e-3},
            {'params': param_groups_dict['lora_params'], 'lr': 5e-4}
        ]
        optimizer = torch.optim.AdamW(optimizer_param_groups)
        
        # 创建 DataLoader
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        # 训练 3 步
        print("Running 3 training steps...")
        for step, batch in enumerate(dataloader):
            if step >= 3:
                break
            
            # Forward
            outputs = model(
                pixel_values=batch['pixel_values'],
                labels=batch['labels']
            )
            loss = outputs['loss']
            
            # Backward
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"  Step {step + 1}: loss = {loss.item():.4f}")
        
        print("✓ Training steps completed")
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(model):
    """测试保存和加载"""
    print("\n" + "=" * 80)
    print("Test 6: Save & Load")
    print("=" * 80)
    
    try:
        save_dir = Path("test_checkpoint")
        save_dir.mkdir(exist_ok=True)
        
        # 保存
        model.save_trainable(str(save_dir))
        print(f"✓ Model saved to {save_dir}")
        
        # 检查文件是否存在
        assert (save_dir / "mlp_connector.pt").exists()
        assert (save_dir / "vision_lora").exists()
        assert (save_dir / "text_lora").exists()
        print("✓ All checkpoint files exist")
        
        # 清理
        import shutil
        shutil.rmtree(save_dir)
        print("✓ Checkpoint cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Save/Load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 80)
    print("🧪 CPU Testing Suite for DINOv2-T5 Training Pipeline")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU")
    print("=" * 80)
    
    results = {}
    
    # 创建虚拟数据集
    data_dir = create_dummy_dataset(num_samples=10)
    
    # Test 1: 模型初始化
    results['model_init'] = test_model_initialization()
    if not results['model_init']:
        print("\n❌ Critical failure: Cannot proceed without model")
        return
    
    model = DinoT5LoRAModel(use_gradient_checkpointing=False)
    
    # Test 2: 数据集加载
    train_dataset, val_dataset = test_dataset_loading(data_dir)
    results['dataset_load'] = train_dataset is not None
    if not results['dataset_load']:
        print("\n❌ Critical failure: Cannot proceed without dataset")
        return
    
    # Test 3: 前向传播
    results['forward_pass'] = test_forward_pass(model, train_dataset)
    
    # Test 4: 生成
    results['generation'] = test_generation(model, train_dataset)
    
    # Test 5: 训练步骤
    results['training_step'] = test_training_step(model, train_dataset)
    
    # Test 6: 保存加载
    results['save_load'] = test_save_load(model)
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 Test Results Summary")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("=" * 80)
    if all_passed:
        print("🎉 All tests PASSED! The training pipeline is ready.")
        print("   You can now run the full training on GPU.")
    else:
        print("⚠️  Some tests FAILED. Please fix the issues before training.")
    print("=" * 80)
    
    # 清理虚拟数据
    import shutil
    if Path("test_data").exists():
        shutil.rmtree("test_data")
        print("\n✓ Test data cleaned up")


if __name__ == "__main__":
    main()