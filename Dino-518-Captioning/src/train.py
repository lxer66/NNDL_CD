"""
Training Script for DINOv2-Base + T5-Small Image Captioning
基于 PyTorch Native Loop + Accelerate
实现改进的两阶段训练策略
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt # 新增：导入绘图库

# 设置 matplotlib 后端为 Agg，适用于无显示器的服务器环境
plt.switch_backend('agg')

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    LR_LORA,
    LR_MLP_PRETRAIN, # 新增
    LR_MLP_FINETUNE, # 新增
    EPOCHS,
    BF16,
    CHECKPOINTS_DIR,
    TENSORBOARD_DIR,
    LOG_INTERVAL,
    SEED,
    IMAGE_SIZE,
    TEXT_MODEL_PATH,
    DATA_DIR
)
from modeling import DinoT5LoRAModel
from dataset import create_dataloaders


def plot_training_curves(train_history, val_history, lr_history, save_dir):
    """
    绘制并保存训练曲线
    Args:
        train_history: List[(step, loss)]
        val_history: List[(step, loss)]
        lr_history: List[(step, mlp_lr, lora_lr)]
        save_dir: 保存目录
    """
    if not train_history:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 1. 绘制 Loss 曲线
    train_steps, train_losses = zip(*train_history)
    ax1.plot(train_steps, train_losses, label='Train Loss', alpha=0.6, color='blue', linewidth=1)
    
    if val_history:
        val_steps, val_losses = zip(*val_history)
        ax1.plot(val_steps, val_losses, label='Val Loss', color='red', linewidth=2, marker='o')
        
        # 标注最低验证损失
        min_val_loss = min(val_losses)
        min_val_idx = val_losses.index(min_val_loss)
        min_val_step = val_steps[min_val_idx]
        ax1.annotate(f'Min Val: {min_val_loss:.4f}', 
                     xy=(min_val_step, min_val_loss), 
                     xytext=(min_val_step, min_val_loss + 0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()

    # 2. 绘制 Learning Rate 曲线
    lr_steps, mlp_lrs, lora_lrs = zip(*lr_history)
    ax2.plot(lr_steps, mlp_lrs, label='MLP LR', color='purple', linestyle='-')
    ax2.plot(lr_steps, lora_lrs, label='LoRA LR', color='green', linestyle='--')
    
    ax2.set_ylabel('Learning Rate')
    ax2.set_xlabel('Global Steps')
    ax2.set_yscale('log') # LR 通常跨度较大，使用对数坐标更清晰
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    plt.close()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def perform_blind_test(model, tokenizer, accelerator):
    """盲人测试 - 检测模型是否真的在使用视觉信息"""
    print("\n" + "=" * 80)
    print("🔍 Performing Blind Test (Posterior Collapse Detection)")
    print("=" * 80)
    
    model.eval()
    
    # 1. 加载一张真实图片
    data_images_dir = DATA_DIR / "images"
    real_image_files = list(data_images_dir.glob("*.jpg"))[:1]
    
    if not real_image_files:
        print("⚠️ No real images found for blind test, skipping...")
        return
    
    real_image = Image.open(real_image_files[0]).convert('RGB')
    real_image = real_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # 2. 创建全黑图片
    black_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    
    # 3. 预处理
    from transformers import AutoImageProcessor
    from config import VISION_MODEL_PATH
    
    processor = AutoImageProcessor.from_pretrained(VISION_MODEL_PATH)
    
    real_pixel_values = processor(
        images=real_image,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        do_center_crop=False,
        return_tensors="pt"
    ).pixel_values.to(accelerator.device)
    
    black_pixel_values = processor(
        images=black_image,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        do_center_crop=False,
        return_tensors="pt"
    ).pixel_values.to(accelerator.device)
    
    # 4. 生成
    with torch.no_grad():
        real_output = model.generate(
            pixel_values=real_pixel_values,
            max_length=50,
            num_beams=3
        )
        
        black_output = model.generate(
            pixel_values=black_pixel_values,
            max_length=50,
            num_beams=3
        )
    
    # 5. 解码
    real_text = tokenizer.decode(real_output[0], skip_special_tokens=True)
    black_text = tokenizer.decode(black_output[0], skip_special_tokens=True)
    
    # 6. 打印结果
    print(f"\n📸 Real Image Output:")
    print(f"   '{real_text}'")
    print(f"\n⬛ Black Image Output:")
    print(f"   '{black_text}'")
    
    # 7. 检测后验崩塌
    if real_text.strip() == black_text.strip():
        print("\n" + "🚨" * 40)
        print("⚠️  WARNING: Potential Posterior Collapse Detected (Model is Blind)!")
        print("🚨" * 40)
    else:
        print("\n✅ Blind Test Passed: Model outputs differ for real vs black images.")
    
    print("=" * 80 + "\n")
    
    model.train()


def train():
    """主训练函数"""
    
    # ============================================================
    # 1. 初始化 Accelerator
    # ============================================================
    accelerator = Accelerator(
        mixed_precision="bf16" if BF16 else "no",
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=TENSORBOARD_DIR
    )
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("🚀 DINOv2-Base (518px) + T5-Small Image Captioning Training")
        print("=" * 80)
        print(f"  Device: {accelerator.device}")
        print(f"  Mixed Precision: {accelerator.mixed_precision}")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Epochs: {EPOCHS}")
        print("=" * 80 + "\n")
    
    set_seed(SEED)
    
    # ============================================================
    # 2. 加载模型
    # ============================================================
    if accelerator.is_main_process:
        print("Loading model...")
    
    model = DinoT5LoRAModel(use_gradient_checkpointing=True)
    
    # ============================================================
    # 3. 加载数据集
    # ============================================================
    if accelerator.is_main_process:
        print("\nLoading datasets...")
    
    dataloaders = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        splits=['train', 'val']
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # ============================================================
    # 4. 配置优化器
    # ============================================================
    if accelerator.is_main_process:
        print("\nConfiguring optimizer...")

    param_groups_dict = model.get_trainable_params()

    optimizer_param_groups = [
        {
            'params': param_groups_dict['mlp_params'],
            'lr': LR_MLP_PRETRAIN,  # 初始使用高学习率 (3e-3)
            'weight_decay': 0.01,
            'name': 'mlp_group'
        },
        {
            'params': param_groups_dict['lora_params'],
            'lr': LR_LORA,          # LoRA 学习率 (3e-4)
            'weight_decay': 0.05,
            'name': 'lora_group'
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_param_groups)

    if accelerator.is_main_process:
        print(f"  ✓ MLP Group: LR={LR_MLP_PRETRAIN}, WD=0.01")
        print(f"  ✓ LoRA Group: LR={LR_LORA}, WD=0.05")
    
    # ============================================================
    # 5. 学习率调度器 - 阶梯式策略
    # ============================================================
    num_training_steps = len(train_loader) * EPOCHS
    steps_per_epoch = len(train_loader)
    
    def get_mlp_lr_lambda(current_step):
        """
        MLP 调度策略:
        - Epoch 0 (Stage 1): Warmup -> 保持 LR_MLP_PRETRAIN (1.0倍)
        - Epoch 1+ (Stage 2): 瞬间降至 LR_MLP_FINETUNE (0.1倍) -> Cosine Decay
        """
        # 计算当前在第几个 epoch
        current_epoch = current_step // steps_per_epoch
        
        if current_epoch == 0:
            # === Stage 1: Pre-training ===
            # 简单的 Warmup，然后保持高位
            warmup_steps = int(steps_per_epoch * 0.1)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # 保持 3e-3
            
        else:
            # === Stage 2: Fine-tuning ===
            # 计算衰减因子: 目标是让基础 LR (3e-3) 变成 (3e-4)
            # 所以 factor = 3e-4 / 3e-3 = 0.1
            target_factor = LR_MLP_FINETUNE / LR_MLP_PRETRAIN
            
            # 计算 Stage 2 的进度
            stage2_total_steps = num_training_steps - steps_per_epoch
            stage2_current_step = current_step - steps_per_epoch
            
            # Cosine Decay 从 target_factor 降到 0
            progress = float(stage2_current_step) / float(max(1, stage2_total_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return target_factor * cosine_decay

    def get_lora_lr_lambda(current_step):
        """
        LoRA 调度策略:
        - Epoch 0: 冻结 (0.0)
        - Epoch 1+: Warmup -> Cosine Decay
        """
        current_epoch = current_step // steps_per_epoch
        
        if current_epoch == 0:
            # Stage 1: 完全冻结
            return 0.0
        
        # Stage 2: 正常训练
        stage2_total_steps = num_training_steps - steps_per_epoch
        stage2_current_step = current_step - steps_per_epoch
        
        warmup_steps = int(stage2_total_steps * 0.05) # 5% warmup
        
        if stage2_current_step < warmup_steps:
             return float(stage2_current_step) / float(max(1, warmup_steps))
        else:
            progress = float(stage2_current_step - warmup_steps) / float(max(1, stage2_total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, [get_mlp_lr_lambda, get_lora_lr_lambda])
    
    if accelerator.is_main_process:
        print(f"\n✓ Advanced Scheduler Configured:")
        print(f"  - MLP Stage 1 (Epoch 1): LR = {LR_MLP_PRETRAIN:.1e} (High)")
        print(f"  - MLP Stage 2 (Epoch 2+): LR drops to {LR_MLP_FINETUNE:.1e} -> 0 (Cosine)")
        print(f"  - LoRA Stage 1: Frozen")
        print(f"  - LoRA Stage 2: LR = {LR_LORA:.1e} (Cosine)")

    # ============================================================
    # 6. Accelerate 包装
    # ============================================================
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # ============================================================
    # 7. TensorBoard
    # ============================================================
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    
    # ============================================================
    # 8. 盲人测试 (训练前)
    # ============================================================
    if accelerator.is_main_process:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        perform_blind_test(
            accelerator.unwrap_model(model),
            tokenizer,
            accelerator
        )
    
    # ============================================================
    # 9. 训练循环
    # ============================================================
    global_step = 0
    best_val_loss = float('inf')
    
    # 新增：用于记录绘图数据的列表
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    
    for epoch in range(EPOCHS):
        # 确定当前阶段
        if epoch == 0:
            stage_name = "Stage 1: MLP Pre-training (LoRA Frozen)"
        else:
            stage_name = "Stage 2: Joint Fine-tuning"
        
        if accelerator.is_main_process:
            print("\n" + "=" * 80)
            print(f"Epoch {epoch + 1}/{EPOCHS} - {stage_name}")
            print("=" * 80)
        
        # ============================================================
        # 训练阶段
        # ============================================================
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            disable=not accelerator.is_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            # Forward
            outputs = model(
                pixel_values=batch['pixel_values'],
                labels=batch['labels']
            )
            loss = outputs['loss']
            
            # Backward
            accelerator.backward(loss)
            
            # 梯度裁剪
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 统计
            epoch_loss += loss.item()
            global_step += 1
            
            # 日志记录
            if global_step % LOG_INTERVAL == 0:
                avg_loss = epoch_loss / (step + 1)
                
                # 获取当前实际学习率
                current_lr_mlp = scheduler.get_last_lr()[0]
                current_lr_lora = scheduler.get_last_lr()[1]
                
                # 新增：记录数据
                train_loss_history.append((global_step, loss.item())) # 记录当前step的loss，或者用avg_loss
                lr_history.append((global_step, current_lr_mlp, current_lr_lora))

                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr_mlp': f'{current_lr_mlp:.2e}',
                    'lr_lora': f'{current_lr_lora:.2e}'
                })
                
                if accelerator.is_main_process:
                    writer.add_scalar('Train/Loss', avg_loss, global_step)
                    writer.add_scalar('Train/LR_MLP', current_lr_mlp, global_step)
                    writer.add_scalar('Train/LR_LoRA', current_lr_lora, global_step)
        
        # ============================================================
        # 验证阶段
        # ============================================================
        model.eval()
        val_loss = 0
        
        if accelerator.is_main_process:
            print("\n  Validating...")
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels']
                )
                val_loss += outputs['loss'].item()
        
        val_loss /= len(val_loader)
        
        # 新增：记录验证损失
        val_loss_history.append((global_step, val_loss))
        
        if accelerator.is_main_process:
            print(f"  ✓ Validation Loss: {val_loss:.4f}")
            writer.add_scalar('Val/Loss', val_loss, epoch)
            
            # 新增：绘制并保存曲线图
            print("  📈 Plotting training curves...")
            
            # === 补全缺失的调用代码 ===
            plot_training_curves(
                train_loss_history, 
                val_loss_history, 
                lr_history, 
                CHECKPOINTS_DIR
            )

            # 获取解包后的模型（去除 DDP/Accelerate 包装）
            unwrapped_model = accelerator.unwrap_model(model)

            # 保存最佳模型逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  ★ New Best Model! Saving to {CHECKPOINTS_DIR}/best_model")
                # 1. 保存 Accelerator 完整状态 (用于恢复训练)
                accelerator.save_state(output_dir=str(CHECKPOINTS_DIR / "best_model_state"))
                # 2. 保存推理所需的组件 (用于 inference.py)
                unwrapped_model.save_trainable(str(CHECKPOINTS_DIR / "best_model"))
            
            # 保存最新模型
            current_ckpt_dir = CHECKPOINTS_DIR / f"checkpoint-{epoch+1}-val_loss_{val_loss:.4f}"
            # 1. 保存 Accelerator 完整状态
            accelerator.save_state(output_dir=str(current_ckpt_dir / "state"))
            # 2. 保存推理所需的组件 (关键修复!!!)
            unwrapped_model.save_trainable(str(current_ckpt_dir))
    
            # 新增：每轮结束后运行盲人测试，实时监控模型状态
            print(f"\n  🔍 Running Blind Test for Epoch {epoch + 1}...")
            perform_blind_test(
                unwrapped_model,
                tokenizer,
                accelerator
            )
    
    # ============================================================
    # 10. 盲人测试 (训练后)
    # ============================================================
    if accelerator.is_main_process:
        print("\n🔍 Performing Blind Test (Posterior Collapse Detection) - After Training")
        perform_blind_test(
            accelerator.unwrap_model(model),
            tokenizer,
            accelerator
        )
    
    # ============================================================
    # 11. 结束
    # ============================================================
    if accelerator.is_main_process:
        print("=" * 80)
        print("✅ Training Complete!")
        print(f"  - Best Validation Loss: {best_val_loss:.4f}")
        print(f"  - Checkpoints saved in: {CHECKPOINTS_DIR}")
        print("=" * 80)


if __name__ == "__main__":
    train()