"""
Training Script for DINOv2-Base + Flan-T5-Base Image Captioning
实现 Adaptive 3-Stage Training Strategy + VC-BDR Loss
Hardware: RTX 5090 (32GB VRAM) + BF16 + Gradient Accumulation
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
import matplotlib.pyplot as plt
import random

# 设置 matplotlib 后端为 Agg，适用于无显示器的服务器环境
plt.switch_backend('agg')

from config import (
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    NUM_WORKERS,
    LR_MLP,
    LR_LORA,
    EPOCHS,
    BF16,
    CHECKPOINTS_DIR,
    TENSORBOARD_DIR,
    LOG_INTERVAL,
    SEED,
    IMAGE_SIZE,
    TEXT_MODEL_PATH,
    DATA_DIR,
    VCBDR_START_EPOCH,
    VCBDR_WEIGHT
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
    ax2.set_title('Differential Multi-Stage LR Schedule')
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
    """主训练函数 - 实现 Adaptive 3-Stage Training Strategy"""
    
    # ============================================================
    # 1. 初始化 Accelerator (with Gradient Accumulation)
    # ============================================================
    accelerator = Accelerator(
        mixed_precision="bf16" if BF16 else "no",
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,  # 关键: 梯度累积
        log_with="tensorboard",
        project_dir=TENSORBOARD_DIR
    )
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("🚀 DINOv2-Base + Flan-T5-Base Image Captioning Training")
        print("   Adaptive 3-Stage Strategy + VC-BDR Loss")
        print("=" * 80)
        print(f"  Device: {accelerator.device}")
        print(f"  Mixed Precision: {accelerator.mixed_precision}")
        print(f"  Physical Batch Size: {BATCH_SIZE}")
        print(f"  Gradient Accumulation Steps: {GRAD_ACCUM_STEPS}")
        print(f"  Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
        print(f"  Epochs: {EPOCHS}")
        print(f"  VC-BDR Start Epoch: {VCBDR_START_EPOCH + 1} (Index {VCBDR_START_EPOCH})")
        print("=" * 80 + "\n")
    
    set_seed(SEED)
    
    # ============================================================
    # 2. 加载模型 (启用梯度检查点)
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
    # 4. 配置优化器 (差异化学习率)
    # ============================================================
    if accelerator.is_main_process:
        print("\nConfiguring optimizer with differential learning rates...")

    param_groups_dict = model.get_trainable_params()

    optimizer_param_groups = [
        {
            'params': param_groups_dict['mlp_params'],
            'lr': LR_MLP,  # 1e-3 (High initial LR)
            'weight_decay': 0.01,
            'name': 'mlp_group'
        },
        {
            'params': param_groups_dict['lora_params'],
            'lr': LR_LORA,  # 2e-4 (Conservative LR)
            'weight_decay': 0.05,
            'name': 'lora_group'
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_param_groups)

    if accelerator.is_main_process:
        print(f"  ✓ MLP LR: {LR_MLP}")
        print(f"  ✓ LoRA LR: {LR_LORA}")
    
    # ============================================================
    # 5. 学习率调度器 - 差异化多阶段策略 (CRITICAL)
    # ============================================================
    num_training_steps = len(train_loader) * EPOCHS
    steps_per_epoch = len(train_loader)
    
    # Epoch 0 Warmup steps (10% of Epoch 0)
    epoch0_warmup_steps = int(0.1 * steps_per_epoch)
    
    # Epoch 1+ Warmup steps (前几步, 约5% of remaining steps)
    stage2_warmup_steps = int(0.05 * steps_per_epoch)
    
    def get_mlp_lr_lambda(current_step):
        """
        MLP Learning Rate Schedule (差异化策略)
        
        Stage 1 (Epoch 0, steps 0 - steps_per_epoch):
          - Warmup (前10%步): Linear 0.0 -> 1.0
          - Then Constant: 1.0x (实际 1e-3)
        
        Stage 2 (Epoch 1-14, steps >= steps_per_epoch):
          - HARD DROP: 立即降至 0.1x (实际 1e-4)
          - Cosine Decay: 0.1x -> 0.0
        
        目的: 防止 MLP 在 T5 解冻后破坏特征
        """
        current_epoch = current_step // steps_per_epoch
        
        # Stage 1: Epoch 0
        if current_epoch == 0:
            # Warmup (前10%步)
            if current_step < epoch0_warmup_steps:
                return current_step / epoch0_warmup_steps
            # Constant at 1.0x
            else:
                return 1.0
        
        # Stage 2: Epoch 1-14
        else:
            # 计算从 Epoch 1 开始的步数
            adjusted_step = current_step - steps_per_epoch
            adjusted_total_steps = num_training_steps - steps_per_epoch
            
            # Cosine Decay from 0.1x to 0.0
            progress = adjusted_step / adjusted_total_steps
            cosine_scale = 0.5 * (1 + math.cos(math.pi * progress))
            
            # HARD DROP: 从 1.0x 降至 0.1x, 然后余弦衰减
            return 0.1 * cosine_scale
    
    def get_lora_lr_lambda(current_step):
        """
        LoRA Learning Rate Schedule (差异化策略)
        
        Stage 1 (Epoch 0):
          - Frozen: 0.0x (实际 0.0)
        
        Stage 2 (Epoch 1-14):
          - Linear Warmup (前5%步): 0.0 -> 1.0x (实际 2e-4)
          - Cosine Decay: 1.0x -> 0.0
        """
        current_epoch = current_step // steps_per_epoch
        
        # Stage 1: Epoch 0 - Frozen
        if current_epoch == 0:
            return 0.0
        
        # Stage 2: Epoch 1-14
        else:
            # 计算从 Epoch 1 开始的步数
            adjusted_step = current_step - steps_per_epoch
            adjusted_total_steps = num_training_steps - steps_per_epoch
            
            # Warmup (前5%步)
            if adjusted_step < stage2_warmup_steps:
                return adjusted_step / stage2_warmup_steps
            
            # Cosine Decay from 1.0x to 0.0
            else:
                progress = (adjusted_step - stage2_warmup_steps) / (adjusted_total_steps - stage2_warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
    
    from torch.optim.lr_scheduler import LambdaLR
    
    # 为每个参数组创建独立的调度器
    schedulers = [
        LambdaLR(optimizer, lr_lambda=get_mlp_lr_lambda),   # 索引 0: MLP
        LambdaLR(optimizer, lr_lambda=get_lora_lr_lambda)   # 索引 1: LoRA
    ]
    
    # 合并为单个调度器 (Accelerate 兼容)
    class CombinedScheduler:
        def __init__(self, schedulers):
            self.schedulers = schedulers
        
        def step(self):
            for scheduler in self.schedulers:
                scheduler.step()
        
        def get_last_lr(self):
            return [scheduler.get_last_lr()[0] for scheduler in self.schedulers]
    
    scheduler = CombinedScheduler(schedulers)
    
    if accelerator.is_main_process:
        print("\n✓ Differential Multi-Stage LR Scheduler configured:")
        print("  MLP Strategy:")
        print(f"    - Epoch 0: Warmup (10% steps) -> Constant 1.0x (1e-3)")
        print(f"    - Epoch 1+: HARD DROP to 0.1x (1e-4) -> Cosine Decay to 0.0")
        print("  LoRA Strategy:")
        print(f"    - Epoch 0: Frozen (0.0x)")
        print(f"    - Epoch 1+: Warmup (5% steps) -> Cosine Decay from 1.0x (2e-4) to 0.0")
        print(f"  Warmup steps: Epoch0={epoch0_warmup_steps}, Stage2={stage2_warmup_steps}")
    
    # ============================================================
    # 6. Accelerate Preparation
    # ============================================================
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # ============================================================
    # 7. 初始化 TensorBoard
    # ============================================================
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    
    # ============================================================
    # 8. 加载 Tokenizer (用于 Blind Test)
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
    
    # ============================================================
    # 9. 训练循环
    # ============================================================
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80 + "\n")
    
    global_step = 0
    best_val_loss = float('inf')
    
    # 历史记录
    train_history = []
    val_history = []
    lr_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0
        
        # 判断是否启用 VC-BDR
        use_vcbdr = (epoch >= VCBDR_START_EPOCH)
        
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            
            # 打印当前阶段信息
            if epoch == 0:
                print("  Stage 1: MLP Warm-up (LoRA Frozen, No VC-BDR)")
            elif epoch < VCBDR_START_EPOCH:
                print(f"  Stage 2: Joint Training (No VC-BDR)")
            else:
                print(f"  Stage 3: VC-BDR Enabled (λ={VCBDR_WEIGHT})")
            
            print(f"{'='*80}\n")
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                pixel_values = batch['pixel_values']
                labels = batch['labels']
                
                # 前向传播 (传递 use_vcbdr 标志)
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                    use_vcbdr=use_vcbdr
                )
                
                loss = outputs['loss']
                main_loss = outputs['main_loss']
                aux_loss = outputs['aux_loss']
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 优化器步骤
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 累积损失
            epoch_train_loss += loss.item()
            epoch_main_loss += main_loss.item()
            epoch_aux_loss += aux_loss.item() if use_vcbdr else 0.0
            
            # 记录到 TensorBoard
            if accelerator.is_main_process and global_step % LOG_INTERVAL == 0:
                writer.add_scalar('Train/Total_Loss', loss.item(), global_step)
                writer.add_scalar('Train/Main_Loss', main_loss.item(), global_step)
                if use_vcbdr:
                    writer.add_scalar('Train/Aux_Loss', aux_loss.item(), global_step)
                
                # 记录学习率 (分别记录 MLP 和 LoRA)
                current_lrs = scheduler.get_last_lr()
                mlp_lr = current_lrs[0]  # MLP group
                lora_lr = current_lrs[1]  # LoRA group
                
                writer.add_scalar('LR/MLP', mlp_lr, global_step)
                writer.add_scalar('LR/LoRA', lora_lr, global_step)
                
                # 保存历史
                train_history.append((global_step, loss.item()))
                lr_history.append((global_step, mlp_lr, lora_lr))
            
            # 更新进度条 (显示当前学习率)
            if accelerator.is_main_process:
                current_lrs = scheduler.get_last_lr()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'main': f'{main_loss.item():.4f}',
                    'aux': f'{aux_loss.item():.4f}' if use_vcbdr else 'N/A',
                    'mlp_lr': f'{current_lrs[0]:.2e}',
                    'lora_lr': f'{current_lrs[1]:.2e}'
                })
            
            global_step += 1
        
        # 计算平均损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_main_loss = epoch_main_loss / len(train_loader)
        avg_aux_loss = epoch_aux_loss / len(train_loader) if use_vcbdr else 0.0
        
        # ============================================================
        # 验证
        # ============================================================
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print("Running Validation...")
            print(f"{'='*60}")
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process):
                pixel_values = batch['pixel_values']
                labels = batch['labels']
                
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                    use_vcbdr=False  # 验证时不使用 VC-BDR
                )
                
                val_loss += outputs['loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if accelerator.is_main_process:
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            val_history.append((global_step, avg_val_loss))
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} Summary:")
            print(f"{'='*60}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Main Loss:  {avg_main_loss:.4f}")
            if use_vcbdr:
                print(f"  Aux Loss:   {avg_aux_loss:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}")
            print(f"{'='*60}\n")
        
        # ============================================================
        # 保存检查点
        # ============================================================
        if accelerator.is_main_process:
            # 保存当前 epoch
            save_dir = CHECKPOINTS_DIR / f"epoch_{epoch + 1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_trainable(str(save_dir))
            
            print(f"✓ Checkpoint saved to {save_dir}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_save_dir = CHECKPOINTS_DIR / "best_model"
                best_save_dir.mkdir(parents=True, exist_ok=True)
                unwrapped_model.save_trainable(str(best_save_dir))
                print(f"✓ Best model updated (Val Loss: {best_val_loss:.4f})")
        
        # ============================================================
        # Blind Test (每个 epoch 后)
        # ============================================================
        if accelerator.is_main_process:
            perform_blind_test(accelerator.unwrap_model(model), tokenizer, accelerator)
    
    # ============================================================
    # 训练完成
    # ============================================================
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("🎉 Training Complete!")
        print("=" * 80)
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Total Steps: {global_step}")
        print("=" * 80 + "\n")
        
        # 绘制训练曲线
        print("Generating training curves...")
        plot_training_curves(train_history, val_history, lr_history, CHECKPOINTS_DIR)
        print(f"✓ Training curves saved to {CHECKPOINTS_DIR / 'training_curves.png'}")
        
        writer.close()


if __name__ == "__main__":
    train()