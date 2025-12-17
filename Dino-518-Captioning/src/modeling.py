"""
DINOv2-Base + SwiGLU MLP Connector + Flan-T5-Base with LoRA
核心模型架构,实现防盲/防泄露策略
使用 Pre-Norm SwiGLU 和 Encoder Injection 方法
"""

import torch
import torch.nn as nn
from transformers import Dinov2Model, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import (
    VISION_MODEL_PATH,
    TEXT_MODEL_PATH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    VISION_LORA_TARGETS,
    TEXT_LORA_TARGETS,
    DINO_HIDDEN_SIZE,
    T5_HIDDEN_SIZE,
    MLP_HIDDEN_SIZE,
    IMAGE_SIZE,
    VCBDR_WEIGHT
)

from loss import VC_BDR_Loss


class SwiGLUConnector(nn.Module):
    """
    SwiGLU MLP Connector with Pre-Norm
    Architecture: LayerNorm -> SwiGLU(768 -> 2048 -> 768)
    SwiGLU: output = W3(SiLU(W1(x)) * W2(x))
    """
    
    def __init__(
        self,
        input_dim: int = DINO_HIDDEN_SIZE,
        hidden_dim: int = MLP_HIDDEN_SIZE,
        output_dim: int = T5_HIDDEN_SIZE
    ):
        super().__init__()
        
        # Pre-Norm: Stabilize DINOv2 features
        self.norm = nn.LayerNorm(input_dim)
        
        # SwiGLU Components
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)   # Gate
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=False)   # Value
        self.w3 = nn.Linear(hidden_dim, output_dim, bias=False)  # Output
        
        # Activation
        self.act = nn.SiLU()  # Swish activation
        
        # Xavier Uniform Initialization
        self._init_weights()
    
    def _init_weights(self):
        """Apply Xavier Uniform initialization to all linear layers"""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, 768] - DINOv2 output
        Returns:
            [batch_size, seq_len, 768] - T5-Base input
        """
        # Pre-Norm
        x_norm = self.norm(x)
        
        # SwiGLU: W3(SiLU(W1(x)) * W2(x))
        gate = self.act(self.w1(x_norm))    # [B, L, 2048]
        value = self.w2(x_norm)              # [B, L, 2048]
        hidden = gate * value                # Element-wise product
        output = self.w3(hidden)             # [B, L, 768]
        
        return output


class DinoT5LoRAModel(nn.Module):
    """
    完整的图像描述生成模型
    架构: DINOv2 (LoRA) -> SwiGLU Connector -> Flan-T5-Base (LoRA)
    使用 Encoder Injection 方法防止信息泄露
    """
    
    def __init__(self, use_gradient_checkpointing: bool = True):
        super().__init__()
        
        print("=" * 60)
        print("Initializing DinoT5LoRAModel (Flan-T5-Base)")
        print("=" * 60)
        
        # 1. 加载 DINOv2 Vision Encoder
        print("\n[1/3] Loading DINOv2-Base...")
        self.vision_model = Dinov2Model.from_pretrained(VISION_MODEL_PATH)
        
        # 冻结 DINOv2 基础权重
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        # 应用 LoRA 到 DINOv2
        vision_lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=VISION_LORA_TARGETS,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.vision_model = get_peft_model(self.vision_model, vision_lora_config)
        
        # 开启梯度检查点 (节省显存)
        if use_gradient_checkpointing:
            self.vision_model.enable_input_require_grads()
            self.vision_model.gradient_checkpointing_enable()
        
        print(f"  ✓ DINOv2 loaded with LoRA (r={LORA_R})")
        self.vision_model.print_trainable_parameters()
        
        # 2. 构建 SwiGLU Connector
        print("\n[2/3] Building SwiGLU Connector...")
        self.connector = SwiGLUConnector()
        print(f"  ✓ SwiGLU: {DINO_HIDDEN_SIZE} -> {MLP_HIDDEN_SIZE} -> {T5_HIDDEN_SIZE}")
        
        # 3. 加载 Flan-T5-Base Language Model
        print("\n[3/3] Loading Flan-T5-Base...")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            TEXT_MODEL_PATH
        )
        
        # 冻结 T5 基础权重
        for param in self.t5_model.parameters():
            param.requires_grad = False
        
        # 应用 LoRA 到 T5
        text_lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TEXT_LORA_TARGETS,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, text_lora_config)
        
        # 开启梯度检查点 (T5 部分)
        if use_gradient_checkpointing:
            self.t5_model.gradient_checkpointing_enable()
        
        print(f"  ✓ Flan-T5-Base loaded with LoRA (r={LORA_R})")
        self.t5_model.print_trainable_parameters()
        
        # 4. 初始化 VC-BDR Loss
        self.vcbdr_loss_fn = VC_BDR_Loss(temperature=0.07)
        
        print("\n" + "=" * 60)
        print("✓ Model initialization complete")
        print("  Architecture: DINOv2-Base -> SwiGLU -> Flan-T5-Base")
        print("  VC-BDR Loss: Enabled")
        print("=" * 60)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
        use_vcbdr: bool = False
    ) -> dict:
        """
        前向传播 - 使用 Encoder Injection 方法防止信息泄露
        支持 VC-BDR 辅助损失
        
        Args:
            pixel_values: [batch_size, 3, 518, 518]
            labels: [batch_size, max_length] - 目标文本,padding位置为-100
            use_vcbdr: 是否启用 VC-BDR 损失
        
        Returns:
            dict: {
                'loss': torch.Tensor - 总损失 (主损失 + VC-BDR)
                'main_loss': torch.Tensor - 主任务损失
                'aux_loss': torch.Tensor - VC-BDR 辅助损失 (如果启用)
                'logits': torch.Tensor,
                'vision_features': torch.Tensor (可选)
            }
        """
        batch_size = pixel_values.shape[0]
        
        # 1. 提取视觉特征 - DINOv2 Encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        last_hidden_state = vision_outputs.last_hidden_state
        # Shape: [batch_size, 1369, 768] (37*37=1369 patches for 518px)
        
        # 2. 通过 SwiGLU Connector 映射到 T5 输入空间
        inputs_embeds = self.connector(last_hidden_state)
        # Shape: [batch_size, 1369, 768]
        
        # 3. 关键防泄露步骤: 显式构造 decoder_input_ids
        # 从 labels 生成 decoder 输入 (shifted right for teacher forcing)
        decoder_input_ids = self.t5_model.prepare_decoder_input_ids_from_labels(
            labels=labels
        )
        
        # 4. Encoder Injection: 调用 T5 with inputs_embeds
        # inputs_embeds 会激活 T5 Encoder 栈处理视觉特征
        # decoder_input_ids 确保正确的 teacher forcing
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,           # 视觉特征注入 T5 Encoder
            decoder_input_ids=decoder_input_ids,   # 显式 Decoder 输入
            labels=labels,                         # 用于计算损失
            output_hidden_states=use_vcbdr,        # 仅在需要时输出隐藏状态
            return_dict=True
        )
        
        # 主任务损失 (CrossEntropy)
        main_loss = outputs.loss
        
        result = {
            'main_loss': main_loss,
            'logits': outputs.logits,
            'vision_features': last_hidden_state
        }
        
        # 5. 计算 VC-BDR 辅助损失 (如果启用)
        if use_vcbdr:
            # 获取 decoder 最后一层隐藏状态
            decoder_hidden_states = outputs.decoder_hidden_states[-1]  # [B, seq_len, 768]
            
            # 创建 attention mask (labels != -100 的位置)
            attention_mask = (labels != -100).long()
            
            # 计算 VC-BDR 损失
            aux_loss = self.vcbdr_loss_fn(
                vision_features=last_hidden_state,
                text_hidden_states=decoder_hidden_states,
                attention_mask=attention_mask
            )
            
            # 总损失 = 主损失 + lambda * 辅助损失
            total_loss = main_loss + VCBDR_WEIGHT * aux_loss
            
            result['aux_loss'] = aux_loss
            result['loss'] = total_loss
        else:
            result['aux_loss'] = torch.tensor(0.0, device=main_loss.device)
            result['loss'] = main_loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 200,
        num_beams: int = 5,
        repetition_penalty: float = 1.2,
        **kwargs
    ) -> torch.Tensor:
        """
        推理生成 - 图像到文本
        
        Args:
            pixel_values: [batch_size, 3, 518, 518]
            max_length: 最大生成长度
            num_beams: Beam search 宽度
            repetition_penalty: 重复惩罚
        
        Returns:
            generated_ids: [batch_size, seq_len]
        """
        # 1. 提取视觉特征 - 关键修正: 使用关键字参数
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        last_hidden_state = vision_outputs.last_hidden_state
        
        # 2. MLP 映射
        inputs_embeds = self.connector(last_hidden_state)
        
        # 3. T5 生成
        generated_ids = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
            **kwargs
        )
        
        return generated_ids
    
    def get_trainable_params(self) -> dict:
        """
        获取可训练参数组 (用于设置不同学习率)
        
        Returns:
            dict: {
                'mlp_params': MLP 参数列表,
                'lora_params': 所有 LoRA 参数列表
            }
        """
        mlp_params = list(self.connector.parameters())
        
        # 收集 Vision 和 Text 的 LoRA 参数
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params.append(param)
        
        return {
            'mlp_params': mlp_params,
            'lora_params': lora_params
        }
    
    def save_trainable(self, save_dir: str):
        """保存可训练组件 - 改进版"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving trainable components to {save_dir}...")
        
        # 1. 保存 MLP Connector
        mlp_path = save_dir / "mlp_connector.pt"
        torch.save(self.connector.state_dict(), mlp_path)
        print(f"  ✓ MLP saved: {mlp_path}")
        
        # 2. 保存 Vision LoRA
        vision_lora_dir = save_dir / "vision_lora"
        self.vision_model.save_pretrained(vision_lora_dir)
        print(f"  ✓ Vision LoRA saved: {vision_lora_dir}")
        
        # 3. 保存 Text LoRA
        text_lora_dir = save_dir / "text_lora"
        self.t5_model.save_pretrained(text_lora_dir)
        print(f"  ✓ Text LoRA saved: {text_lora_dir}")
        
        # 4. 保存模型配置信息
        import json
        config_info = {
            'vision_model': VISION_MODEL_PATH,
            'text_model': TEXT_MODEL_PATH,
            'lora_r': LORA_R,
            'image_size': IMAGE_SIZE,
        }
        with open(save_dir / 'model_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
        print(f"  ✓ Model config saved")


    def load_trainable(self, load_dir: str):
        """加载可训练组件 - 改进版"""
        load_dir = Path(load_dir)
        
        print(f"Loading trainable components from {load_dir}...")
        
        # 1. 加载 MLP
        mlp_path = load_dir / "mlp_connector.pt"
        if mlp_path.exists():
            state_dict = torch.load(mlp_path, map_location='cpu')
            missing, unexpected = self.connector.load_state_dict(state_dict, strict=False)
            
            if missing:
                print(f"  ⚠️ Missing keys in MLP: {missing}")
            if unexpected:
                print(f"  ⚠️ Unexpected keys in MLP: {unexpected}")
            
            print(f"  ✓ MLP loaded from {mlp_path}")
        else:
            raise FileNotFoundError(f"MLP checkpoint not found: {mlp_path}")
        
        # 2. 加载 Vision LoRA
        vision_lora_dir = load_dir / "vision_lora"
        if vision_lora_dir.exists():
            from peft import PeftModel
            self.vision_model = PeftModel.from_pretrained(
                self.vision_model.get_base_model(), 
                vision_lora_dir
            )
            print(f"  ✓ Vision LoRA loaded from {vision_lora_dir}")
        else:
            print(f"  ⚠️ Vision LoRA not found, using base model")
        
        # 3. 加载 Text LoRA
        text_lora_dir = load_dir / "text_lora"
        if text_lora_dir.exists():
            from peft import PeftModel
            self.t5_model = PeftModel.from_pretrained(
                self.t5_model.get_base_model(),
                text_lora_dir
            )
            print(f"  ✓ Text LoRA loaded from {text_lora_dir}")
        else:
            print(f"  ⚠️ Text LoRA not found, using base model")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    """测试模型构建和前向传播"""
    print("\nTesting DinoT5LoRAModel...")
    
    # 创建模型
    model = DinoT5LoRAModel(use_gradient_checkpointing=True)
    model.eval()
    
    # 创建虚拟输入
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 518, 518)
    labels = torch.randint(0, 1000, (batch_size, 200))
    labels[:, 100:] = -100  # 模拟 padding
    
    print("\n" + "=" * 60)
    print("Testing forward pass...")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, labels=labels)
    
    print(f"✓ Loss: {outputs['loss'].item():.4f}")
    print(f"✓ Logits shape: {outputs['logits'].shape}")
    print(f"✓ Vision features shape: {outputs['vision_features'].shape}")
    
    # 测试生成
    print("\n" + "=" * 60)
    print("Testing generation...")
    
    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_length=50,
        num_beams=3
    )
    
    print(f"✓ Generated IDs shape: {generated_ids.shape}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("Parameter Statistics:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # 测试参数分组
    param_groups = model.get_trainable_params()
    mlp_params_count = sum(p.numel() for p in param_groups['mlp_params'])
    lora_params_count = sum(p.numel() for p in param_groups['lora_params'])
    
    print(f"  MLP params: {mlp_params_count:,}")
    print(f"  LoRA params: {lora_params_count:,}")
    print("=" * 60)