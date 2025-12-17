"""
VC-BDR Loss: Visually-Constrained Batch Diversity Regularization
用于防止后验崩塌的辅助损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VC_BDR_Loss(nn.Module):
    """
    VC-BDR (Visually-Constrained Batch Diversity Regularization) Loss
    
    核心思想:
    - 如果两张图片视觉上相似，则它们的文本描述也应相似
    - 如果两张图片视觉上不同，则文本描述应该有差异
    
    实现逻辑:
    1. 聚合视觉特征 (Mean Pooling + Normalize)
    2. 聚合文本隐藏状态 (Masked Mean Pooling + Normalize)
    3. 计算视觉相似度矩阵 V_sim: [B, B]
    4. 计算文本相似度矩阵 T_sim: [B, B]
    5. Loss = Mean( T_sim * ReLU(1.0 - V_sim) )
    
    含义:
    - 当 V_sim 低时 (图片不同)，ReLU(1 - V_sim) 大，惩罚高 T_sim (文本相似)
    - 当 V_sim 高时 (图片相似)，ReLU(1 - V_sim) 小，允许高 T_sim
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 温度系数，控制相似度分布的锐度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算 VC-BDR 损失
        
        Args:
            vision_features: [batch_size, seq_len_v, hidden_dim_v] - DINOv2 输出
            text_hidden_states: [batch_size, seq_len_t, hidden_dim_t] - T5 Decoder 输出
            attention_mask: [batch_size, seq_len_t] - 文本 attention mask (可选)
        
        Returns:
            loss: VC-BDR 损失标量
        """
        batch_size = vision_features.shape[0]
        
        # 1. 聚合视觉特征 (Mean Pooling over sequence)
        vision_agg = vision_features.mean(dim=1)  # [B, hidden_dim_v]
        vision_agg = F.normalize(vision_agg, p=2, dim=-1)  # L2 归一化
        
        # 2. 聚合文本隐藏状态 (Masked Mean Pooling)
        if attention_mask is not None:
            # 将 attention_mask 扩展到与 hidden_states 相同维度
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len_t, 1]
            
            # Masked sum
            text_sum = (text_hidden_states * mask_expanded).sum(dim=1)  # [B, hidden_dim_t]
            
            # 计算有效长度
            valid_lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            
            # Mean
            text_agg = text_sum / valid_lengths  # [B, hidden_dim_t]
        else:
            # 无 mask，直接 mean pooling
            text_agg = text_hidden_states.mean(dim=1)  # [B, hidden_dim_t]
        
        text_agg = F.normalize(text_agg, p=2, dim=-1)  # L2 归一化
        
        # 3. 计算相似度矩阵
        # Vision Similarity: [B, B]
        vision_sim = torch.matmul(vision_agg, vision_agg.T) / self.temperature
        
        # Text Similarity: [B, B]
        text_sim = torch.matmul(text_agg, text_agg.T) / self.temperature
        
        # 4. Mask 对角线 (排除自相似度)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=vision_features.device)
        
        # 5. 计算 VC-BDR 损失
        # 核心公式: Loss = Mean( T_sim * ReLU(1.0 - V_sim) )
        # 当 V_sim 低时，惩罚高 T_sim
        diversity_penalty = text_sim * F.relu(1.0 - vision_sim)
        
        # 只计算非对角线元素
        loss = diversity_penalty[mask].mean()
        
        return loss


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    """测试 VC-BDR Loss"""
    print("Testing VC-BDR Loss...")
    
    # 创建虚拟数据
    batch_size = 4
    vision_seq_len = 1369  # DINOv2 patch 数量 (37*37)
    text_seq_len = 50
    vision_dim = 768
    text_dim = 768
    
    vision_features = torch.randn(batch_size, vision_seq_len, vision_dim)
    text_hidden_states = torch.randn(batch_size, text_seq_len, text_dim)
    attention_mask = torch.ones(batch_size, text_seq_len)
    
    # 模拟 padding (最后 20 个 token)
    attention_mask[:, -20:] = 0
    
    # 计算损失
    loss_fn = VC_BDR_Loss(temperature=0.07)
    loss = loss_fn(vision_features, text_hidden_states, attention_mask)
    
    print(f"✓ VC-BDR Loss: {loss.item():.4f}")
    print(f"✓ Loss shape: {loss.shape}")
    print(f"✓ Loss is scalar: {loss.dim() == 0}")
    
    # 测试梯度
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\n" + "=" * 60)
    print("VC-BDR Loss Test Complete")
    print("=" * 60)
