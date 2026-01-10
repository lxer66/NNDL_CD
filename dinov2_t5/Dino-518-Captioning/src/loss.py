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

    目标:
    - 视觉不相似时，惩罚文本过于相似
    - 视觉相似时，放松约束
    """

    def __init__(self, lambda_scale: float = 1.0, margin: float = 1.0):
        super().__init__()
        self.lambda_scale = lambda_scale
        self.margin = margin

    def forward(
        self,
        vision_features: torch.Tensor,
        text_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [B, seq_len_v, hidden_v]
            text_hidden_states: [B, seq_len_t, hidden_t]
            attention_mask: [B, seq_len_t] (1 for valid, 0 for pad)
        Returns:
            loss: scalar VC-BDR loss
        """
        batch_size = vision_features.shape[0]

        # 1) Vision pooling + L2 norm
        vision_agg = F.normalize(vision_features.mean(dim=1), p=2, dim=-1)  # [B, d]

        # 2) Text masked mean pooling + L2 norm
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len_t, 1]
            text_sum = (text_hidden_states * mask_expanded).sum(dim=1)
            valid_lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
            text_agg = text_sum / valid_lengths
        else:
            text_agg = text_hidden_states.mean(dim=1)
        text_agg = F.normalize(text_agg, p=2, dim=-1)  # [B, d]

        # 3) Cosine similarity matrices (no temperature)
        vision_sim = torch.matmul(vision_agg, vision_agg.T)  # [B, B]
        text_sim = torch.matmul(text_agg, text_agg.T)        # [B, B]

        # 4) Weights
        visual_weight = F.relu(self.margin - vision_sim)
        text_weight = F.relu(text_sim)
        loss_matrix = visual_weight * text_weight

        # 5) Mask diagonal and reduce
        off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=vision_features.device)
        loss = loss_matrix[off_diag].mean()

        return loss * self.lambda_scale


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
    loss_fn = VC_BDR_Loss(lambda_scale=1.0, margin=1.0)
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
