import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# [ 超参数配置区 - 投影网络与空间边界 ]
# =============================================================================
# --- 网络结构维度 ---
INPUT_DIM = 768        # ChemBERTa 提取出的原始特征维度
HIDDEN_DIM = 256       # MLP 隐藏层维度 (若觉得表达能力不够，可调大至 512)
TARGET_DIM = 64        # 目标潜空间维度 (与你当前 SE3TD 扩散模型严格对齐)

# --- 动态排斥边界 (Margin) 参数 ---
# 公式: Margin = BASE_MARGIN * (1.0 + ALPHA_PENALTY / HAC)
BASE_MARGIN = 2.0      # 基础排斥距离。决定了普通大片段之间(HAC很大时)至少要保持多远的距离
ALPHA_PENALTY = 4.0    # 焦点放大系数。控制单原子(HAC=1)的领地能膨胀多少倍。
                       # 按当前设定，单原子的排斥距离为 2.0 * (1 + 4/1) = 10.0，是普通片段的 5 倍！
# =============================================================================

class FragmentProjector(nn.Module):
    """
    轻量级非线性投影头：将 768 维映射为 64 维。
    加入 LayerNorm 和 SiLU 激活函数，确保输出的空间对扩散模型极其平滑友好。
    """
    def __init__(self, in_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, out_dim=TARGET_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class HACDynamicTripletLoss(nn.Module):
    """
    HAC 动态边界三元组损失 (引擎 A：扩充单原子靶区)
    强迫不同片段(负样本)互相远离，且 HAC 越小的片段，排斥力越强，专属空间越大。
    """
    def __init__(self, base_margin=BASE_MARGIN, alpha=ALPHA_PENALTY):
        super().__init__()
        self.base_margin = base_margin
        self.alpha = alpha

    def forward(self, anchor_out, pos_out, neg_out, hac_anchor):
        """
        参数:
        anchor_out: 锚点的 64 维预测向量
        pos_out:    正样本的 64 维预测向量
        neg_out:    负样本的 64 维预测向量
        hac_anchor: 锚点片段的重原子数 [Batch_Size]
        """
        # 计算 64 维空间中的 L2 欧氏距离
        d_ap = F.pairwise_distance(anchor_out, pos_out, p=2.0)
        d_an = F.pairwise_distance(anchor_out, neg_out, p=2.0)

        # 动态计算排斥边界 (HAC 越小，Margin 越大)
        dynamic_margin = self.base_margin * (1.0 + self.alpha / hac_anchor)

        # Triplet Loss = max(距离_正 - 距离_负 + 动态边界, 0)
        losses = F.relu(d_ap - d_an + dynamic_margin)
        
        return losses.mean()