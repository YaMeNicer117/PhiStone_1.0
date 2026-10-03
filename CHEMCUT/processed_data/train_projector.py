import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

# 屏蔽 RDKit 烦人的底层警告信息
RDLogger.DisableLog('rdApp.*')

# =============================================================================
# [ 终极超参数配置区 ]
# =============================================================================
# --- 路径配置 ---
INPUT_VOCAB_PATH = "fragment_vocabulary_768.json"
OUTPUT_VOCAB_PATH = "fragment_vocabulary_64_dynamic_margin.json"

# --- 网络结构与训练维度 ---
INPUT_DIM = 768        
HIDDEN_DIM = 256       
TARGET_DIM = 64        

# --- 训练策略 ---
EPOCHS = 600              # [用户指定] 总轮数提升至 200，确保空间变形充分收敛
BATCH_SIZE = 256          # [极佳平衡点] 3098个样本，每个Epoch 12步，负样本足够多样
LEARNING_RATE = 1e-4      # [平滑调参] 降低至 2e-4，防止拓扑保序 MSE 损失发生震荡

# --- 双引擎核心参数 ---
BASE_MARGIN = 2.0      # 基础排斥距离 (大片段之间的最小间距)
ALPHA_PENALTY = 4.0    # 焦点放大系数 (HAC=1 的单原子排斥距离会被放大 5 倍)
POS_NOISE_STD = 0.05   # 自我对比时加入的正样本高斯噪声标准差
LAMBDA_TOPO = 10.0     # 拓扑保持正则化系数 (强迫 64 维严格模仿 768 维的语义排布)

# 兜底机制：对于解析失败的特例，赋予大片段权重，让其退化为基础 Margin
FALLBACK_HAC = 100.0    
# =============================================================================


# =============================================================================
# 1. 模型与损失函数定义
# =============================================================================
class FragmentProjector(nn.Module):
    """
    轻量级非线性投影头：将 768 维映射为 64 维。
    使用 SiLU 激活函数和 LayerNorm 保证潜空间的平滑过渡。
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
    """
    def __init__(self, base_margin=BASE_MARGIN, alpha=ALPHA_PENALTY):
        super().__init__()
        self.base_margin = base_margin
        self.alpha = alpha

    def forward(self, anchor_out, pos_out, neg_out, hac_anchor):
        d_ap = F.pairwise_distance(anchor_out, pos_out, p=2.0)
        d_an = F.pairwise_distance(anchor_out, neg_out, p=2.0)
        
        # 【安全锁 1】：强制 HAC 最小为 1.0，杜绝除零导致 inf 报错
        safe_hac = torch.clamp(hac_anchor, min=1.0)
        
        dynamic_margin = self.base_margin * (1.0 + self.alpha / safe_hac)
        losses = F.relu(d_ap - d_an + dynamic_margin)
        return losses.mean()


# =============================================================================
# 2. 核心训练逻辑
# =============================================================================
def train_dynamic_projection_space():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 开始训练 64 维神级拓扑潜空间 | 使用设备: {device}")
    
    # -------------------------------------------------------------------------
    # A. 数据加载与清洗
    # -------------------------------------------------------------------------
    if not os.path.exists(INPUT_VOCAB_PATH):
        raise FileNotFoundError(f"找不到词汇表文件: {INPUT_VOCAB_PATH}")
        
    print(f"正在读取 768 维原始词汇表: {INPUT_VOCAB_PATH}")
    with open(INPUT_VOCAB_PATH, 'r', encoding='utf-8') as f: 
        vocab_768d = json.load(f)

    # 【安全锁 2】：果断剔除全零占位符，防止度量空间坍缩
    if "<UNK>" in vocab_768d:
        vocab_768d.pop("<UNK>")
        print("🔧 已成功剔除 <UNK> 全零占位符，保护空间纯洁度。")

    hac_dict = {}
    
    for smi in tqdm(vocab_768d.keys(), desc="计算重原子数 (HAC)"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # 【安全锁 3】：即使是 [HH] 这种骨架为空的特例，也强制按单原子对待
            hac_dict[smi] = max(1.0, float(mol.GetNumHeavyAtoms()))
        else:
            # 解析失败兜底
            hac_dict[smi] = FALLBACK_HAC

    smiles_list = list(vocab_768d.keys())
    tensors_768 = torch.tensor(list(vocab_768d.values()), dtype=torch.float32).to(device)
    hacs = torch.tensor([hac_dict[smi] for smi in smiles_list], dtype=torch.float32).to(device)
    
    num_samples = len(smiles_list)
    print(f"✅ 数据准备完毕！清洗后剩余 {num_samples} 个有效的化学片段。")

    # -------------------------------------------------------------------------
    # B. 模型与优化器初始化
    # -------------------------------------------------------------------------
    projector = FragmentProjector().to(device)
    criterion_triplet = HACDynamicTripletLoss()
    optimizer = optim.AdamW(projector.parameters(), lr=LEARNING_RATE)

    # -------------------------------------------------------------------------
    # C. 双引擎微调循环
    # -------------------------------------------------------------------------
    print(f"\n开始微调 (总轮数: {EPOCHS}, 批次大小: {BATCH_SIZE})...")
    projector.train()
    
    for epoch in range(EPOCHS):
        indices = torch.randperm(num_samples)
        epoch_loss_triplet = 0.0
        epoch_loss_topo = 0.0
        
        pbar = tqdm(range(0, num_samples, BATCH_SIZE), desc=f"Epoch {epoch+1:03d}/{EPOCHS}", leave=False)
        for i in pbar:
            batch_idx = indices[i : i+BATCH_SIZE]
            
            anchor_768 = tensors_768[batch_idx]
            anchor_hac = hacs[batch_idx]
            
            # --- 构造样本对 ---
            # 正样本：原位抖动，加上微小的高斯噪声
            pos_768 = anchor_768 + torch.randn_like(anchor_768) * POS_NOISE_STD
            
            # 负样本：在当前全量词表中随机抽取
            neg_idx = torch.randint(0, num_samples, (len(batch_idx),))
            neg_768 = tensors_768[neg_idx]
            
            # --- 前向传播降维 ---
            optimizer.zero_grad()
            out_anchor = projector(anchor_768)
            out_pos = projector(pos_768)
            out_neg = projector(neg_768)
            
            # --- 引擎 A: 动态边界扩边 ---
            loss_triplet = criterion_triplet(out_anchor, out_pos, out_neg, anchor_hac)
            
            # --- 引擎 B: 拓扑保序约束 ---
            # 算 768 维真实余弦相似度排布
            sim_768 = F.cosine_similarity(anchor_768.unsqueeze(1), anchor_768.unsqueeze(0), dim=2)
            # 算 64 维降维后的余弦相似度排布
            sim_64 = F.cosine_similarity(out_anchor.unsqueeze(1), out_anchor.unsqueeze(0), dim=2)
            
            # 强制用 MSE 拉齐两个空间的语义拓扑
            loss_topo = F.mse_loss(sim_64, sim_768)
            
            # --- 融合损失与反向传播 ---
            loss = loss_triplet + LAMBDA_TOPO * loss_topo
            loss.backward()
            optimizer.step()
            
            epoch_loss_triplet += loss_triplet.item()
            epoch_loss_topo += loss_topo.item()
            
            pbar.set_postfix({
                "L_Trip": f"{loss_triplet.item():.4f}", 
                "L_Topo": f"{loss_topo.item():.4f}"
            })
            
        # 每 10 轮打印一次日志
        if (epoch + 1) % 10 == 0:
            avg_trip = epoch_loss_triplet / (num_samples/BATCH_SIZE)
            avg_topo = epoch_loss_topo / (num_samples/BATCH_SIZE)
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | Triplet(扩边): {avg_trip:.4f} | Topo(保序): {avg_topo:.4f}")

    # -------------------------------------------------------------------------
    # D. 导出词汇表
    # -------------------------------------------------------------------------
    print(f"\n🎉 200 轮训练结束！正在推演全新的 64 维特征...")
    projector.eval()
    new_vocab_64d = {}
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_smi = smiles_list[i : i+BATCH_SIZE]
            batch_768 = tensors_768[i : i+BATCH_SIZE]
            batch_64 = projector(batch_768).cpu().tolist()
            
            for smi, vec_64 in zip(batch_smi, batch_64):
                new_vocab_64d[smi] = vec_64
    new_vocab_64d["<UNK>"] = [0.0] * TARGET_DIM

    # 写入 JSON (优雅换行版)
    with open(OUTPUT_VOCAB_PATH, 'w', encoding='utf-8') as f:
        f.write("{\n")
        
        # 把 <UNK> 拎出来放在第一行
        unk_vec = new_vocab_64d.pop("<UNK>")
        f.write(f'  "<UNK>": {json.dumps(unk_vec)}')
        
        sorted_items = sorted(new_vocab_64d.items())
        if sorted_items:
            f.write(",\n")
            for i, (smi, vec) in enumerate(sorted_items):
                line = f'  {json.dumps(smi, ensure_ascii=False)}: {json.dumps(vec)}'
                if i < len(sorted_items) - 1:
                    f.write(f"{line},\n")
                else:
                    f.write(f"{line}\n")
        f.write("}")
        
    print(f"✅ 全新动态边界词汇表已成功保存至: {OUTPUT_VOCAB_PATH}")
    
    torch.save(projector.state_dict(), "fragment_projector.pth")
    print("✅ 投影网络权重已保存至: fragment_projector.pth")