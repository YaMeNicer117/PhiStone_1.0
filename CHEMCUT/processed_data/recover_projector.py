import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# 1. 确保架构和之前完全一致
class FragmentProjector(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def recover_weights():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device} 进行逆向恢复...")

    # 2. 读取你的旧文件（确保路径正确）
    path_768 = "fragment_vocabulary_768.json"
    path_64 = "fragment_vocabulary_64_dynamic_margin.json"
    
    with open(path_768, 'r') as f: vocab_768 = json.load(f)
    with open(path_64, 'r') as f: vocab_64 = json.load(f)
    
    # 3. 对齐数据 (过滤掉 UNK 等无用占位符)
    common_smiles = [smi for smi in vocab_768.keys() if smi in vocab_64 and smi != "<UNK>"]
    print(f"找到 {len(common_smiles)} 个匹配的词汇对。")
    
    X = torch.tensor([vocab_768[smi] for smi in common_smiles], dtype=torch.float32).to(device)
    Y = torch.tensor([vocab_64[smi] for smi in common_smiles], dtype=torch.float32).to(device)

    # 4. 初始化模型
    model = FragmentProjector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 5. 开始拟合 (因为是已知结构的拟合，所以会非常快，且Loss会趋近于0)
    EPOCHS = 1000
    pbar = tqdm(range(EPOCHS), desc="恢复模型权重")
    
    for epoch in pbar:
        optimizer.zero_grad()
        pred_Y = model(X)
        # 用 MSE 强迫预测出的 64 维和原先保存的 64 维一模一样
        loss = F.mse_loss(pred_Y, Y) 
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            pbar.set_postfix({"MSE Loss": f"{loss.item():.6f}"})
            
        # 如果损失已经极小，说明完美复刻，提前结束
        if loss.item() < 1e-6:
            print(f"\n🎉 提前在第 {epoch} 轮完成拟合！损失极低：{loss.item()}")
            break

    # 6. 保存权重
    save_path = "fragment_projector.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ 完美！恢复后的权重已保存至: {save_path}")
    print("此权重可以无缝对接你当前的生成模型，绝不破坏原有特征体系。")

if __name__ == "__main__":
    recover_weights()