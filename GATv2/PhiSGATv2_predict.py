# PhiSGATv2_predict.py (回归适配版)

import os
import torch
from torch_geometric.loader import DataLoader

# 导入我们拆分好的各个模块
import PhiSGATv2_config as config
from PhiSGATv2_dataset import MoleculeDataset
from PhiSGATv2_model import GATv2Model

def predict():
    """
    使用训练好的最佳模型对未知分子数据进行回归预测。
    """
    print("-" * 50)
    print("开始执行分子活性预测...")
    print(f"将使用设备: {config.DEVICE}")

    # --- 1. 定义未知数据路径 ---
    unknown_data_dir = config.UNKNOWN_DATA_DIR
    
    if not os.path.exists(unknown_data_dir):
        print(f"错误: 预测数据目录不存在: '{unknown_data_dir}'")
        return

    # --- 2. 实例化模型并加载权重 ---
    print("\n正在加载模型...")
    try:
        model = GATv2Model().to(config.DEVICE)
        
        # 加载最佳模型权重
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=True))
        model.eval()  # !! 必须设置为评估模式 !!
        
        print(f"成功从 '{config.MODEL_SAVE_PATH}' 加载模型。")

    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{config.MODEL_SAVE_PATH}'。请先运行训练脚本。")
        return
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    # --- 3. 准备预测数据集和 DataLoader ---
    print(f"\n正在从 '{unknown_data_dir}' 加载待预测数据...")
    
    predict_dataset = MoleculeDataset(root=config.CHEMCUT_ROOT, data_type='unknown')

    if not predict_dataset.data_files:
        print(f"错误: 在预测目录 '{unknown_data_dir}' 中没有找到任何 .pt 文件。")
        return
        
    print(f"找到了 {len(predict_dataset)} 个待预测的分子。")

    predict_loader = DataLoader(
        predict_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        exclude_keys=[
            key for key in config.EXCLUDE_KEYS_IN_LOADER
            if key not in ('mol_id',)
        ]
    )

    # --- 4. 执行预测 ---
    all_predictions = []
    all_mol_ids = []

    print("\n开始进行推理...")
    with torch.no_grad():
        for batch in predict_loader:
            batch = batch.to(config.DEVICE)
            
            # 模型前向传播，此时 out 的 shape 直接是: [batch_size, num_tasks]
            out = model(batch)
            
            # [核心修改]: 移除 out.argmax(dim=-1)，直接收集回归的浮点数值
            all_predictions.extend(out.cpu().numpy().tolist())
            
            if hasattr(batch, 'mol_id'):
                all_mol_ids.extend(batch.mol_id)
            else:
                num_graphs = batch.num_graphs
                all_mol_ids.extend([f"Unknown_Molecule_{i}" for i in range(len(all_mol_ids), len(all_mol_ids) + num_graphs)])


    # --- 5. 展示预测结果 ---
    print("\n--- [ 预测结果报告 ] ---")
    if not all_mol_ids:
        print("未能获取任何分子ID，无法展示详细报告。")
    elif not all_predictions:
        print("未能生成任何预测结果。")
    else:
        for mol_id, preds in zip(all_mol_ids, all_predictions):
            # [核心修改]: 格式化浮点数输出，保留 4 位小数，使界面更清爽
            preds_str = " | ".join([f" activity {i}: {p:.4f}" for i, p in enumerate(preds)])
            print(f"分子 ID: {mol_id:<20} -> 预测连续活性: [ {preds_str} ]")
    
    print("-" * 50)

if __name__ == '__main__':
    predict()
