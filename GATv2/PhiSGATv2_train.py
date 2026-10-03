# PhiSGATv2_train.py 

import os
import time
import torch
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# 导入我们拆分好的各个模块
import PhiSGATv2_config as config
from PhiSGATv2_dataset import MoleculeDataset
from PhiSGATv2_model import GATv2Model
import PhiSGATv2_utils as utils

def main():
    """
    主函数，执行整个模型的训练、验证和评估流程。
    """
    # --- 1. 环境设置 (此部分保持不变) ---
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print(f"将使用设备: {config.DEVICE}")
    print("-" * 50)

    # --- 2. 准备和划分数据集 (此部分保持不变) ---
    print("开始加载数据集...")
    try:
        # 加载包含训练和验证数据的主数据集
        full_dataset = MoleculeDataset(root=config.CHEMCUT_ROOT, data_type='train_val')
        print(f"成功加载训练/验证数据集，总共 {len(full_dataset)} 个分子。")

        # 定义训练集和验证集的数量
        dataset_size = len(full_dataset)
        if dataset_size < 2:
            print("错误: 训练/验证数据至少需要 2 个分子，当前数据量不足。")
            return

        train_size = int(config.TRAIN_RATIO * dataset_size)
        train_size = min(max(train_size, 1), dataset_size - 1)
        val_size = dataset_size - train_size

        # 使用 random_split 进行划分
        split_generator = torch.Generator().manual_seed(config.RANDOM_SEED)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=split_generator
        )

        print(f"数据集划分完成:")
        print(f" - 训练集数量: {len(train_dataset)}")
        print(f" - 验证集数量: {len(val_dataset)}")
        
    except (RuntimeError, FileNotFoundError) as e:
        print(f"加载训练/验证数据时出错: {e}")
        return

    try:
        # 加载独立的测试集
        test_dataset = MoleculeDataset(root=config.CHEMCUT_ROOT, data_type='test')
        if len(test_dataset) > 0:
            print(f" - 独立测试集数量: {len(test_dataset)}")
        else:
            print(" - 独立测试集为空或不存在，将跳过最终评估。")
            test_dataset = None
    except (RuntimeError, FileNotFoundError) as e:
        print(f"加载独立测试数据时出错: {e}")
        test_dataset = None
    
    print("-" * 50)

    # --- 3. 创建 DataLoader (此部分保持不变) ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        exclude_keys=config.EXCLUDE_KEYS_IN_LOADER
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        exclude_keys=config.EXCLUDE_KEYS_IN_LOADER
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            exclude_keys=config.EXCLUDE_KEYS_IN_LOADER
        )

    # --- 4. 实例化模型、优化器 ---
    
    # --- [已移除] ---
    # 原本需要从数据集中获取词汇表大小
    # vocab_size = len(full_dataset.vocabulary)
    
    # --- [修订] ---
    # 实例化模型时不再需要传入 vocab_size
    model = GATv2Model().to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    print("模型和优化器已成功创建。")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 50)

    # --- 5. 主训练循环 (此部分保持不变) ---
    best_val_loss = float('inf')
    best_epoch = 0

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    print("开始训练...")
    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()
        
        train_loss = utils.train_epoch(model, train_loader, optimizer, config.DEVICE)
        # [修改] 接收回归指标
        val_loss, val_metrics, val_avg_rmse, val_avg_mae = utils.evaluate(model, val_loader, config.DEVICE)
        
        epoch_duration = time.time() - epoch_start_time
        
        # [修改] 打印输出回归的 Loss (MSE)、RMSE 和 MAE
        print(f"Epoch {epoch:03d}/{config.EPOCHS} | "
              f"Time: {epoch_duration:.2f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val RMSE: {val_avg_rmse:.4f} | Val MAE: {val_avg_mae:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"    -> 验证损失降低，模型已保存至 '{config.MODEL_SAVE_PATH}'")

    total_training_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_training_time / 60:.2f} 分钟")
    print(f"最佳模型出现在 Epoch {best_epoch}，对应的最低验证损失为: {best_val_loss:.4f}")
    print("-" * 50)

    # --- 6. 最终评估 ---
    if test_loader:
        print("在独立测试集上进行最终评估...")
        
        try:
            # --- [修订] ---
            # 重新实例化模型时也不再需要 vocab_size
            final_model = GATv2Model().to(config.DEVICE)
            
            final_model.load_state_dict(torch.load(
                config.MODEL_SAVE_PATH,
                map_location=config.DEVICE,
                weights_only=True
            ))
            print(f"成功从 '{config.MODEL_SAVE_PATH}' 加载最佳模型权重。")

            # 在测试集上执行评估
            test_loss, test_metrics, test_avg_rmse, test_avg_mae = utils.evaluate(
                final_model, test_loader, config.DEVICE
            )

            print("\n--- [ 最终模型性能评估报告 ] ---")
            print(f"在独立测试集上的表现:")
            print(f"  -> 平均损失 (MSE Loss): {test_loss:.4f}")
            print(f"  -> 平均 RMSE (更低更好): {test_avg_rmse:.4f}")
            print(f"  -> 平均 MAE  (更低更好): {test_avg_mae:.4f}")
            print("\n  --- 各项任务回归指标详情 ---")
            for i, metric in test_metrics.items():
                print(f"    -> 任务 {i} | RMSE: {metric['RMSE']:.4f} | MAE: {metric['MAE']:.4f}")
            print("------------------------------------")
            
        except FileNotFoundError:
            print(f"错误: 找不到已保存的模型文件 '{config.MODEL_SAVE_PATH}'。")
        except Exception as e:
            print(f"评估过程中发生未知错误: {e}")
            
    else:
        print("未找到测试数据加载器 (test_loader)，跳过最终评估。")

if __name__ == '__main__':
    main()
