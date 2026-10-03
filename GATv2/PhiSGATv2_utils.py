# PhiSGATv2_utils.py

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入配置文件，以便使用其中定义的超参数
import PhiSGATv2_config as config


def _prepare_multitask_labels(labels, predictions):
    labels = labels.to(predictions.device).float()
    if predictions.dim() != 2 or predictions.size(1) != config.NUM_TASKS:
        raise ValueError(
            f"predictions must have shape [batch_size, {config.NUM_TASKS}], "
            f"got {tuple(predictions.shape)}"
        )

    if labels.dim() == 1:
        if config.NUM_TASKS == 1:
            labels = labels.view(-1, 1)
        elif labels.numel() == predictions.size(0) * config.NUM_TASKS:
            labels = labels.view(predictions.size(0), config.NUM_TASKS)
        else:
            raise ValueError(
                f"1D labels cannot be reshaped to [batch_size, {config.NUM_TASKS}], "
                f"got {tuple(labels.shape)}"
            )

    if labels.dim() != 2 or labels.size(0) != predictions.size(0) or labels.size(1) < config.NUM_TASKS:
        raise ValueError(
            f"labels must have shape [batch_size, >= {config.NUM_TASKS}], "
            f"got {tuple(labels.shape)}"
        )
    return labels[:, :config.NUM_TASKS]


def masked_multitask_loss(predictions, labels):
    labels = _prepare_multitask_labels(labels, predictions)
    total_loss = predictions.new_tensor(0.0)
    valid_task_count = 0
    
    for i in range(config.NUM_TASKS):
        # [核心修改 2] 现在 predictions 形状是 (batch_size, num_tasks)，直接切片即可
        task_preds = predictions[:, i]
        task_labels = labels[:, i]
        
        mask = ~torch.isnan(task_labels)
        
        if mask.sum() > 0:
            valid_preds = task_preds[mask]
            valid_labels = task_labels[mask]
            
            # [核心修改 3] 使用 F.mse_loss 计算均方误差
            # 注意：回归任务要求标签必须是 float 类型，不再是 long
            total_loss = total_loss + F.mse_loss(valid_preds, valid_labels)
            valid_task_count += 1
            
    if valid_task_count == 0:
        return predictions.sum() * 0.0
        
    return total_loss / valid_task_count

def train_epoch(model, loader, optimizer, device):
    """
    执行一个完整的训练轮次 (epoch)，加入梯度截断与监控。
    """
    model.train()  
    total_loss = 0
    total_graphs = 0
    skipped_loss_batches = 0
    skipped_grad_batches = 0
    if len(loader.dataset) == 0:
        raise ValueError("Training dataset is empty.")
    
    # [修改 1] 加入 enumerate，以便获取 batch_idx
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad() 
        
        out = model(batch)
        loss = masked_multitask_loss(out, batch.y.to(device))
        
        if not (torch.is_tensor(loss) and torch.isfinite(loss)):
            skipped_loss_batches += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"    [Warning] Batch {batch_idx:04d}/{len(loader)} skipped: "
                "non-finite loss detected."
            )
            continue

        loss.backward() 
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=config.GRAD_CLIP_MAX_NORM
        )

        if not torch.isfinite(grad_norm):
            skipped_grad_batches += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"    [Warning] Batch {batch_idx:04d}/{len(loader)} skipped: "
                "non-finite gradient norm detected."
            )
            continue
        
        # clip_grad_norm_ 返回裁剪前的梯度范数，便于观察是否频繁触发裁剪。
        if batch_idx % 10 == 0:
            print(
                f"    [Batch {batch_idx:04d}/{len(loader)}] "
                f"Loss: {loss.item():.4f} | "
                f"Grad Norm(before clip): {grad_norm:.4f} | "
                f"Clip Max: {config.GRAD_CLIP_MAX_NORM:.2f}"
            )
        
        optimizer.step()  
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    if skipped_loss_batches or skipped_grad_batches:
        print(
            "    [Warning] Skipped batches in this epoch: "
            f"non-finite loss={skipped_loss_batches}, "
            f"non-finite gradients={skipped_grad_batches}"
        )

    if total_graphs == 0:
        raise RuntimeError(
            "No valid training batches were processed. "
            "Check labels, feature scaling, learning rate, and gradient stability."
        )
            
    return total_loss / total_graphs

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_graphs = 0
    y_true = defaultdict(list)
    y_pred = defaultdict(list)

    if len(loader.dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            
            labels = _prepare_multitask_labels(batch.y, out)
            loss = masked_multitask_loss(out, labels)
            if torch.is_tensor(loss) and torch.isfinite(loss):
                total_loss += loss.item() * batch.num_graphs
                total_graphs += batch.num_graphs
            
            # [核心修改 4] 移除 argmax 操作，直接按任务收集模型预测的连续浮点数
            for i in range(config.NUM_TASKS):
                task_labels = labels[:, i]
                task_preds = out[:, i]
                
                mask = ~torch.isnan(task_labels)
                if mask.sum() > 0:
                    y_true[i].extend(task_labels[mask].cpu().numpy().tolist())
                    y_pred[i].extend(task_preds[mask].cpu().numpy().tolist())

    # [核心修改 5] 计算回归评价指标 RMSE 和 MAE
    metrics = {}
    for i in range(config.NUM_TASKS):
        if len(y_true[i]) > 0:
            mse = mean_squared_error(y_true[i], y_pred[i])
            mae = mean_absolute_error(y_true[i], y_pred[i])
            rmse = np.sqrt(mse)
            metrics[i] = {'RMSE': rmse, 'MAE': mae}
        else:
            metrics[i] = {'RMSE': float('nan'), 'MAE': float('nan')}
            
    avg_rmse = np.nanmean([m['RMSE'] for m in metrics.values()])
    avg_mae = np.nanmean([m['MAE'] for m in metrics.values()])
    
    # 返回损失，任务级指标字典，以及全局平均指标
    return total_loss / max(total_graphs, 1), metrics, avg_rmse, avg_mae
