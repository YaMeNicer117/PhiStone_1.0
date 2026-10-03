import os
import sys
import contextlib 
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

# Local Imports
import PhiSSE3TD_config as config
from PhiSSE3TD_dataset import get_train_val_dataloaders
from PhiSSE3TD_model import E3NNTransformerDiffusion
from PhiSSE3TD_diffusion import DiffusionProcess
from PhiSSE3TD_utils_geom import (
    compute_edge_topology_loss,
    compute_physical_constraint_loss,
    compute_weighted_feat_loss,
    compute_sparse_distance_loss  
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.autograd.set_detect_anomaly(False)

# =============================================================================
# 0. DDP 辅助函数
# =============================================================================

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# =============================================================================
# 1. 导入外部指导模型 (PhiSGATv2)
# =============================================================================
if is_main_process():
    print(f"正在导入外部指导模型 PhiSGATv2，路径: {config.PHISGAT_ROOT}")

if config.PHISGAT_ROOT not in sys.path:
    sys.path.append(config.PHISGAT_ROOT)

try:
    from PhiSGATv2_model import GATv2Model
    if is_main_process():
        print("成功导入 PhiSGATv2 模块。")
except ImportError as e:
    print(f"[严重错误] 无法导入 PhiSGATv2。请检查 config.PHISGAT_ROOT 是否正确指向 GATv2 项目根目录。")
    raise e

# =============================================================================
# 2. 辅助函数：重构预测器所需的 Batch
# =============================================================================

def build_predictor_batch(x0_pred_feat, ligand_batch_indices, filtered_edge_index, edge_logits, is_ligand_mask, num_nodes, num_graphs):
    """
    [重构极速版] 构建活性预测器 Batch 
    - 纯 GPU 连通分量计算，0 CPU 同步。
    - 空图自动注入占位节点，完美对齐 Batch Size 防崩溃。
    """
    device = x0_pred_feat.device
    data_list = []
    
    pred_classes = torch.argmax(edge_logits, dim=-1)
    
    valid_edge_mask = (pred_classes == 0) | (pred_classes == 3)
    
    src, dst = filtered_edge_index
    ll_mask = is_ligand_mask[src] & is_ligand_mask[dst]
    final_edge_mask = valid_edge_mask & ll_mask
    
    final_filtered_edges = filtered_edge_index[:, final_edge_mask]
    filtered_edge_classes = pred_classes[final_edge_mask]
    
    attr_cov = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float)
    attr_glob = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float)
    
    global_ligand_idx = torch.nonzero(is_ligand_mask).squeeze(1)
    
    for i in range(num_graphs):
        local_ligand_mask = (ligand_batch_indices == i)
        current_global_nodes = global_ligand_idx[local_ligand_mask]
        N_i = len(current_global_nodes)
        
        if N_i == 0:
            dummy_x = torch.zeros((1, 11), device=device)
            dummy_embed = torch.zeros((1, config.EMBEDDING_DIM_IN), device=device)
            dummy_edge = torch.empty((2, 0), dtype=torch.long, device=device)
            dummy_attr = torch.empty((0, 3), device=device)
            data_list.append(Data(x=dummy_x, frag_embeds=dummy_embed, edge_index=dummy_edge, edge_attr=dummy_attr))
            continue
            
        edge_mask_i = torch.isin(final_filtered_edges[0], current_global_nodes) & \
                      torch.isin(final_filtered_edges[1], current_global_nodes)
        edges_i = final_filtered_edges[:, edge_mask_i]
        classes_i = filtered_edge_classes[edge_mask_i]
        
        mapping_tensor = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        mapping_tensor[current_global_nodes] = torch.arange(N_i, device=device)
        local_edges_i = mapping_tensor[edges_i]
        
        if local_edges_i.shape[1] > 0:
            adj = torch.eye(N_i, dtype=torch.float, device=device)
            adj[local_edges_i[0], local_edges_i[1]] = 1.0
            adj[local_edges_i[1], local_edges_i[0]] = 1.0
            
            num_squarings = N_i.bit_length()
            for _ in range(num_squarings):
                adj = torch.matmul(adj, adj)
                adj = (adj > 0).float() 
                
            comp_sizes = adj.sum(dim=1)
            largest_comp_node = torch.argmax(comp_sizes)
            keep_node_mask = (adj[largest_comp_node] > 0)
        else:
            keep_node_mask = torch.zeros(N_i, device=device, dtype=torch.bool)
            keep_node_mask[0] = True 
            
        kept_local_nodes = torch.nonzero(keep_node_mask).squeeze(1)
        N_kept = len(kept_local_nodes)
        
        feat_i = x0_pred_feat[local_ligand_mask][keep_node_mask]
        embeds_i = feat_i[:, :config.EMBEDDING_DIM_IN]
        chem_props_sliced = feat_i[:, config.EMBEDDING_DIM_IN:config.EMBEDDING_DIM_IN+11] 
        
        keep_edge_mask = keep_node_mask[local_edges_i[0]] & keep_node_mask[local_edges_i[1]]
        final_local_edges = local_edges_i[:, keep_edge_mask]
        final_classes = classes_i[keep_edge_mask]
        
        mapping_tensor_2 = torch.full((N_i,), -1, dtype=torch.long, device=device)
        mapping_tensor_2[kept_local_nodes] = torch.arange(N_kept, device=device)
        final_edges_reindexed = mapping_tensor_2[final_local_edges]
        
        edge_attr_i = torch.zeros((final_edges_reindexed.shape[1], 3), device=device, dtype=torch.float)
        edge_attr_i[final_classes == 0] = attr_cov
        edge_attr_i[final_classes == 3] = attr_glob
        
        data = Data(
            x=chem_props_sliced,     
            frag_embeds=embeds_i,      
            edge_index=final_edges_reindexed,   
            edge_attr=edge_attr_i      
        )
        data_list.append(data)
        
    return Batch.from_data_list(data_list).to(device) if data_list else None

# =============================================================================
# 3. 微调 epoch 函数
# =============================================================================
def finetune_epoch(model, loader, diffusion, phisgat_model, optimizer, scaler, epoch_num, device, sampler=None, accumulation_steps=1, amp_dtype=torch.bfloat16):
    if sampler is not None:
        sampler.set_epoch(epoch_num)

    model.train()
    phisgat_model.eval() 
    
    optimizer.zero_grad(set_to_none=True)
    
    metrics = {
        'total': 0.0, 'diff': 0.0, 'act': 0.0, 'var': 0.0, 'phy': 0.0, 'edge': 0.0,
        'feat': 0.0, 'pos': 0.0, 'frame': 0.0, 'dist_ll': 0.0, 'dist_lp': 0.0, 'feat_small': 0.0, 'feat_large': 0.0,
        'phy_clash_raw': 0.0
    }
    
    epoch_max_raw_grad = 0.0
    loss_fn_diff = nn.SmoothL1Loss(reduction='mean', beta=config.SMOOTH_L1_BETA)
    
    activity_threshold = diffusion.num_timesteps * config.ACTIVITY_GUIDANCE_RATIO
    physics_threshold = diffusion.num_timesteps * config.PHYSICS_GUIDANCE_RATIO

    if is_main_process():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{config.FINETUNE_EPOCHS} [Finetuning]", leave=False, mininterval=30.0)
    else:
        progress_bar = loader
    
    num_batches = len(loader)

    for i, batch in enumerate(progress_bar):
        batch = batch.to(device)
        is_update_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == num_batches)
        sync_context = model.no_sync if (not is_update_step) and isinstance(model, DDP) else contextlib.nullcontext

        with sync_context():
            is_ligand_mask = batch.is_ligand
            x0_feat = batch.x[is_ligand_mask, :config.LIGAND_FEATURE_DIM]
            x0_pos = batch.pos[is_ligand_mask]
            x0_ref_coords = batch.ref_coords[is_ligand_mask]
            ligand_batch_indices = batch.batch[is_ligand_mask]
            ligand_hac = batch.hac[is_ligand_mask] if hasattr(batch, 'hac') else None

            # =================================================================
            # 🚀 [核心机制 0]：提取蛋白掩码与特征 (修复未定义 Bug)
            # =================================================================
            is_dummy_mask = getattr(batch, 'is_dummy', torch.zeros_like(batch.is_ligand))
            protein_mask_global = (~batch.is_ligand) & (~is_dummy_mask)
            protein_batch_indices = batch.batch[protein_mask_global]
            sub_prot_pos = batch.pos[protein_mask_global]
            sub_prot_ref = batch.ref_coords[protein_mask_global]

            # =================================================================
            # 🚀 [核心机制 1]：掩码生成 (兼容分轨训练)
            # =================================================================
            num_lig_nodes = x0_pos.shape[0]
            is_fixed_ligand = torch.zeros(num_lig_nodes, dtype=torch.bool, device=device)
            
            for b_idx in torch.unique(ligand_batch_indices):
                graph_node_idx = torch.nonzero(ligand_batch_indices == b_idx).squeeze(-1)
                n_nodes = graph_node_idx.numel()
                
                if n_nodes > 1:
                    # 如果你拆分了微调脚本：在 Creative 版这里写 1.0 (纯0锚点)；在 Survival 版这里写 0.0 (纯锚点)。
                    # 当前保持 0.2 以支持混合双修
                    if torch.rand(1).item() < 0.2:
                        num_fixed = 0
                    else:
                        keep_ratio = torch.empty(1).uniform_(0.1, 0.7).item()
                        num_fixed = int(n_nodes * keep_ratio)
                        num_fixed = max(1, min(num_fixed, n_nodes - 1))
                    
                    if num_fixed > 0:
                        perm = torch.randperm(n_nodes, device=device)
                        fixed_idx = graph_node_idx[perm[:num_fixed]]
                        is_fixed_ligand[fixed_idx] = True
                        
            is_free_ligand = ~is_fixed_ligand

            t = torch.randint(0, diffusion.num_timesteps, (batch.num_graphs,), device=device)
            (xt_feat, xt_pos, xt_ref_coords), (true_noise_feat, true_noise_pos, true_noise_frame) = diffusion.q_sample(
                x0_feat, x0_pos, x0_ref_coords, t, ligand_batch_indices
            )
            
            # =================================================================
            # 🚀 [核心机制 2]：Clean Context Conditioning (洗白固定锚点)
            # =================================================================
            xt_feat[is_fixed_ligand] = x0_feat[is_fixed_ligand]
            xt_pos[is_fixed_ligand] = x0_pos[is_fixed_ligand]
            xt_ref_coords[is_fixed_ligand] = x0_ref_coords[is_fixed_ligand]
            
            noisy_batch = batch.clone()
            noisy_batch.x[is_ligand_mask, :config.LIGAND_FEATURE_DIM] = xt_feat
            noisy_batch.pos[is_ligand_mask] = xt_pos
            noisy_batch.ref_coords[is_ligand_mask] = xt_ref_coords

            noisy_batch_dynamic_edges = diffusion.rebuild_graph_with_dynamic_edges(noisy_batch)

            with autocast(device_type='cuda', enabled=True, dtype=amp_dtype):
                # --- A. 扩散特征预测 ---
                pred_noise_feat, pred_noise_pos, pred_noise_frame, edge_logits, filtered_edge_index = model(noisy_batch_dynamic_edges, t)

                # =================================================================
                # 🚀 [核心机制 3]：仅对自由节点计算扩散损失
                # =================================================================
                if is_free_ligand.any():
                    free_pred_noise_feat = pred_noise_feat[is_free_ligand]
                    free_true_noise_feat = true_noise_feat[is_free_ligand]
                    free_ligand_hac = ligand_hac[is_free_ligand] if ligand_hac is not None else None
                    
                    loss_diff_feat, loss_small_pure, loss_large_pure = compute_weighted_feat_loss(
                        free_pred_noise_feat, free_true_noise_feat, free_ligand_hac, beta=config.SMOOTH_L1_BETA
                    )
                    loss_diff_pos = loss_fn_diff(pred_noise_pos[is_free_ligand], true_noise_pos[is_free_ligand])
                    loss_diff_frame = loss_fn_diff(pred_noise_frame[is_free_ligand], true_noise_frame[is_free_ligand])
                else:
                    loss_diff_feat, loss_small_pure, loss_large_pure = torch.tensor(0.0, device=device), 0.0, 0.0
                    loss_diff_pos, loss_diff_frame = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

                # --- 提取预测无噪结构 ---
                x0_pred_feat, x0_pred_pos, x0_pred_ref = diffusion.predict_x0_from_noise(
                    (xt_feat, xt_pos, xt_ref_coords),
                    (pred_noise_feat, pred_noise_pos, pred_noise_frame),
                    t, ligand_batch_indices 
                )

                # =================================================================
                # 🚀 [核心机制 4]：修正物理损失与活性预测器的输入
                # =================================================================
                x0_pred_pos_corrected = x0_pred_pos.clone()
                x0_pred_pos_corrected[is_fixed_ligand] = x0_pos[is_fixed_ligand]
                
                x0_pred_ref_corrected = x0_pred_ref.clone()
                x0_pred_ref_corrected[is_fixed_ligand] = x0_ref_coords[is_fixed_ligand]

                # 必须将特征也洗白，否则外部的 PhiSGAT 看到的是垃圾特征，打分会错乱！
                x0_pred_feat_corrected = x0_pred_feat.clone()
                x0_pred_feat_corrected[is_fixed_ligand] = x0_feat[is_fixed_ligand]
                
                # 距离损失计算
                loss_dist_ll, loss_dist_lp = compute_sparse_distance_loss(
                    pos_pred=x0_pred_pos_corrected, 
                    pos_true=x0_pos, 
                    batch_ligand=ligand_batch_indices,
                    protein_pos=sub_prot_pos,
                    protein_batch=protein_batch_indices,
                    ll_radius=config.DIST_LL_MAX_RADIUS, 
                    lp_radius=config.DIST_LP_MAX_RADIUS, 
                    beta=config.SMOOTH_L1_BETA
                )
                
                loss_diffusion = (config.W_FEAT_LOSS * loss_diff_feat + 
                                  config.W_POS_LOSS * loss_diff_pos + 
                                  config.W_FRAME_LOSS * loss_diff_frame + 
                                  config.W_DIST_LL_LOSS * loss_dist_ll +
                                  config.W_DIST_LP_LOSS * loss_dist_lp)

                # --- 拓扑边预测损失 (屏蔽固定边) ---
                is_fixed_global = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
                is_fixed_global[is_ligand_mask] = is_fixed_ligand
                u, v = filtered_edge_index
                target_edge_mask = ~(is_fixed_global[u] & is_fixed_global[v])

                loss_edge = compute_edge_topology_loss(
                    filtered_edge_index=filtered_edge_index[:, target_edge_mask], 
                    edge_logits=edge_logits[target_edge_mask], 
                    gt_edge_index=batch.gt_edge_index, 
                    gt_edge_attr=batch.gt_edge_attr, 
                    num_nodes=batch.num_nodes,
                    gamma=config.FOCAL_LOSS_GAMMA
                )

                # --- B. 活性与多样性损失 ---
                # 传入修正后的特征
                predictor_batch = build_predictor_batch(
                    x0_pred_feat_corrected, ligand_batch_indices, 
                    filtered_edge_index, edge_logits, 
                    batch.is_ligand, batch.num_nodes,
                    batch.num_graphs
                )
                
                if predictor_batch is not None:
                    predicted_levels = phisgat_model(predictor_batch)
                    num_tasks = predicted_levels.shape[1]
                    target_sum = num_tasks * 4.2
                    sum_of_levels = predicted_levels.sum(dim=1)
                    
                    loss_act_per_graph = torch.nn.functional.relu(target_sum - sum_of_levels)
                    variance_per_graph = predicted_levels.var(dim=1, unbiased=False)
                    
                    graph_mask_act = (t < activity_threshold).float()
                    valid_act_graphs = torch.clamp(graph_mask_act.sum(), min=1.0)
                    
                    loss_activity = (loss_act_per_graph * graph_mask_act).sum() / valid_act_graphs
                    loss_variance = (variance_per_graph * graph_mask_act).sum() / valid_act_graphs
                else:
                    loss_activity = torch.tensor(0.0, device=device)
                    loss_variance = torch.tensor(0.0, device=device)

                # --- C. 物理约束损失 ---
                t_nodes_float = t[ligand_batch_indices].float()
                physics_weight_nodes = torch.clamp(1.0 - (t_nodes_float / physics_threshold), min=0.0, max=1.0)
                physics_node_mask = physics_weight_nodes > 0.0
                
                l_clash_ll, l_clash_lp, _ = compute_physical_constraint_loss(
                    x0_pred_pos_corrected, x0_pred_ref_corrected,
                    sub_prot_pos, sub_prot_ref,
                    ligand_batch_indices, protein_batch_indices,
                    mask_physics=physics_node_mask,
                    physics_weight_nodes=physics_weight_nodes,
                    hac_tensor=ligand_hac
                )
                
                loss_physics = config.W_PHYSICS_CLASH * (l_clash_ll + l_clash_lp)
                
                # --- D. 汇总加权总损失 ---
                W_EDGE_LOSS = getattr(config, 'W_EDGE_LOSS', 1.0)
                
                total_current_loss = (config.W_DIFFUSION * (loss_diffusion + loss_physics + W_EDGE_LOSS * loss_edge) + 
                                      config.W_ACTIVITY * loss_activity + 
                                      config.W_VARIANCE * loss_variance)
                
                if (i + 1) == num_batches:
                    remainder = num_batches % accumulation_steps
                    current_accum_steps = remainder if remainder != 0 else accumulation_steps
                else:
                    current_accum_steps = accumulation_steps
                    
                scaled_loss = total_current_loss / current_accum_steps
                
            scaler.scale(scaled_loss).backward()

        metrics['total'] += total_current_loss.item()
        metrics['diff'] += loss_diffusion.item()
        metrics['edge'] += loss_edge.item()
        metrics['act'] += loss_activity.item()
        metrics['var'] += loss_variance.item()
        metrics['phy'] += loss_physics.item()
        metrics['phy_clash_raw'] += (l_clash_ll + l_clash_lp).item()
        metrics['feat'] += loss_diff_feat.item()
        metrics['pos'] += loss_diff_pos.item()
        metrics['frame'] += loss_diff_frame.item()
        metrics['dist_ll'] += loss_dist_ll.item()
        metrics['dist_lp'] += loss_dist_lp.item()
        metrics['feat_small'] += loss_small_pure  
        metrics['feat_large'] += loss_large_pure

        if is_update_step:
            scaler.unscale_(optimizer)
            
            hold_epochs = getattr(config, 'FINETUNE_HOLD_EPOCHS', 0)
            if epoch_num < (config.FINETUNE_WARMUP_EPOCHS + hold_epochs):
                current_max_norm = config.FINETUNE_GRAD_CLIP_WARMUP_NORM
            else:
                current_max_norm = config.FINETUNE_GRAD_CLIP_STABLE_NORM
                
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=current_max_norm
            )
            
            current_raw_norm_val = raw_grad_norm.item() if isinstance(raw_grad_norm, torch.Tensor) else raw_grad_norm

            if not math.isfinite(current_raw_norm_val):
                if is_main_process():
                    progress_bar.write(
                        f"[警告] 检测到非有限梯度 Norm: {current_raw_norm_val}，"
                        "已清空梯度并跳过本次优化步骤。"
                    )
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                continue

            epoch_max_raw_grad = max(epoch_max_raw_grad, current_raw_norm_val)
            
            if current_raw_norm_val > current_max_norm and is_main_process():
                progress_bar.write(f"[警告] 截断前梯度 Norm: {current_raw_norm_val:.2e} (超出阈值 {current_max_norm:.1f}，已截断)")
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
    n = len(loader)
    local_avgs = {k: v/n for k, v in metrics.items()}
    
    if dist.is_initialized():
        keys = sorted(local_avgs.keys())
        vals = torch.tensor([local_avgs[k] for k in keys], device=device)
        vals = reduce_mean(vals, dist.get_world_size())
        res_dict = {k: v.item() for k, v in zip(keys, vals)}
    else:
        res_dict = local_avgs
        
    return res_dict, epoch_max_raw_grad


@torch.no_grad()
def validate_finetune_epoch(model, loader, diffusion, phisgat_model, loss_fn_diff, epoch_num, device, amp_dtype=torch.bfloat16):
    model.eval()
    phisgat_model.eval()
    
    metrics = {
        'total': 0.0, 'diff': 0.0, 'act': 0.0, 'var': 0.0, 'edge': 0.0, 'phy': 0.0,
        'feat': 0.0, 'feat_small': 0.0, 'feat_large': 0.0, 'pos': 0.0, 'frame': 0.0, 'dist_ll': 0.0, 'dist_lp': 0.0,
        'phy_clash_raw': 0.0
    }
    
    activity_threshold = diffusion.num_timesteps * config.ACTIVITY_GUIDANCE_RATIO
    physics_threshold = diffusion.num_timesteps * config.PHYSICS_GUIDANCE_RATIO
    W_EDGE_LOSS = getattr(config, 'W_EDGE_LOSS', 1.0)
    
    if is_main_process():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{config.FINETUNE_EPOCHS} [Validation]", leave=False)
    else:
        progress_bar = loader

    for batch in progress_bar:
        batch = batch.to(device)
        is_ligand_mask = batch.is_ligand
        x0_feat = batch.x[is_ligand_mask, :config.LIGAND_FEATURE_DIM]
        x0_pos = batch.pos[is_ligand_mask]
        x0_ref_coords = batch.ref_coords[is_ligand_mask]
        ligand_batch_indices = batch.batch[is_ligand_mask]
        ligand_hac = batch.hac[is_ligand_mask] if hasattr(batch, 'hac') else None

        # 提取蛋白掩码
        is_dummy_mask = getattr(batch, 'is_dummy', torch.zeros_like(batch.is_ligand))
        protein_mask_global = (~batch.is_ligand) & (~is_dummy_mask)
        protein_batch_indices = batch.batch[protein_mask_global]
        sub_prot_pos = batch.pos[protein_mask_global]
        sub_prot_ref = batch.ref_coords[protein_mask_global]

        # 掩码生成
        num_lig_nodes = x0_pos.shape[0]
        is_fixed_ligand = torch.zeros(num_lig_nodes, dtype=torch.bool, device=device)
        for b_idx in torch.unique(ligand_batch_indices):
            graph_node_idx = torch.nonzero(ligand_batch_indices == b_idx).squeeze(-1)
            n_nodes = graph_node_idx.numel()
            if n_nodes > 1:
                if torch.rand(1).item() < 0.2:
                    num_fixed = 0
                else:
                    keep_ratio = torch.empty(1).uniform_(0.1, 0.7).item()
                    num_fixed = min(max(1, int(n_nodes * keep_ratio)), n_nodes - 1)
                
                if num_fixed > 0:
                    perm = torch.randperm(n_nodes, device=device)
                    is_fixed_ligand[graph_node_idx[perm[:num_fixed]]] = True
                
        is_free_ligand = ~is_fixed_ligand

        t = torch.randint(0, diffusion.num_timesteps, (batch.num_graphs,), device=device)
        (xt_feat, xt_pos, xt_ref_coords), (true_noise_feat, true_noise_pos, true_noise_frame) = diffusion.q_sample(
            x0_feat, x0_pos, x0_ref_coords, t, ligand_batch_indices
        )
        
        xt_feat[is_fixed_ligand] = x0_feat[is_fixed_ligand]
        xt_pos[is_fixed_ligand] = x0_pos[is_fixed_ligand]
        xt_ref_coords[is_fixed_ligand] = x0_ref_coords[is_fixed_ligand]
        
        noisy_batch = batch.clone()
        noisy_batch.x[is_ligand_mask, :config.LIGAND_FEATURE_DIM] = xt_feat
        noisy_batch.pos[is_ligand_mask] = xt_pos
        noisy_batch.ref_coords[is_ligand_mask] = xt_ref_coords
        noisy_batch_dynamic_edges = diffusion.rebuild_graph_with_dynamic_edges(noisy_batch)

        with autocast(device_type='cuda', enabled=True, dtype=amp_dtype):
            pred_noise_feat, pred_noise_pos, pred_noise_frame, edge_logits, filtered_edge_index = model(noisy_batch_dynamic_edges, t)

            if is_free_ligand.any():
                l_feat, loss_small_pure, loss_large_pure = compute_weighted_feat_loss(
                    pred_noise_feat[is_free_ligand], true_noise_feat[is_free_ligand], ligand_hac[is_free_ligand] if ligand_hac is not None else None, beta=config.SMOOTH_L1_BETA
                )
                l_pos = loss_fn_diff(pred_noise_pos[is_free_ligand], true_noise_pos[is_free_ligand])
                l_frame = loss_fn_diff(pred_noise_frame[is_free_ligand], true_noise_frame[is_free_ligand])
            else:
                l_feat, loss_small_pure, loss_large_pure = torch.tensor(0.0, device=device), 0.0, 0.0
                l_pos, l_frame = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            x0_pred_feat, x0_pred_pos, x0_pred_ref = diffusion.predict_x0_from_noise(
                (xt_feat, xt_pos, xt_ref_coords),
                (pred_noise_feat, pred_noise_pos, pred_noise_frame),
                t, ligand_batch_indices 
            )

            # 修正输入
            x0_pred_pos_corrected = x0_pred_pos.clone()
            x0_pred_pos_corrected[is_fixed_ligand] = x0_pos[is_fixed_ligand]
            
            x0_pred_ref_corrected = x0_pred_ref.clone()
            x0_pred_ref_corrected[is_fixed_ligand] = x0_ref_coords[is_fixed_ligand]

            x0_pred_feat_corrected = x0_pred_feat.clone()
            x0_pred_feat_corrected[is_fixed_ligand] = x0_feat[is_fixed_ligand]

            l_dist_ll, l_dist_lp = compute_sparse_distance_loss(
                pos_pred=x0_pred_pos_corrected, pos_true=x0_pos, batch_ligand=ligand_batch_indices,
                protein_pos=sub_prot_pos, protein_batch=protein_batch_indices,
                ll_radius=config.DIST_LL_MAX_RADIUS, lp_radius=config.DIST_LP_MAX_RADIUS, beta=config.SMOOTH_L1_BETA
            )
            
            loss_diffusion = (config.W_FEAT_LOSS * l_feat + config.W_POS_LOSS * l_pos + config.W_FRAME_LOSS * l_frame + config.W_DIST_LL_LOSS * l_dist_ll + config.W_DIST_LP_LOSS * l_dist_lp)
            
            is_fixed_global = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
            is_fixed_global[is_ligand_mask] = is_fixed_ligand
            u, v = filtered_edge_index
            target_edge_mask = ~(is_fixed_global[u] & is_fixed_global[v])

            loss_edge = compute_edge_topology_loss(
                filtered_edge_index=filtered_edge_index[:, target_edge_mask], edge_logits=edge_logits[target_edge_mask], 
                gt_edge_index=batch.gt_edge_index, gt_edge_attr=batch.gt_edge_attr, 
                num_nodes=batch.num_nodes, gamma=config.FOCAL_LOSS_GAMMA
            )

            predictor_batch = build_predictor_batch(
                x0_pred_feat_corrected, ligand_batch_indices, filtered_edge_index, edge_logits, 
                batch.is_ligand, batch.num_nodes, batch.num_graphs
            )
            
            if predictor_batch is not None:
                predicted_levels = phisgat_model(predictor_batch)
                target_sum = predicted_levels.shape[1] * 4.2
                sum_of_levels = predicted_levels.sum(dim=1)
                
                loss_act_per_graph = torch.nn.functional.relu(target_sum - sum_of_levels)
                variance_per_graph = predicted_levels.var(dim=1, unbiased=False)
                
                graph_mask_act = (t < activity_threshold).float()
                valid_act_graphs = torch.clamp(graph_mask_act.sum(), min=1.0)
                
                loss_activity = (loss_act_per_graph * graph_mask_act).sum() / valid_act_graphs
                loss_variance = (variance_per_graph * graph_mask_act).sum() / valid_act_graphs
            else:
                loss_activity, loss_variance = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            t_nodes_float = t[ligand_batch_indices].float()
            physics_weight_nodes = torch.clamp(1.0 - (t_nodes_float / physics_threshold), min=0.0, max=1.0)
            
            l_clash_ll, l_clash_lp, _ = compute_physical_constraint_loss(
                x0_pred_pos_corrected, x0_pred_ref_corrected,
                sub_prot_pos, sub_prot_ref,
                ligand_batch_indices, protein_batch_indices,
                mask_physics=(physics_weight_nodes > 0.0),
                physics_weight_nodes=physics_weight_nodes,
                hac_tensor=ligand_hac
            )
            
            loss_physics = config.W_PHYSICS_CLASH * (l_clash_ll + l_clash_lp)

            total_current_loss = (config.W_DIFFUSION * (loss_diffusion + loss_physics + W_EDGE_LOSS * loss_edge) + config.W_ACTIVITY * loss_activity + config.W_VARIANCE * loss_variance)

        metrics['total'] += total_current_loss.item()
        metrics['diff'] += loss_diffusion.item()
        metrics['edge'] += loss_edge.item()
        metrics['phy'] += loss_physics.item()
        metrics['phy_clash_raw'] += (l_clash_ll + l_clash_lp).item()
        metrics['act'] += loss_activity.item()
        metrics['var'] += loss_variance.item()
        metrics['feat'] += l_feat.item()
        metrics['pos'] += l_pos.item()
        metrics['frame'] += l_frame.item()
        metrics['dist_ll'] += l_dist_ll.item() 
        metrics['dist_lp'] += l_dist_lp.item() 
        metrics['feat_small'] += loss_small_pure   
        metrics['feat_large'] += loss_large_pure

    n = len(loader)
    local_avgs = {k: v/n for k, v in metrics.items()}
    if dist.is_initialized():
        keys = sorted(local_avgs.keys())
        vals = torch.tensor([local_avgs[k] for k in keys], device=device)
        vals = reduce_mean(vals, dist.get_world_size())
        res_dict = {k: v.item() for k, v in zip(keys, vals)}
    else:
        res_dict = local_avgs
        
    return res_dict

# =============================================================================
# 4. 主执行流程
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # DDP 初始化
    # -------------------------------------------------------------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    device = torch.device(f"cuda:{local_rank}")
    config.DEVICE = device

    if is_main_process():
        print("\n" + "="*60)
        print("--- 开始第二阶段：基于活性的微调 (Fine-tuning) [DDP Mode] ---")
        print("="*60)
        
        world_size = dist.get_world_size()
        accum_steps = config.GRAD_ACCUMULATION_STEPS
        effective_batch_size = config.BATCH_SIZE * accum_steps * world_size
        
        print(f"World Size (GPUs)        : {world_size}")
        print(f"Per-GPU Batch Size       : {config.BATCH_SIZE}")
        print(f"Gradient Accumulation    : {accum_steps} steps")
        print(f"Effective Batch Size     : {effective_batch_size}")
        print("-" * 60)
        
        os.makedirs(config.CHECKPOINT_DIR_FINETUNE, exist_ok=True)
        os.makedirs(config.LOG_DIR_FINETUNE_SA, exist_ok=True)
        print(f"Checkpoints: {config.CHECKPOINT_DIR_FINETUNE}")
        print(f"Logs: {config.LOG_DIR_FINETUNE_SA}")
        writer = SummaryWriter(config.LOG_DIR_FINETUNE_SA)
    else:
        writer = None

    train_loader, val_loader, train_sampler = get_train_val_dataloaders(distributed=True)
    best_finetune_val_loss = float('inf')
    
    # --- A. 加载模型与权重 ---
    e3nn_model_finetune = E3NNTransformerDiffusion().to(device)
    e3nn_model_finetune = DDP(e3nn_model_finetune, device_ids=[local_rank], find_unused_parameters=False)
    
    if is_main_process():
        print(f"E3NN 模型实例化并 DDP 包装成功。")
    
    best_finetune_train_loss = float('inf')
    start_epoch = 0

    loaded_checkpoint = None

    if os.path.exists(config.BEST_MODEL_PATH_FINETUNE_SA):
        if is_main_process():
            print(f"  -> 发现微调断点，正在加载微调权重以续训: '{config.BEST_MODEL_PATH_FINETUNE_SA}'")
        try:
            checkpoint = torch.load(config.BEST_MODEL_PATH_FINETUNE_SA, map_location=device, weights_only=False)
            loaded_checkpoint = checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                e3nn_model_finetune.module.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_finetune_train_loss = checkpoint.get('best_loss', float('inf'))
            else:
                e3nn_model_finetune.module.load_state_dict(checkpoint)
            if is_main_process():
                print(f"  -> 微调断点加载成功！将从 Epoch {start_epoch} 继续优化。")
        except Exception as e:
            if is_main_process():
                print(f"  -> [警告] 微调权重加载失败: {e}。程序退出。")
            dist.destroy_process_group()
            sys.exit(1)
            
    elif os.path.exists(config.BEST_MODEL_PATH_PRETRAIN):
        if is_main_process():
            print(f"  -> 从头开始微调，正在加载预训练基座权重: '{config.BEST_MODEL_PATH_PRETRAIN}'")
        try:
            checkpoint = torch.load(config.BEST_MODEL_PATH_PRETRAIN, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                e3nn_model_finetune.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                e3nn_model_finetune.module.load_state_dict(checkpoint)
                
            if is_main_process():
                print("  -> 预训练基座加载成功。")
        except Exception as e:
            if is_main_process():
                print(f"  -> [致命错误] 预训练基座加载失败: {e}\n     微调必须基于预训练模型。程序退出。")
            dist.destroy_process_group()
            sys.exit(1)
    else:
        if is_main_process():
            print(f"  -> [致命错误] 找不到任何权重文件。请先运行预训练。")
        dist.destroy_process_group()
        sys.exit(1)

    # --- B. 加载指导模型 ---
    phisgat_weight_path = os.path.join(config.PHISGAT_DATA_DIR, 'best_model.pth')
    if not os.path.exists(phisgat_weight_path):
        phisgat_weight_path = os.path.join(config.PHISGAT_DATA_DIR, 'best_model.pt')
        
    if is_main_process():
        print(f"  -> 正在加载 PhiSGATv2 权重: '{phisgat_weight_path}'")
    
    try:
        phisgatv2_model = GATv2Model().to(device)
        state_dict = torch.load(phisgat_weight_path, map_location=device, weights_only=True)
        phisgatv2_model.load_state_dict(state_dict)
        
        phisgatv2_model.eval()
        for param in phisgatv2_model.parameters():
            param.requires_grad = False
            
        if is_main_process():
            print(f"  -> PhiSGATv2 权重加载成功并已冻结。")
            
    except (FileNotFoundError, ImportError, Exception) as e:
        if is_main_process():
            print(f"  -> [致命错误] PhiSGATv2 模型加载失败: {e}")
        dist.destroy_process_group()
        sys.exit(1)

    # --- C. 初始化组件与调度器 ---
    diffusion_controller = DiffusionProcess(model=e3nn_model_finetune, device=device)
    optimizer = optim.AdamW(e3nn_model_finetune.parameters(), lr=config.FINETUNE_LR, weight_decay=config.FINETUNE_WEIGHT_DECAY)
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    SUPPORTS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    AMP_DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16
    
    scaler = GradScaler(enabled=(AMP_DTYPE == torch.float16))
    
    if is_main_process():
        print(f"  -> AMP 精度模式: {AMP_DTYPE}")
        print(f"  -> GradScaler 状态: {'开启 (防下溢)' if scaler.is_enabled() else '关闭 (无冗余)'}")

    def get_finetune_lr_multiplier(current_epoch):
        warmup_epochs = config.FINETUNE_WARMUP_EPOCHS
        hold_epochs = getattr(config, 'FINETUNE_HOLD_EPOCHS', 0)
        total_epochs = config.FINETUNE_EPOCHS
        
        if current_epoch < warmup_epochs:
            return config.FINETUNE_WARMUP_START_LR_RATIO + (1.0 - config.FINETUNE_WARMUP_START_LR_RATIO) * (current_epoch / warmup_epochs)
        elif current_epoch < warmup_epochs + hold_epochs:
            return 1.0
        else:
            decay_epochs = total_epochs - warmup_epochs - hold_epochs
            if decay_epochs <= 0:
                return 1.0  
                
            current_decay_epoch = current_epoch - warmup_epochs - hold_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * current_decay_epoch / decay_epochs))
            min_lr_ratio = 0.01  
            
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=get_finetune_lr_multiplier, last_epoch=start_epoch - 1)

    if loaded_checkpoint is not None and isinstance(loaded_checkpoint, dict):
        if 'optimizer_state_dict' in loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            if is_main_process():
                print("  -> 已成功恢复 AdamW 优化器动量状态。")
        if 'scheduler_state_dict' in loaded_checkpoint:
            scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])
            if is_main_process():
                print("  -> 已成功恢复 LR Scheduler 状态。")

    # 4. 微调循环
    if is_main_process():
        print("\n--- 开始微调循环 ---")
    
    for epoch in range(start_epoch, config.FINETUNE_EPOCHS):
        
        # === 1. 执行训练 Epoch ===
        metrics, epoch_max_grad = finetune_epoch(
            model=e3nn_model_finetune, 
            loader=train_loader, 
            diffusion=diffusion_controller,
            phisgat_model=phisgatv2_model, 
            optimizer=optimizer, 
            scaler=scaler, 
            epoch_num=epoch, 
            device=device, 
            sampler=train_sampler,
            accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
            amp_dtype=AMP_DTYPE 
        )
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # === 2. 执行验证 Epoch ===
        val_metrics = validate_finetune_epoch(
            model=e3nn_model_finetune, 
            loader=val_loader, 
            diffusion=diffusion_controller,
            phisgat_model=phisgatv2_model, 
            loss_fn_diff=nn.SmoothL1Loss(reduction='mean', beta=config.SMOOTH_L1_BETA),
            epoch_num=epoch, 
            device=device, 
            amp_dtype=AMP_DTYPE
        )
        
        # === 3. 打印日志与保存最佳模型 ===
        if is_main_process():
            print(f"Epoch {epoch+1}/{config.FINETUNE_EPOCHS} | LR: {current_lr:.2e} | Max Raw Grad: {epoch_max_grad:.2e}")

            train_clash_w = config.W_DIFFUSION * config.W_PHYSICS_CLASH * metrics['phy_clash_raw']
            val_clash_w = config.W_DIFFUSION * config.W_PHYSICS_CLASH * val_metrics['phy_clash_raw']
            train_dist_ll_w = config.W_DIFFUSION * config.W_DIST_LL_LOSS * metrics['dist_ll']
            train_dist_lp_w = config.W_DIFFUSION * config.W_DIST_LP_LOSS * metrics['dist_lp']
            val_dist_ll_w = config.W_DIFFUSION * config.W_DIST_LL_LOSS * val_metrics['dist_ll']
            val_dist_lp_w = config.W_DIFFUSION * config.W_DIST_LP_LOSS * val_metrics['dist_lp']

            # 【输出界面增加 Dist 监控】
            print(f"  [Train] Total: {metrics['total']:.4f} | Diff: {metrics['diff']:.4f} | Clash: {train_clash_w:.4f} | Dist(LL): {train_dist_ll_w:.4f} | Dist(LP): {train_dist_lp_w:.4f} | Act: {metrics['act']:.4f}")
            print(f"  [Valid] Total: {val_metrics['total']:.4f} | Diff: {val_metrics['diff']:.4f} | Clash: {val_clash_w:.4f} | Dist(LL): {val_dist_ll_w:.4f} | Dist(LP): {val_dist_lp_w:.4f} | Act: {val_metrics['act']:.4f}")
            
            if writer:
                writer.add_scalar('Learning_Rate/epoch_lr', current_lr, epoch)
                writer.add_scalar('Gradient/max_raw_norm_before_clip', epoch_max_grad, epoch)
                
                # --- Train 记录 ---
                writer.add_scalar('Loss/total_train', metrics['total'], epoch)
                writer.add_scalar('Loss/diffusion_total_train', config.W_DIFFUSION * metrics['diff'], epoch)
                writer.add_scalar('Loss/topology_edge_train', config.W_DIFFUSION * getattr(config, 'W_EDGE_LOSS', 1.0) * metrics['edge'], epoch)
                writer.add_scalar('Loss/physics_clash_train', config.W_DIFFUSION * config.W_PHYSICS_CLASH * metrics['phy_clash_raw'], epoch)  
                writer.add_scalar('Loss/activity_train', config.W_ACTIVITY * metrics['act'], epoch)
                writer.add_scalar('Loss/variance_train', config.W_VARIANCE * metrics['var'], epoch)
                
                # 扩散分解
                writer.add_scalar('Diffusion_Components/feature_loss', config.W_DIFFUSION * config.W_FEAT_LOSS * metrics['feat'], epoch)
                writer.add_scalar('Diffusion_Components/position_loss', config.W_DIFFUSION * config.W_POS_LOSS * metrics['pos'], epoch)
                writer.add_scalar('Diffusion_Components/frame_loss', config.W_DIFFUSION * config.W_FRAME_LOSS * metrics['frame'], epoch)
                writer.add_scalar('Diffusion_Components/distance_ll_loss', train_dist_ll_w, epoch)
                writer.add_scalar('Diffusion_Components/distance_lp_loss', train_dist_lp_w, epoch)
                
                writer.add_scalar('Diagnosis_Feat_Pure_Error/Train_Small_Frags(HAC<=3)', metrics['feat_small'], epoch)
                writer.add_scalar('Diagnosis_Feat_Pure_Error/Train_Large_Frags(HAC>=10)', metrics['feat_large'], epoch)
                writer.add_scalar('Diagnosis_Feat_Pure_Error/Train_Physics_Clash_Raw', metrics['phy_clash_raw'], epoch)
                
                # --- Valid 记录 ---
                writer.add_scalar('Loss/total_validation', val_metrics['total'], epoch)
                writer.add_scalar('Loss/diffusion_total_validation', config.W_DIFFUSION * val_metrics['diff'], epoch)
                writer.add_scalar('Loss/topology_edge_validation', config.W_DIFFUSION * getattr(config, 'W_EDGE_LOSS', 1.0) * val_metrics['edge'], epoch)
                writer.add_scalar('Loss/physics_clash_validation', config.W_DIFFUSION * config.W_PHYSICS_CLASH * val_metrics['phy_clash_raw'], epoch)
                writer.add_scalar('Loss/activity_validation', config.W_ACTIVITY * val_metrics['act'], epoch)
                writer.add_scalar('Loss/variance_validation', config.W_VARIANCE * val_metrics['var'], epoch)
                
                writer.add_scalar('Diagnosis_Feat_Pure_Error/Valid_Small_Frags(HAC<=3)', val_metrics['feat_small'], epoch)
                writer.add_scalar('Diagnosis_Feat_Pure_Error/Valid_Large_Frags(HAC>=10)', val_metrics['feat_large'], epoch)
                writer.add_scalar('Diagnosis_Feat_Pure_Error/Valid_Physics_Clash_Raw', val_metrics['phy_clash_raw'], epoch)

            # --- 严格依据验证集总损失来保存模型 ---
            if val_metrics['total'] < best_finetune_val_loss:
                best_finetune_val_loss = val_metrics['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': e3nn_model_finetune.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'best_loss': best_finetune_val_loss
                }, config.BEST_MODEL_PATH_FINETUNE_SA)
                
                if is_main_process():
                    print(f"  -> 新的最佳联合微调模型已保存 (Val Loss: {best_finetune_val_loss:.4f})")

    if writer:
        writer.close()
        
    if is_main_process():
        print("\n--- 微调完成 ---")
        print(f"最佳模型保存在: '{config.BEST_MODEL_PATH_FINETUNE_SA}'")
    
    dist.destroy_process_group()