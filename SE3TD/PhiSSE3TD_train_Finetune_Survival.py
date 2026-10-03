import os
import contextlib 
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
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

# 生产环境性能优化设置
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
# 1. 训练与验证函数 (Train & Val Steps) - 生存模式定制
# =============================================================================
def train_epoch(model, loader, diffusion, optimizer, loss_fn, scaler, epoch_num, device, sampler=None, accumulation_steps=1, amp_dtype=torch.bfloat16):
    if sampler is not None:
        sampler.set_epoch(epoch_num)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    total_loss, total_feat, total_pos, total_frame = 0.0, 0.0, 0.0, 0.0
    total_physics, total_edge = 0.0, 0.0
    total_dist_ll, total_dist_lp = 0.0, 0.0
    total_physics_clash_raw = 0.0
    total_feat_small, total_feat_large = 0.0, 0.0
    epoch_max_raw_grad = 0.0
    
    if is_main_process():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{config.FINETUNE_EPOCHS} [Survival Fine-tune]", 
                            leave=False, mininterval=30.0)
    else:
        progress_bar = loader
    
    num_batches = len(loader)
    physics_threshold = diffusion.num_timesteps * config.PHYSICS_GUIDANCE_RATIO

    for i, batch in enumerate(progress_bar):
        batch = batch.to(device)
        is_update_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == num_batches)
        sync_context = model.no_sync if (not is_update_step) and isinstance(model, DDP) else contextlib.nullcontext

        with sync_context():
            is_ligand_mask = batch.is_ligand
            x0_feat_all = batch.x[is_ligand_mask]
            x0_feat = x0_feat_all[:, :config.LIGAND_FEATURE_DIM]
            x0_pos = batch.pos[is_ligand_mask]
            x0_ref_coords = batch.ref_coords[is_ligand_mask]
            ligand_batch_indices = batch.batch[is_ligand_mask]
            ligand_hac = batch.hac[is_ligand_mask] if hasattr(batch, 'hac') else None
            
            # =================================================================
            # 🚀 [核心机制 1]：混合掩码生成 (20% 概率纯从头生成，80% 概率片段生长)
            # =================================================================
            num_lig_nodes = x0_pos.shape[0]
            is_fixed_ligand = torch.zeros(num_lig_nodes, dtype=torch.bool, device=device)
            
            for b_idx in torch.unique(ligand_batch_indices):
                graph_node_idx = torch.nonzero(ligand_batch_indices == b_idx).squeeze(-1)
                n_nodes = graph_node_idx.numel()
                
                if n_nodes > 1:
                    # 20% 的概率不留任何锚点，全面加噪 (Creative 模式)
                    if torch.rand(1).item() < 1.0:
                        num_fixed = 0
                    else:
                        # 80% 的概率保留 10% ~ 60% 的节点作为锚点 (Survival 模式)
                        keep_ratio = torch.empty(1).uniform_(0.0, 0.0).item()
                        num_fixed = int(n_nodes * keep_ratio)
                        num_fixed = max(1, min(num_fixed, n_nodes - 1))
                    
                    if num_fixed > 0:
                        perm = torch.randperm(n_nodes, device=device)
                        fixed_idx = graph_node_idx[perm[:num_fixed]]
                        is_fixed_ligand[fixed_idx] = True
                    
            is_free_ligand = ~is_fixed_ligand

            # 加噪
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
            # =================================================================
            
            # 构建动态图 (固定节点以干净状态参与构图，引导周围加噪节点)
            noisy_batch = batch.clone()
            noisy_batch.x[is_ligand_mask, :config.LIGAND_FEATURE_DIM] = xt_feat
            noisy_batch.pos[is_ligand_mask] = xt_pos
            noisy_batch.ref_coords[is_ligand_mask] = xt_ref_coords
            noisy_batch_dynamic_edges = diffusion.rebuild_graph_with_dynamic_edges(noisy_batch)
            
            with autocast(device_type='cuda', enabled=True, dtype=amp_dtype):
                pred_noise_feat, pred_noise_pos, pred_noise_frame, edge_logits, filtered_edge_index = model(noisy_batch_dynamic_edges, t)
                
                # 🚀 [核心机制 3]：仅对自由节点算基础 Diffusion Loss
                if is_free_ligand.any():
                    free_pred_noise_feat = pred_noise_feat[is_free_ligand]
                    free_true_noise_feat = true_noise_feat[is_free_ligand]
                    free_ligand_hac = ligand_hac[is_free_ligand] if ligand_hac is not None else None
                    
                    loss_feat, loss_small_pure, loss_large_pure = compute_weighted_feat_loss(
                        free_pred_noise_feat, free_true_noise_feat, free_ligand_hac, beta=config.SMOOTH_L1_BETA
                    )
                    loss_pos = loss_fn(pred_noise_pos[is_free_ligand], true_noise_pos[is_free_ligand])
                    loss_frame = loss_fn(pred_noise_frame[is_free_ligand], true_noise_frame[is_free_ligand])
                else:
                    loss_feat, loss_small_pure, loss_large_pure = torch.tensor(0.0, device=device), 0.0, 0.0
                    loss_pos, loss_frame = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

                loss_raw = (config.W_FEAT_LOSS * loss_feat + 
                            config.W_POS_LOSS * loss_pos + 
                            config.W_FRAME_LOSS * loss_frame)
                
                # 🚀 [核心机制 4]：拓扑边界焦点转移 (免除 Fixed-Fixed 边的惩罚)
                is_fixed_global = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
                is_fixed_global[is_ligand_mask] = is_fixed_ligand
                
                u, v = filtered_edge_index
                is_fixed_edge = is_fixed_global[u] & is_fixed_global[v]
                target_edge_mask = ~is_fixed_edge
                
                filtered_edge_index_for_loss = filtered_edge_index[:, target_edge_mask]
                edge_logits_for_loss = edge_logits[target_edge_mask]

                loss_edge = compute_edge_topology_loss(
                    filtered_edge_index=filtered_edge_index_for_loss, 
                    edge_logits=edge_logits_for_loss, 
                    gt_edge_index=batch.gt_edge_index, 
                    gt_edge_attr=batch.gt_edge_attr, 
                    num_nodes=batch.num_nodes,
                    gamma=config.FOCAL_LOSS_GAMMA
                )
                            
                # 预测 x0 (还原用于计算几何损失)
                x0_pred_feat, x0_pred_pos, x0_pred_ref = diffusion.predict_x0_from_noise(
                    (xt_feat, xt_pos, xt_ref_coords),
                    (pred_noise_feat, pred_noise_pos, pred_noise_frame),
                    t, ligand_batch_indices 
                )

                # =================================================================
                # 🚀 [核心机制 5]：修正物理损失的输入，拦截垃圾坐标防梯度爆炸
                # =================================================================
                x0_pred_pos_corrected = x0_pred_pos.clone()
                x0_pred_pos_corrected[is_fixed_ligand] = x0_pos[is_fixed_ligand]
                
                x0_pred_ref_corrected = x0_pred_ref.clone()
                x0_pred_ref_corrected[is_fixed_ligand] = x0_ref_coords[is_fixed_ligand]
                # =================================================================

                is_dummy_mask = getattr(batch, 'is_dummy', torch.zeros_like(batch.is_ligand))
                protein_mask_global = (~batch.is_ligand) & (~is_dummy_mask)
                protein_batch_indices = batch.batch[protein_mask_global]
                sub_prot_pos = batch.pos[protein_mask_global]
                sub_prot_ref = batch.ref_coords[protein_mask_global]

                # 这里统一传入修正后的 x0_pred_pos_corrected
                loss_dist_ll, loss_dist_lp = compute_sparse_distance_loss(
                    pos_pred=x0_pred_pos_corrected, pos_true=x0_pos, batch_ligand=ligand_batch_indices,
                    protein_pos=sub_prot_pos, protein_batch=protein_batch_indices,
                    ll_radius=config.DIST_LL_MAX_RADIUS, lp_radius=config.DIST_LP_MAX_RADIUS, 
                    beta=config.SMOOTH_L1_BETA
                )

                t_nodes_float = t[ligand_batch_indices].float()
                physics_weight_nodes = torch.clamp(1.0 - (t_nodes_float / physics_threshold), min=0.0, max=1.0)
                physics_node_mask = physics_weight_nodes > 0.0
                
                # 这里统一传入修正后的 corrected 变量
                l_clash_ll, l_clash_lp, _ = compute_physical_constraint_loss(
                    x0_pred_pos_corrected, x0_pred_ref_corrected, sub_prot_pos, sub_prot_ref,
                    ligand_batch_indices, protein_batch_indices,
                    mask_physics=physics_node_mask, physics_weight_nodes=physics_weight_nodes, hac_tensor=ligand_hac
                )
                loss_physics = config.W_PHYSICS_CLASH * (l_clash_ll + l_clash_lp)
                
                total_loss_final = (loss_raw + loss_physics + 
                                    config.W_EDGE_LOSS * loss_edge + 
                                    config.W_DIST_LL_LOSS * loss_dist_ll +
                                    config.W_DIST_LP_LOSS * loss_dist_lp)
                
                if (i + 1) == num_batches:
                    remainder = num_batches % accumulation_steps
                    current_accum_steps = remainder if remainder != 0 else accumulation_steps
                else:
                    current_accum_steps = accumulation_steps
                    
                loss_scaled = total_loss_final / current_accum_steps

            scaler.scale(loss_scaled).backward()
        
        total_loss += total_loss_final.item()
        total_feat += loss_feat.item()
        total_pos += loss_pos.item()
        total_frame += loss_frame.item()
        total_physics += loss_physics.item()
        total_physics_clash_raw += (l_clash_ll + l_clash_lp).item()
        total_edge += loss_edge.item()
        total_dist_ll += loss_dist_ll.item()
        total_dist_lp += loss_dist_lp.item()
        total_feat_small += loss_small_pure
        total_feat_large += loss_large_pure

        if is_update_step:
            scaler.unscale_(optimizer)
            hold_epochs = getattr(config, 'FINETUNE_HOLD_EPOCHS', 0)
            if epoch_num < (config.FINETUNE_WARMUP_EPOCHS + hold_epochs):
                current_max_norm = config.FINETUNE_GRAD_CLIP_WARMUP_NORM
            else:
                current_max_norm = config.FINETUNE_GRAD_CLIP_STABLE_NORM
            
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=current_max_norm)
            current_raw_norm_val = raw_grad_norm.item() if isinstance(raw_grad_norm, torch.Tensor) else raw_grad_norm
            epoch_max_raw_grad = max(epoch_max_raw_grad, current_raw_norm_val)
            
            if current_raw_norm_val > current_max_norm and is_main_process():
                progress_bar.write(f"[警告] 截断前梯度 Norm: {current_raw_norm_val:.2e} (超出阈值 {current_max_norm:.1f}，已截断)")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    
    if dist.is_initialized():
        metrics = torch.tensor([
            avg_loss, total_feat/num_batches, total_pos/num_batches, total_frame/num_batches, 
            total_physics/num_batches, total_edge/num_batches, total_dist_ll/num_batches, total_dist_lp/num_batches,
            total_feat_small/num_batches, total_feat_large/num_batches, total_physics_clash_raw/num_batches
        ], device=device)
        metrics = reduce_mean(metrics, dist.get_world_size())
        return tuple(metrics.tolist()) + (epoch_max_raw_grad,)
    
    return (
        avg_loss, total_feat/num_batches, total_pos/num_batches, total_frame/num_batches, 
        total_physics/num_batches, total_edge/num_batches, total_dist_ll/num_batches, total_dist_lp/num_batches,
        total_feat_small/num_batches, total_feat_large/num_batches, total_physics_clash_raw/num_batches, epoch_max_raw_grad
    )

@torch.no_grad()
def validate_epoch(model, loader, diffusion, loss_fn, epoch_num, device, amp_dtype=torch.bfloat16):
    model.eval()
    
    total_loss, total_feat, total_pos, total_frame = 0.0, 0.0, 0.0, 0.0
    total_physics, total_edge = 0.0, 0.0
    total_dist_ll, total_dist_lp = 0.0, 0.0
    total_physics_clash_raw = 0.0
    total_feat_small, total_feat_large = 0.0, 0.0
    
    if is_main_process():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{config.FINETUNE_EPOCHS} [Validation]", leave=False)
    else:
        progress_bar = loader
        
    physics_threshold = diffusion.num_timesteps * config.PHYSICS_GUIDANCE_RATIO
    
    for batch in progress_bar:
        batch = batch.to(device)
        is_ligand_mask = batch.is_ligand
        x0_feat_all = batch.x[is_ligand_mask]
        x0_feat = x0_feat_all[:, :config.LIGAND_FEATURE_DIM]
        x0_pos = batch.pos[is_ligand_mask]
        x0_ref_coords = batch.ref_coords[is_ligand_mask]
        ligand_batch_indices = batch.batch[is_ligand_mask]
        ligand_hac = batch.hac[is_ligand_mask] if hasattr(batch, 'hac') else None
        
        # 验证集也做混合随机掩码
        num_lig_nodes = x0_pos.shape[0]
        is_fixed_ligand = torch.zeros(num_lig_nodes, dtype=torch.bool, device=device)
        for b_idx in torch.unique(ligand_batch_indices):
            graph_node_idx = torch.nonzero(ligand_batch_indices == b_idx).squeeze(-1)
            n_nodes = graph_node_idx.numel()
            if n_nodes > 1:
                if torch.rand(1).item() < 1.0:
                    num_fixed = 0
                else:
                    keep_ratio = torch.empty(1).uniform_(0.0, 0.0).item()
                    num_fixed = int(n_nodes * keep_ratio)
                    num_fixed = max(1, min(num_fixed, n_nodes - 1))
                
                if num_fixed > 0:
                    perm = torch.randperm(n_nodes, device=device)
                    is_fixed_ligand[graph_node_idx[perm[:num_fixed]]] = True
                
        is_free_ligand = ~is_fixed_ligand

        t = torch.randint(0, diffusion.num_timesteps, (batch.num_graphs,), device=device)
        (xt_feat, xt_pos, xt_ref_coords), (true_noise_feat, true_noise_pos, true_noise_frame) = diffusion.q_sample(
            x0_feat, x0_pos, x0_ref_coords, t, ligand_batch_indices
        )
        
        # 同样洗白固定锚点
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
                loss_feat, loss_small_pure, loss_large_pure = compute_weighted_feat_loss(
                    pred_noise_feat[is_free_ligand], true_noise_feat[is_free_ligand], 
                    ligand_hac[is_free_ligand] if ligand_hac is not None else None, beta=config.SMOOTH_L1_BETA
                )
                loss_pos = loss_fn(pred_noise_pos[is_free_ligand], true_noise_pos[is_free_ligand])
                loss_frame = loss_fn(pred_noise_frame[is_free_ligand], true_noise_frame[is_free_ligand])
            else:
                loss_feat, loss_small_pure, loss_large_pure = torch.tensor(0.0, device=device), 0.0, 0.0
                loss_pos, loss_frame = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            loss_raw = config.W_FEAT_LOSS * loss_feat + config.W_POS_LOSS * loss_pos + config.W_FRAME_LOSS * loss_frame
            
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
            
            x0_pred_feat, x0_pred_pos, x0_pred_ref = diffusion.predict_x0_from_noise(
                (xt_feat, xt_pos, xt_ref_coords), (pred_noise_feat, pred_noise_pos, pred_noise_frame), t, ligand_batch_indices 
            )

            # 同样拦截修正坐标
            x0_pred_pos_corrected = x0_pred_pos.clone()
            x0_pred_pos_corrected[is_fixed_ligand] = x0_pos[is_fixed_ligand]
            
            x0_pred_ref_corrected = x0_pred_ref.clone()
            x0_pred_ref_corrected[is_fixed_ligand] = x0_ref_coords[is_fixed_ligand]

            is_dummy_mask = getattr(batch, 'is_dummy', torch.zeros_like(batch.is_ligand))
            protein_mask_global = (~batch.is_ligand) & (~is_dummy_mask)
            
            loss_dist_ll, loss_dist_lp = compute_sparse_distance_loss(
                pos_pred=x0_pred_pos_corrected, pos_true=x0_pos, batch_ligand=ligand_batch_indices,
                protein_pos=batch.pos[protein_mask_global], protein_batch=batch.batch[protein_mask_global],
                ll_radius=config.DIST_LL_MAX_RADIUS, lp_radius=config.DIST_LP_MAX_RADIUS, beta=config.SMOOTH_L1_BETA
            )

            t_nodes_float = t[ligand_batch_indices].float()
            physics_weight_nodes = torch.clamp(1.0 - (t_nodes_float / physics_threshold), min=0.0, max=1.0)
            
            l_clash_ll, l_clash_lp, _ = compute_physical_constraint_loss(
                x0_pred_pos_corrected, x0_pred_ref_corrected, batch.pos[protein_mask_global], batch.ref_coords[protein_mask_global],
                ligand_batch_indices, batch.batch[protein_mask_global],
                mask_physics=(physics_weight_nodes > 0.0), physics_weight_nodes=physics_weight_nodes, 
                hac_tensor=ligand_hac
            )
            loss_physics = config.W_PHYSICS_CLASH * (l_clash_ll + l_clash_lp)
            
            current_loss = loss_raw + loss_physics + config.W_EDGE_LOSS * loss_edge + config.W_DIST_LL_LOSS * loss_dist_ll + config.W_DIST_LP_LOSS * loss_dist_lp
            
        total_loss += current_loss.item()
        total_feat += loss_feat.item()
        total_pos += loss_pos.item()
        total_frame += loss_frame.item()
        total_edge += loss_edge.item()
        total_dist_ll += loss_dist_ll.item()
        total_dist_lp += loss_dist_lp.item()
        total_physics += loss_physics.item()
        total_physics_clash_raw += (l_clash_ll + l_clash_lp).item()
        total_feat_small += loss_small_pure
        total_feat_large += loss_large_pure
        
    num_val_batches = len(loader)
    avg_loss = total_loss / num_val_batches
    
    if dist.is_initialized():
        metrics = torch.tensor([
            avg_loss, total_feat/num_val_batches, total_pos/num_val_batches, total_frame/num_val_batches, 
            total_physics/num_val_batches, total_edge/num_val_batches, total_dist_ll/num_val_batches, total_dist_lp/num_val_batches,
            total_feat_small/num_val_batches, total_feat_large/num_val_batches, total_physics_clash_raw/num_val_batches
        ], device=device)
        metrics = reduce_mean(metrics, dist.get_world_size())
        return tuple(metrics.tolist())
    
    return (
        avg_loss, total_feat/num_val_batches, total_pos/num_val_batches, total_frame/num_val_batches, 
        total_physics/num_val_batches, total_edge/num_val_batches, total_dist_ll/num_val_batches, total_dist_lp/num_val_batches,
        total_feat_small/num_val_batches, total_feat_large/num_val_batches, total_physics_clash_raw/num_val_batches
    )
# =============================================================================
# 2. 主执行流程 (Main Execution)
# =============================================================================
if __name__ == '__main__':
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    device = torch.device(f"cuda:{local_rank}")
    config.DEVICE = device 

    if is_main_process():
        print("\n" + "="*60)
        print(f"--- 开始阶段二：生存模式微调 (Survival Fine-tuning) [DDP Mode] ---")
        print("="*60)
        
        world_size = dist.get_world_size()
        effective_batch_size = config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS * world_size
        print(f"World Size (GPUs)        : {world_size}")
        print(f"Effective Batch Size     : {effective_batch_size}") 
        print(f"Finetune Learning Rate   : {config.FINETUNE_LR}")
        print("-" * 60)
        
        os.makedirs(config.CHECKPOINT_DIR_FINETUNE, exist_ok=True)
        os.makedirs(config.LOG_DIR_FINETUNE_S, exist_ok=True)
        writer = SummaryWriter(config.LOG_DIR_FINETUNE_S)
    else:
        writer = None

    train_loader, val_loader, train_sampler = get_train_val_dataloaders(distributed=True)
    
    e3nn_model = E3NNTransformerDiffusion().to(device)
    e3nn_model = DDP(e3nn_model, device_ids=[local_rank], find_unused_parameters=False)
    diffusion_controller = DiffusionProcess(model=e3nn_model, device=device)
    best_val_loss = float('inf')
    start_epoch = 0
    loaded_checkpoint = None

    if os.path.exists(config.BEST_MODEL_PATH_FINETUNE_S):
        if is_main_process():
            print(f"  -> 发现微调断点，正在加载微调权重以续训: '{config.BEST_MODEL_PATH_FINETUNE_S}'")
        try:
            checkpoint = torch.load(config.BEST_MODEL_PATH_FINETUNE_S, map_location=device, weights_only=False)
            loaded_checkpoint = checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                e3nn_model.module.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                # best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                # =========================================================
                best_val_loss = float('inf') 
                if is_main_process(): 
                    print(f"  -> 成功从 Epoch {start_epoch} 继续。")
                    print("  -> [提示] 由于验证集损失计算公式更新，已重置历史 best_val_loss。")
                # =========================================================
            else:
                e3nn_model.module.load_state_dict(checkpoint)
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
                e3nn_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                e3nn_model.module.load_state_dict(checkpoint)
                
            if is_main_process():
                print("  -> 预训练基座加载成功。")
        except Exception as e:
            if is_main_process():
                print(f"  -> [致命错误] 预训练基座加载失败: {e}\n     程序退出。")
            dist.destroy_process_group()
            sys.exit(1)
    else:
        # 修改点：移除退出指令，改为打印提示信息
        if is_main_process():
            print(f"  -> [提示] 找不到任何预训练或微调权重文件。模型将从头开始 (随机初始化) 训练。")

    # ========= 继续执行后续的优化器初始化 =========
    optimizer = optim.AdamW(e3nn_model.parameters(), lr=config.FINETUNE_LR, weight_decay=config.FINETUNE_WEIGHT_DECAY)
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        
    loss_fn = nn.SmoothL1Loss(reduction='mean', beta=config.SMOOTH_L1_BETA)
    
    SUPPORTS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    AMP_DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16
    scaler = GradScaler(enabled=(AMP_DTYPE == torch.float16))
    
    def get_lr_multiplier(current_epoch):
        warmup_epochs = config.FINETUNE_WARMUP_EPOCHS
        hold_epochs = getattr(config, 'FINETUNE_HOLD_EPOCHS', 0)
        total_epochs = config.FINETUNE_EPOCHS
        if current_epoch < warmup_epochs:
            return config.FINETUNE_WARMUP_START_LR_RATIO + (1.0 - config.FINETUNE_WARMUP_START_LR_RATIO) * (current_epoch / warmup_epochs)
        elif current_epoch < warmup_epochs + hold_epochs:
            return 1.0
        else:
            decay_epochs = total_epochs - warmup_epochs - hold_epochs
            if decay_epochs <= 0: return 1.0
            current_decay_epoch = current_epoch - warmup_epochs - hold_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * current_decay_epoch / decay_epochs))
            return 0.01 + (1.0 - 0.01) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=start_epoch - 1)

    # 仅在续微调时才恢复优化器状态
    if loaded_checkpoint is not None and isinstance(loaded_checkpoint, dict):
        if 'optimizer_state_dict' in loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            if is_main_process():
                print("  -> 已成功恢复 AdamW 优化器动量状态。")
        if 'scheduler_state_dict' in loaded_checkpoint:
            scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])
            if is_main_process():
                print("  -> 已成功恢复 LR Scheduler 状态。")
                
    if is_main_process(): print("\n--- 开始微调循环 ---")
    
    for epoch in range(start_epoch, config.FINETUNE_EPOCHS):
        
        metrics_train = train_epoch(
            e3nn_model, train_loader, diffusion_controller, optimizer, loss_fn, scaler, epoch, device, 
            train_sampler, config.GRAD_ACCUMULATION_STEPS, AMP_DTYPE 
        )
        
        train_loss, train_feat, train_pos, train_frame, train_physics, train_edge, tr_dist_ll, tr_dist_lp, tr_small, tr_large, tr_phy_clash_raw, epoch_max_grad = metrics_train

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        metrics_val = validate_epoch(
            e3nn_model, val_loader, diffusion_controller, loss_fn, epoch, device, amp_dtype=AMP_DTYPE 
        )
        
        val_loss, val_feat, val_pos, val_frame, val_physics, val_edge, val_dist_ll, val_dist_lp, val_small, val_large, val_phy_clash_raw = metrics_val
        
        if is_main_process():
            print(f"Epoch {epoch+1}/{config.FINETUNE_EPOCHS} | LR: {current_lr:.2e} | Max Raw Grad: {epoch_max_grad:.2e}")
            
            train_clash_w = config.W_PHYSICS_CLASH * tr_phy_clash_raw
            val_clash_w = config.W_PHYSICS_CLASH * val_phy_clash_raw
            train_dist_ll_w = config.W_DIST_LL_LOSS * tr_dist_ll
            train_dist_lp_w = config.W_DIST_LP_LOSS * tr_dist_lp
            val_dist_ll_w = config.W_DIST_LL_LOSS * val_dist_ll
            val_dist_lp_w = config.W_DIST_LP_LOSS * val_dist_lp

            # 1. 补全 Print 输出
            print(f"  [Train] Total: {train_loss:.4f} | Pos: {config.W_POS_LOSS * train_pos:.4f} | Dist(LL): {train_dist_ll_w:.4f} | Dist(LP): {train_dist_lp_w:.4f} | Clash: {train_clash_w:.4f} | Edge: {config.W_EDGE_LOSS * train_edge:.4f}")
            print(f"  [Valid] Total: {val_loss:.4f} | Pos: {config.W_POS_LOSS * val_pos:.4f} | Dist(LL): {val_dist_ll_w:.4f} | Dist(LP): {val_dist_lp_w:.4f} | Clash: {val_clash_w:.4f} | Edge: {config.W_EDGE_LOSS * val_edge:.4f}")

            # 2. 补全 TensorBoard 写入
            if writer:
                writer.add_scalar('FineTune_Meta/learning_rate', current_lr, epoch)
                writer.add_scalar('FineTune_Meta/max_raw_grad_norm', epoch_max_grad, epoch)
                
                # 核心 Loss
                writer.add_scalar('Loss_Finetune_Train/1_Total', train_loss, epoch)
                writer.add_scalar('Loss_Finetune_Train/2_Diffusion_Feat_FreeOnly', config.W_FEAT_LOSS * train_feat, epoch)
                writer.add_scalar('Loss_Finetune_Train/3_Diffusion_Pos_FreeOnly', config.W_POS_LOSS * train_pos, epoch)
                writer.add_scalar('Loss_Finetune_Train/6_Edge_Topology_Filtered', config.W_EDGE_LOSS * train_edge, epoch)
                
                # 物理与几何约束 (此前遗漏的)
                writer.add_scalar('Loss_Finetune_Train/5a_Physics_Clash', train_clash_w, epoch)
                writer.add_scalar('Loss_Finetune_Train/7a_Distance_LL', train_dist_ll_w, epoch)
                writer.add_scalar('Loss_Finetune_Train/7b_Distance_LP', train_dist_lp_w, epoch)
                
                # 验证集同理
                writer.add_scalar('Loss_Finetune_Valid/1_Total', val_loss, epoch)
                writer.add_scalar('Loss_Finetune_Valid/2_Diffusion_Feat_FreeOnly', config.W_FEAT_LOSS * val_feat, epoch)
                writer.add_scalar('Loss_Finetune_Valid/3_Diffusion_Pos_FreeOnly', config.W_POS_LOSS * val_pos, epoch)
                writer.add_scalar('Loss_Finetune_Valid/6_Edge_Topology_Filtered', config.W_EDGE_LOSS * val_edge, epoch)
                writer.add_scalar('Loss_Finetune_Valid/5a_Physics_Clash', val_clash_w, epoch)
                writer.add_scalar('Loss_Finetune_Valid/7a_Distance_LL', val_dist_ll_w, epoch)
                writer.add_scalar('Loss_Finetune_Valid/7b_Distance_LP', val_dist_lp_w, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': e3nn_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),    
                    'scheduler_state_dict': scheduler.state_dict(),    
                    'best_val_loss': best_val_loss
                }, config.BEST_MODEL_PATH_FINETUNE_S)
                print(f"  -> 发现微调最佳模型！(Val Total Loss 下降至: {best_val_loss:.4f})")

    if writer: writer.close()
    dist.destroy_process_group()