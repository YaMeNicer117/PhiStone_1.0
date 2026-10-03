import os
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

# Local Imports
import PhiSSE3TD_config as config
from PhiSSE3TD_dataset import get_test_dataloader
from PhiSSE3TD_model import E3NNTransformerDiffusion
from PhiSSE3TD_diffusion import DiffusionProcess

# 【核心修订 1】：清理废弃函数，导入全新的内部相对距离损失
from PhiSSE3TD_utils_geom import (
    compute_edge_topology_loss,
    compute_physical_constraint_loss,
    compute_weighted_feat_loss,
    compute_sparse_distance_loss  
)

# 性能优化：关闭异常检测
torch.autograd.set_detect_anomaly(False)

def test_evaluation():
    # 固定随机种子以保证测试结果可复现
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    print("\n" + "="*60)
    print("--- 开始测试集评估 (Test Set Evaluation) ---")
    print("="*60)

    # 1. 设置设备与硬件自适应 AMP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [同步修订] 硬件自适应 AMP 嗅探
    SUPPORTS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    AMP_DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16
    
    print(f"测试设备: {device}")
    print(f"AMP 精度模式: {AMP_DTYPE} (动态自适应)")

    # 2. 获取测试集 DataLoader
    test_loader = get_test_dataloader(batch_size=config.BATCH_SIZE)

    if test_loader is None or len(test_loader) == 0:
        print("错误: 未检测到测试数据或目录为空。")
        return

    # 3. 实例化模型
    print("正在实例化模型...")
    model = E3NNTransformerDiffusion().to(device)
    
    # =================================================================
    # 🚀 [核心修复]：支持动态路由，读取 Config 中的默认模型配置
    # =================================================================
    model_choice = getattr(config, 'DEFAULT_MODEL_CHOICE', 'PRETRAIN')
    model_path = config.MODEL_WEIGHT_ROUTES.get(model_choice)
    
    if model_path is None or not os.path.exists(model_path):
        print(f"[警告] 模型文件未找到: {model_path}。尝试回退加载预训练基座...")
        model_path = getattr(config, 'BEST_MODEL_PATH_PRETRAIN', None)

    print(f"正在加载测试权重 [{model_choice}]: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"读取到训练元数据 -> Epoch: {checkpoint.get('epoch', 'N/A')}")
        else:
            state_dict = checkpoint
            
        # 兼容 DDP 保存的前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        print("权重加载成功。")
    except Exception as e:
        print(f"加载权重失败: {e}")
        return

    # 5. 准备扩散控制器
    diffusion = DiffusionProcess(model=model, device=device)
    loss_fn = nn.SmoothL1Loss(reduction='mean', beta=config.SMOOTH_L1_BETA)

    # 6. 开始评估
    model.eval()
    
    # 初始化全局统计量 (分离 LL 和 LP 的距离损失)
    total_loss = total_feat = total_pos = total_frame = total_edge = total_physics = total_dist_ll = total_dist_lp = 0.0
    num_batches = len(test_loader)

    # 测试采样参数
    NUM_T_SAMPLES = 4 
    physics_threshold = diffusion.num_timesteps * config.PHYSICS_GUIDANCE_RATIO
    
    # 安全获取配置中的新权重，防止老 config 报错
    W_EDGE_LOSS = getattr(config, 'W_EDGE_LOSS', 2.0)
    W_DIST_LL_LOSS = getattr(config, 'W_DIST_LL_LOSS', 1.0)
    W_DIST_LP_LOSS = getattr(config, 'W_DIST_LP_LOSS', 1.0)
    DIST_LL_MAX_RADIUS = getattr(config, 'DIST_LL_MAX_RADIUS', 12.0)
    DIST_LP_MAX_RADIUS = getattr(config, 'DIST_LP_MAX_RADIUS', 6.0)

    print(f"\n开始推断... (每个 Batch 采样 {NUM_T_SAMPLES} 次时间步以降低方差)")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            
            # 准备 Ground Truth
            is_ligand_mask = batch.is_ligand
            x0_feat_all = batch.x[is_ligand_mask]
            x0_feat = x0_feat_all[:, :config.LIGAND_FEATURE_DIM]
            x0_pos = batch.pos[is_ligand_mask]
            x0_ref_coords = batch.ref_coords[is_ligand_mask]
            ligand_batch_indices = batch.batch[is_ligand_mask]
            
            ligand_hac = batch.hac[is_ligand_mask] if hasattr(batch, 'hac') else None
            
            is_dummy_mask = getattr(batch, 'is_dummy', torch.zeros_like(batch.is_ligand))
            protein_mask_global = (~batch.is_ligand) & (~is_dummy_mask)
            
            protein_batch_indices = batch.batch[protein_mask_global]
            sub_prot_pos = batch.pos[protein_mask_global]
            sub_prot_ref = batch.ref_coords[protein_mask_global]

            batch_loss = batch_feat = batch_pos = batch_frame = batch_edge = batch_physics = batch_dist_ll = batch_dist_lp = 0.0

            for _ in range(NUM_T_SAMPLES):
                # =================================================================
                # 🚀 [核心修复]：引入测试期动态掩码，50% 测 Creative，50% 测 Survival
                # =================================================================
                num_lig_nodes = x0_pos.shape[0]
                is_fixed_ligand = torch.zeros(num_lig_nodes, dtype=torch.bool, device=device)
                
                for b_idx in torch.unique(ligand_batch_indices):
                    graph_node_idx = torch.nonzero(ligand_batch_indices == b_idx).squeeze(-1)
                    n_nodes = graph_node_idx.numel()
                    
                    if n_nodes > 1:
                        # 综合评估模型的泛化能力
                        if torch.rand(1).item() < 0.5:
                            num_fixed = 0
                        else:
                            keep_ratio = torch.empty(1).uniform_(0.1, 0.6).item()
                            num_fixed = max(1, min(int(n_nodes * keep_ratio), n_nodes - 1))
                        
                        if num_fixed > 0:
                            perm = torch.randperm(n_nodes, device=device)
                            is_fixed_ligand[graph_node_idx[perm[:num_fixed]]] = True
                        
                is_free_ligand = ~is_fixed_ligand
                
                t = torch.randint(0, diffusion.num_timesteps, (batch.num_graphs,), device=device)
                
                # 前向扩散
                (xt_feat, xt_pos, xt_ref_coords), (true_noise_feat, true_noise_pos, true_noise_frame) = diffusion.q_sample(
                    x0_feat, x0_pos, x0_ref_coords, t, ligand_batch_indices
                )
                
                # =================================================================
                # 🚀 [核心修复]：洗白固定锚点
                # =================================================================
                xt_feat[is_fixed_ligand] = x0_feat[is_fixed_ligand]
                xt_pos[is_fixed_ligand] = x0_pos[is_fixed_ligand]
                xt_ref_coords[is_fixed_ligand] = x0_ref_coords[is_fixed_ligand]
                
                noisy_batch = batch.clone()
                noisy_batch.x[is_ligand_mask, :config.LIGAND_FEATURE_DIM] = xt_feat
                noisy_batch.pos[is_ligand_mask] = xt_pos
                noisy_batch.ref_coords[is_ligand_mask] = xt_ref_coords
                
                noisy_batch_dynamic_edges = diffusion.rebuild_graph_with_dynamic_edges(noisy_batch)

                with autocast(device_type='cuda', enabled=True, dtype=AMP_DTYPE):
                    pred_noise_feat, pred_noise_pos, pred_noise_frame, edge_logits, filtered_edge_index = model(noisy_batch_dynamic_edges, t)

                    # =================================================================
                    # 🚀 [核心修复]：仅仅对自由节点 (is_free_ligand) 计算基础扩散损失
                    # =================================================================
                    if is_free_ligand.any():
                        free_pred_noise_feat = pred_noise_feat[is_free_ligand]
                        free_true_noise_feat = true_noise_feat[is_free_ligand]
                        free_ligand_hac = ligand_hac[is_free_ligand] if ligand_hac is not None else None
                        
                        l_feat, _, _ = compute_weighted_feat_loss(
                            free_pred_noise_feat, free_true_noise_feat, free_ligand_hac, beta=config.SMOOTH_L1_BETA
                        )
                        l_pos = loss_fn(pred_noise_pos[is_free_ligand], true_noise_pos[is_free_ligand])
                        l_frame = loss_fn(pred_noise_frame[is_free_ligand], true_noise_frame[is_free_ligand])
                    else:
                        l_feat, l_pos, l_frame = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                    
                    # 拓扑边损失 (Focal Loss) 忽略 Fixed-Fixed
                    is_fixed_global = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
                    is_fixed_global[is_ligand_mask] = is_fixed_ligand
                    u, v = filtered_edge_index
                    target_edge_mask = ~(is_fixed_global[u] & is_fixed_global[v])
                    
                    l_edge = compute_edge_topology_loss(
                        filtered_edge_index=filtered_edge_index[:, target_edge_mask], 
                        edge_logits=edge_logits[target_edge_mask], 
                        gt_edge_index=batch.gt_edge_index, 
                        gt_edge_attr=batch.gt_edge_attr, 
                        num_nodes=batch.num_nodes,
                        gamma=config.FOCAL_LOSS_GAMMA
                    )

                    t_nodes_float = t[ligand_batch_indices].float()
                    physics_weight_nodes = torch.clamp(1.0 - (t_nodes_float / physics_threshold), min=0.0, max=1.0)
                    physics_node_mask = physics_weight_nodes > 0.0
                    
                    x0_pred_feat, x0_pred_pos, x0_pred_ref = diffusion.predict_x0_from_noise(
                        (xt_feat, xt_pos, xt_ref_coords),
                        (pred_noise_feat, pred_noise_pos, pred_noise_frame),
                        t, ligand_batch_indices 
                    )
                    
                    # =================================================================
                    # 🚀 [核心修复]：保护锚点坐标，阻止垃圾物理梯度影响指标
                    # =================================================================
                    x0_pred_pos_corrected = x0_pred_pos.clone()
                    x0_pred_pos_corrected[is_fixed_ligand] = x0_pos[is_fixed_ligand]
                    x0_pred_ref_corrected = x0_pred_ref.clone()
                    x0_pred_ref_corrected[is_fixed_ligand] = x0_ref_coords[is_fixed_ligand]
                    
                    l_dist_ll, l_dist_lp = compute_sparse_distance_loss(
                        pos_pred=x0_pred_pos_corrected, 
                        pos_true=x0_pos, 
                        batch_ligand=ligand_batch_indices,
                        protein_pos=sub_prot_pos,
                        protein_batch=protein_batch_indices,
                        ll_radius=DIST_LL_MAX_RADIUS, 
                        lp_radius=DIST_LP_MAX_RADIUS, 
                        beta=config.SMOOTH_L1_BETA
                    )
                    
                    l_clash_ll, l_clash_lp, _ = compute_physical_constraint_loss(
                        x0_pred_pos_corrected, x0_pred_ref_corrected,
                        sub_prot_pos, sub_prot_ref,
                        ligand_batch_indices, protein_batch_indices,
                        mask_physics=physics_node_mask,
                        physics_weight_nodes=physics_weight_nodes,
                        hac_tensor=ligand_hac  
                    )
                    
                    l_physics = config.W_PHYSICS_CLASH * (l_clash_ll + l_clash_lp)

                    current_loss_val = (config.W_FEAT_LOSS * l_feat + 
                                        config.W_POS_LOSS * l_pos + 
                                        config.W_FRAME_LOSS * l_frame +
                                        W_EDGE_LOSS * l_edge + 
                                        W_DIST_LL_LOSS * l_dist_ll + 
                                        W_DIST_LP_LOSS * l_dist_lp + 
                                        l_physics)
                
                batch_loss += current_loss_val.item()
                batch_feat += l_feat.item()
                batch_pos += l_pos.item()
                batch_frame += l_frame.item()
                batch_edge += l_edge.item()
                batch_dist_ll += l_dist_ll.item()
                batch_dist_lp += l_dist_lp.item()
                batch_physics += l_physics.item()
            
            total_loss += batch_loss / NUM_T_SAMPLES
            total_feat += batch_feat / NUM_T_SAMPLES
            total_pos += batch_pos / NUM_T_SAMPLES
            total_frame += batch_frame / NUM_T_SAMPLES
            total_edge += batch_edge / NUM_T_SAMPLES
            total_dist_ll += batch_dist_ll / NUM_T_SAMPLES
            total_dist_lp += batch_dist_lp / NUM_T_SAMPLES
            total_physics += batch_physics / NUM_T_SAMPLES

    # 7. 计算并输出最终平均结果
    avg_loss = total_loss / num_batches
    avg_feat = total_feat / num_batches
    avg_pos = total_pos / num_batches
    avg_frame = total_frame / num_batches
    avg_edge = total_edge / num_batches
    avg_dist_ll = total_dist_ll / num_batches
    avg_dist_lp = total_dist_lp / num_batches
    avg_physics = total_physics / num_batches 

    print("\n" + "*"*50)
    print(f"测试集最终评估报告 (Weighted Evaluation) | 模型: {model_choice}")
    print(f"Avg Total Loss   : {avg_loss:.6f} (加权后总和)")
    print("-" * 30)
    print(f"Feature (L1)     : {avg_feat:.6f}")
    print(f"Position (L1)    : {avg_pos:.6f} (绝对位置)")
    print(f"Dist Matrix (LL) : {avg_dist_ll:.6f} (配体刚性)")
    print(f"Dist Matrix (LP) : {avg_dist_lp:.6f} (口袋互作)")
    print(f"Frame (L1)       : {avg_frame:.6f}")
    print(f"Edge Topo (Focal): {avg_edge:.6f}")
    print(f"Physics (Penalty): {avg_physics:.6f}") 
    print("*"*50 + "\n")

if __name__ == '__main__':
    test_evaluation()