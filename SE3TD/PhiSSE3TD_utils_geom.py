import torch
import numpy as np
import sys
import os
import PhiSSE3TD_config as config
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, radius
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch

DEBUG_TENSOR_HEALTH = os.environ.get('DEBUG_TENSOR_HEALTH', '0') == '1'

def compute_edge_topology_loss(filtered_edge_index, edge_logits, gt_edge_index, gt_edge_attr, num_nodes, gamma=2.0):
    """
    [极速版] 拓扑边分类 Focal Loss 计算
    注意：此时传入的 edge_logits 和 filtered_edge_index 已经完全剥离了 PP 边，纯天然无污染，直接算！
    """
    device = edge_logits.device
    num_filtered_edges = filtered_edge_index.shape[1]
    
    # 极速短路：如果没有有效边，直接返回 0，但要挂在计算图上防止 DDP 报错断流
    if num_filtered_edges == 0 or edge_logits.shape[0] == 0:
        return edge_logits.sum() * 0.0

    # --- 1. 唯一标识 (Hash) 边 ---
    dynamic_ids = filtered_edge_index[0] * num_nodes + filtered_edge_index[1]
    gt_ids = gt_edge_index[0] * num_nodes + gt_edge_index[1]
    
    # --- 2. 构建目标标签 (默认全部为 5: Null) ---
    target_classes = torch.full((num_filtered_edges,), 5, dtype=torch.long, device=device)
    
    if gt_ids.numel() > 0:
        gt_classes = torch.argmax(gt_edge_attr, dim=-1)
        
        # 排序以便使用 searchsorted 进行光速匹配
        sorted_indices = torch.argsort(gt_ids)
        sorted_gt_ids = gt_ids[sorted_indices]
        sorted_gt_classes = gt_classes[sorted_indices]
        
        search_idx = torch.searchsorted(sorted_gt_ids, dynamic_ids)
        search_idx = torch.clamp(search_idx, 0, sorted_gt_ids.shape[0] - 1)
        
        # 确保真的匹配上了
        valid_mask = (sorted_gt_ids[search_idx] == dynamic_ids)
        target_classes[valid_mask] = sorted_gt_classes[search_idx[valid_mask]]

    # --- 3. 计算 Focal Loss ---
    alpha = torch.tensor(config.EDGE_CLASS_WEIGHTS, device=device)
    
    ce_loss = F.cross_entropy(edge_logits, target_classes, reduction='none')
    pt = torch.exp(-ce_loss)
    
    # 强制截断 pt，防止 1 - pt 变成 0 或负数，杜绝混合精度下的 NaN 梯度
    # 最高限制在 0.9999，保证底数至少有 1e-4 的安全空间
    pt = torch.clamp(pt, min=0.0, max=0.9999)
    
    alpha_t = alpha[target_classes]
    focal_loss = alpha_t * ((1 - pt) ** gamma) * ce_loss
    
    return focal_loss.mean()


def compute_sparse_distance_loss(
    pos_pred, pos_true, batch_ligand,
    protein_pos=None, protein_batch=None,
    ll_radius=12.0, lp_radius=6.0, beta=1.0
):
    """
    [重构版] 基于稀疏图边逻辑的统一相对距离损失 (包含 LL 与 LP)
    - 抛弃 O(N^2) 的稠密矩阵，改用 O(|E|) 的稀疏边计算。
    - LL 维持配体内部刚性，LP 提供口袋锚点定位。
    """
    device = pos_pred.device
    loss_ll = torch.tensor(0.0, device=device)
    loss_lp = torch.tensor(0.0, device=device)

    # 极速短路：如果配体节点不足
    if pos_pred.shape[0] <= 1:
        return loss_ll, loss_lp

    # =================================================================
    # 1. LL 距离损失 (配体-配体 内部刚性)
    # =================================================================
    # 【核心】：使用真实坐标 (pos_true) 寻找物理上真正存在的边
    edge_index_ll = radius_graph(
        pos_true, 
        r=ll_radius, 
        batch=batch_ligand, 
        loop=False, 
        max_num_neighbors=128
    )
    src_ll, dst_ll = edge_index_ll

    if src_ll.numel() > 0:
        # 分别计算预测坐标和真实坐标在这些边上的欧式距离
        # 注意：不要加 1e-8，因为这是真实边，不会重合导致梯度爆炸
        dist_ll_pred = torch.norm(pos_pred[src_ll] - pos_pred[dst_ll], dim=-1)
        dist_ll_true = torch.norm(pos_true[src_ll] - pos_true[dst_ll], dim=-1)
        
        # 使用 Smooth L1 逼迫模型还原真实键长/构象
        loss_ll = F.smooth_l1_loss(dist_ll_pred, dist_ll_true, beta=beta)

    # =================================================================
    # 2. LP 距离损失 (配体-蛋白 互作感知)
    # =================================================================
    if protein_pos is not None and protein_batch is not None and protein_pos.shape[0] > 0:
        # 使用截断半径，仅关注真实世界中发生互作的片段对
        edge_index_lp = radius(
            x=protein_pos, 
            y=pos_true, 
            r=lp_radius, 
            batch_x=protein_batch, 
            batch_y=batch_ligand, 
            max_num_neighbors=64
        )
        
        # radius 的返回值: edge_index[0] 对应 y (配体), edge_index[1] 对应 x (蛋白)
        idx_lig = edge_index_lp[0]
        idx_prot = edge_index_lp[1]

        if idx_lig.numel() > 0:
            dist_lp_pred = torch.norm(pos_pred[idx_lig] - protein_pos[idx_prot], dim=-1)
            dist_lp_true = torch.norm(pos_true[idx_lig] - protein_pos[idx_prot], dim=-1)
            
            loss_lp = F.smooth_l1_loss(dist_lp_pred, dist_lp_true, beta=beta)

    return loss_ll, loss_lp


def compute_physical_constraint_loss(
    ligand_pos, ligand_ref, 
    protein_pos, protein_ref, 
    batch_ligand, batch_protein,
    mask_physics=None,
    physics_weight_nodes=None,
    hac_tensor=None  # <--- 【新增】传入配体的重原子数
):
    """
    [终极稀疏版 + HAC 自适应防穿插] 
    前置时间步掩码，彻底根除无效计算周期。
    引入基于 HAC 的动态碰撞阈值与惩罚权重，利用二次方梯度解决刚体穿插问题。
    """
    device = ligand_pos.device
    num_lig_nodes = ligand_pos.shape[0]

    # =================================================================
    # [DDP 续命锁] 
    # =================================================================
    dummy_zero = torch.tensor(0.0, device=device)

    if num_lig_nodes == 0:
        return dummy_zero, dummy_zero, dummy_zero

    # 1. 极速短路拦截：如果整个 Batch 都在加噪早期（95% 的情况），直接拔管
    if mask_physics is not None and not mask_physics.any():
        return dummy_zero, dummy_zero, dummy_zero

    num_prot_nodes = protein_pos.shape[0] if protein_pos is not None else 0

    # 2. 还原绝对坐标 (由于每个片段有3个参考点，节点数会扩展3倍)
    points_L = (ligand_pos.unsqueeze(1) + ligand_ref).view(-1, 3)
    batch_L_expanded = batch_ligand.repeat_interleave(3)

    if mask_physics is not None:
        mask_physics_expanded = mask_physics.repeat_interleave(3)
    else:
        mask_physics_expanded = torch.ones_like(batch_L_expanded, dtype=torch.bool)

    # -----------------------------------------------------------------
    # 同步扩展掩码、权重和 HAC
    # -----------------------------------------------------------------
    if physics_weight_nodes is not None:
        weight_expanded = physics_weight_nodes.repeat_interleave(3)
    else:
        weight_expanded = torch.ones_like(batch_L_expanded, dtype=torch.float32)
        
    if hac_tensor is not None:
        hac_expanded = hac_tensor.repeat_interleave(3)
    else:
        # 兜底：未传入 HAC 时默认按最大的刚体处理
        hac_expanded = torch.full_like(batch_L_expanded, config.CLASH_HAC_CAP, dtype=torch.float32)

    # =================================================================
    # [核心优化：前置预过滤] 剥离出真正需要物理指导的骨架原子
    # =================================================================
    valid_idx = torch.nonzero(mask_physics_expanded).squeeze(1)

    if valid_idx.numel() == 0:
        return dummy_zero, dummy_zero, dummy_zero

    points_L_valid = points_L[valid_idx]
    batch_L_valid = batch_L_expanded[valid_idx]

    # 提取与有效坐标一一对应的有效权重和 HAC
    weight_valid = weight_expanded[valid_idx]
    hac_valid = hac_expanded[valid_idx]

    # 计算真正参与物理约束的配体原子数量
    num_valid_ligand_nodes = max(1, valid_idx.numel() // 3)

    loss_clash_LL = dummy_zero
    loss_clash_LP = dummy_zero
    loss_bound = dummy_zero

    # =================================================================
    # --- 3. [稀疏优化] 配体-配体 (LL) 碰撞 (仅在有效点之间) ---
    # =================================================================
    # 【修改】：使用最大的距离阈值作为雷达搜索半径，捕获所有潜在碰撞
    edge_index_LL = radius_graph(
        points_L_valid, 
        r=config.CLASH_DIST_MAX, 
        batch=batch_L_valid, 
        loop=False, 
        max_num_neighbors=64
    )
    
    src_L, dst_L = edge_index_LL
    
    if src_L.numel() > 0:
        original_src_idx = valid_idx[src_L] // 3
        original_dst_idx = valid_idx[dst_L] // 3
        mask_not_self = original_src_idx != original_dst_idx
        
        src_L = src_L[mask_not_self]
        dst_L = dst_L[mask_not_self]
        
        if src_L.numel() > 0:
            with torch.amp.autocast('cuda', enabled=False):
                p_src_f32 = points_L_valid[src_L].float()
                p_dst_f32 = points_L_valid[dst_L].float()
                diff_LL_f32 = p_src_f32 - p_dst_f32
                dist_LL_f32 = torch.sqrt(torch.sum(diff_LL_f32 ** 2, dim=-1) + 1e-8)
            
            dist_LL = dist_LL_f32.to(points_L_valid.dtype)
            
            # 【核心修订】：HAC 自适应计算
            # 1. 有效 HAC 为发生碰撞的两者中较小的一个
            hac_eff = torch.min(hac_valid[src_L], hac_valid[dst_L])
            
            # 2. 插值系数 S (0.0 ~ 1.0)
            S = torch.clamp((hac_eff - 1.0) / (config.CLASH_HAC_CAP - 1.0), min=0.0, max=1.0)
            
            # 3. 动态阈值与动态权重
            dyn_thresh = config.CLASH_DIST_MIN + S * (config.CLASH_DIST_MAX - config.CLASH_DIST_MIN)
            dyn_weight = config.CLASH_WEIGHT_MIN + S * (config.CLASH_WEIGHT_MAX - config.CLASH_WEIGHT_MIN)
            
            # 4. 二次方排斥力
            clash_LL = torch.relu(dyn_thresh - dist_LL) ** 2
            
            # 5. 应用动态惩罚权重与时间步调度权重
            clash_LL_weighted = clash_LL * dyn_weight * weight_valid[src_L]
            loss_clash_LL = clash_LL_weighted.sum() / num_valid_ligand_nodes

    # =================================================================
    # --- 4. [稀疏优化] 配体-蛋白 (LP) 碰撞 ---
    # =================================================================
    if num_prot_nodes > 0:
        points_P = (protein_pos.unsqueeze(1) + protein_ref).view(-1, 3)
        batch_P_expanded = batch_protein.repeat_interleave(3)
        
        # 【修改】：雷达扫描同样使用最大阈值
        edge_index_LP = radius(
            x=points_P, y=points_L_valid, 
            r=config.CLASH_DIST_MAX, 
            batch_x=batch_P_expanded, batch_y=batch_L_valid, 
            max_num_neighbors=64
        )
        
        idx_L = edge_index_LP[0] # 对应 points_L_valid
        idx_P = edge_index_LP[1] # 对应 points_P
        
        if idx_L.numel() > 0:
            with torch.amp.autocast('cuda', enabled=False):
                p_L_f32 = points_L_valid[idx_L].float()
                p_P_f32 = points_P[idx_P].float()
                diff_LP_f32 = p_L_f32 - p_P_f32
                
                dist_LP_f32 = torch.sqrt(torch.sum(diff_LP_f32 ** 2, dim=-1) + 1e-8)
                
            dist_LP = dist_LP_f32.to(points_L_valid.dtype)
            
            # 【核心修订】：针对 L-P 碰撞，由配体的有效 HAC 决定动态约束
            hac_eff_LP = hac_valid[idx_L]
            S_LP = torch.clamp((hac_eff_LP - 1.0) / (config.CLASH_HAC_CAP - 1.0), min=0.0, max=1.0)
            
            dyn_thresh_LP = config.CLASH_DIST_MIN + S_LP * (config.CLASH_DIST_MAX - config.CLASH_DIST_MIN)
            dyn_weight_LP = config.CLASH_WEIGHT_MIN + S_LP * (config.CLASH_WEIGHT_MAX - config.CLASH_WEIGHT_MIN)
            
            clash_LP = torch.relu(dyn_thresh_LP - dist_LP) ** 2
            
            clash_LP_weighted = clash_LP * dyn_weight_LP * weight_valid[idx_L]
            loss_clash_LP = clash_LP_weighted.sum() / num_valid_ligand_nodes

    # =================================================================
    # --- 5. 边界溢出惩罚 (保持为 Dummy Zero) ---
    # =================================================================
    return loss_clash_LL, loss_clash_LP, loss_bound

def check_tensor_health(tensor, variable_name: str, location_tag: str):
    """
    检查一个张量是否包含 NaN 或 inf。
    警告：此操作会触发 CPU-GPU 同步，严重拖慢训练速度。仅限 Debug 时使用。
    """
    # 极速短路：如果未开启调试模式，直接返回，开销为 0
    if not DEBUG_TENSOR_HEALTH:
        return

    # 只有在 DEBUG 模式下，才执行会阻塞线程的张量检查
    if not torch.all(torch.isfinite(tensor)):
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        rank = os.environ.get('RANK', '0')
        
        print("\n" + "="*80)
        print(f" 严重诊断错误：在 [{location_tag}] (Rank {rank}) 发现不健康的张量! ")
        print(f"变量名: {variable_name}")
        print(f"形状: {tensor.shape}")
        print(f"数据类型: {tensor.dtype}")
        print(f"设备: {tensor.device}")
        print(f"包含 NaN: {has_nan.item()}")
        print(f"包含 Inf: {has_inf.item()}")
        
        finite_vals = tensor[torch.isfinite(tensor)]
        if finite_vals.numel() > 0:
            print(f"有限值的最大值: {finite_vals.max().item()}")
            print(f"有限值的最小值: {finite_vals.min().item()}")
            print(f"有限值的平均值: {finite_vals.mean().item()}")
        else:
            print("张量中不包含任何有限值。")
            
        print("="*80 + "\n")
        
        raise RuntimeError(f"程序已终止 (Rank {rank})，以防止 NaN/Inf 传播。请检查上面的报告。")
    
def get_rotation_matrix_from_vectors(vec1, vec2):
    """
    [原模块五] 计算将向量 vec1 旋转到 vec2 的旋转矩阵 (罗德里格旋转公式)。
    
    Args:
        vec1: 源向量 (numpy array)
        vec2: 目标向量 (numpy array)
    Returns:
        R: 3x3 旋转矩阵 (numpy array)
    """
    a = vec1 / (np.linalg.norm(vec1) + 1e-8)
    b = vec2 / (np.linalg.norm(vec2) + 1e-8)
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    # 如果向量平行 (s ≈ 0)
    if s < 1e-6:
        # 同向
        if c > 0: return np.identity(3)
        # 反向：找到任意垂直轴旋转180度
        else:
            axis = np.array([1, 0, 0])
            if np.abs(np.dot(a, axis)) > 0.99: axis = np.array([0, 1, 0])
            k = np.cross(a, axis); k /= np.linalg.norm(k)
            # Rodrigues rotation for 180 degrees
            K_mat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            return np.identity(3) + 2 * K_mat @ K_mat

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.identity(3) + vx + vx.dot(vx) * ((1 - c) / (s ** 2))
    return R

def check_is_linear_molecule(positions, threshold=config.LINEARITY_THRESHOLD):
    """
    [原模块五] 检查分子构象是否为线性 (所有原子近似在一条直线上)。
    通过 SVD (PCA) 检查第二主成分的大小。
    
    Args:
        positions: [N, 3] 原子坐标 (numpy array)
        threshold: 判定阈值 (默认使用 Config 中的值)
    Returns:
        bool: 是否为线性分子
    """
    if len(positions) < 3: return True
    centered = positions - positions.mean(axis=0)
    _, S, _ = np.linalg.svd(centered)
    return S[1] < threshold

def compute_weighted_feat_loss(pred_noise_feat, true_noise_feat, hac_tensor, beta=1.0):
    """
    [新增] 计算基于重原子数 (HAC) 动态加权的特征损失 (平滑反比例衰减)。
    并返回整体加权 Loss，以及小片段/大片段的纯净误差(用于监控)。
    """
    # 1. 基础 Loss (reduction='none'，保留每个节点的独立损失)
    raw_loss = F.smooth_l1_loss(pred_noise_feat, true_noise_feat, reduction='none', beta=beta)
    
    # 沿着特征维度求均值，得到每个节点的标量 Loss -> shape: [num_ligand_nodes]
    node_loss = raw_loss.mean(dim=-1) 
    
    loss_small_pure = 0.0
    loss_large_pure = 0.0
    
    # 2. 计算 HAC 权重
    if getattr(config, 'USE_HAC_FOCAL_LOSS', False) and hac_tensor is not None:
        # 确保 hac_tensor 至少为 1 (防止除 0 报错)
        safe_hac = torch.clamp(hac_tensor, min=1.0)
        
        # 核心公式: Weight = 1.0 + (Max_Penalty - 1.0) / (HAC ^ Decay_Beta)
        max_penalty = getattr(config, 'HAC_PENALTY_MAX', 5.0)
        decay_beta = getattr(config, 'HAC_DECAY_BETA', 1.0)
        
        node_weights = 1.0 + (max_penalty - 1.0) / (safe_hac ** decay_beta)
        
        # 3. 加权并求均值得到最终参与反向传播的 Loss
        weighted_feat_loss = (node_loss * node_weights).mean()
        
        # --- 分离监控指标 (纯净误差，不乘权重) ---
        # 仅在需要监控时计算，使用 detach 防止影响梯度图
        with torch.no_grad():
            small_mask = hac_tensor <= 3.0
            large_mask = hac_tensor >= 10.0
            
            if small_mask.any():
                loss_small_pure = node_loss[small_mask].mean().item()
            if large_mask.any():
                loss_large_pure = node_loss[large_mask].mean().item()
    else:
        # 如果没开启，退化为普通均值
        weighted_feat_loss = node_loss.mean()
        
    return weighted_feat_loss, loss_small_pure, loss_large_pure
