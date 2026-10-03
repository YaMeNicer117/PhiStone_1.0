import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, radius
import math

# Local Imports
import PhiSSE3TD_config as config

class DiffusionProcess:
    def __init__(self, model: nn.Module, num_timesteps=config.NUM_TIMESTEPS, 
                beta_schedule=config.BETA_SCHEDULE, device=config.DEVICE):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        
        if beta_schedule == 'linear':
            self.betas = self._get_linear_schedule_betas(num_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._get_cosine_schedule_betas(num_timesteps)
        else:
            raise ValueError(f"不支持的 beta_schedule 类型: {beta_schedule}")
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def _get_linear_schedule_betas(self, num_timesteps, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, num_timesteps, device='cpu').to(self.device)

    def _get_cosine_schedule_betas(self, num_timesteps, s=0.008):
        steps = torch.arange(num_timesteps + 1, device='cpu', dtype=torch.float32)
        f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        return torch.clip(betas[1:], 0.0001, 0.9999).to(self.device)

    # =========================================================================
    # 核心优化：动态图构建 (Optimized Dynamic Graph Construction)
    # =========================================================================

    def rebuild_graph_with_dynamic_edges(self, graph_data: Data, fixed_ligand_edges=None) -> Data:
        ligand_mask = graph_data.is_ligand
        
        # 【核心修订 1】：保护蛋白掩码，将其与虚拟节点严格剥离！
        # 兼容性获取 is_dummy (防止有的老数据没被 transform 覆盖到)
        dummy_mask = getattr(graph_data, 'is_dummy', torch.zeros_like(ligand_mask))
        protein_mask = (~ligand_mask) & (~dummy_mask)

        global_ligand_indices = torch.nonzero(ligand_mask).squeeze(1)
        global_protein_indices = torch.nonzero(protein_mask).squeeze(1)
        global_dummy_indices = torch.nonzero(dummy_mask).squeeze(1)

        pos_ligand = graph_data.pos[ligand_mask]
        batch_ligand = graph_data.batch[ligand_mask]

        # ===== 向量化：构建固定节点 Tensor（替代 Python set）=====
        fixed_tensor = torch.empty(0, dtype=torch.long, device=self.device)
        if fixed_ligand_edges is not None:
            fixed_edge_idx = fixed_ligand_edges['edge_index']
            all_fixed_nodes = torch.cat([fixed_edge_idx[0], fixed_edge_idx[1]]).unique()
            fixed_tensor = all_fixed_nodes[ligand_mask[all_fixed_nodes]]

        # -----------------------------------------------------------
        # A. L-L 边
        # -----------------------------------------------------------
        if global_ligand_indices.numel() > 0:
            edge_index_ll_local_idx = radius_graph(
                pos_ligand,
                r=config.DYNAMIC_GRAPH_MAX_RADIUS,
                batch=batch_ligand,
                loop=False,
                max_num_neighbors=64
            )
            src_local = edge_index_ll_local_idx[0]
            dst_local = edge_index_ll_local_idx[1]
            src_global = global_ligand_indices[src_local]
            dst_global = global_ligand_indices[dst_local]

            if fixed_tensor.numel() > 0:
                is_src_fixed = torch.isin(src_global, fixed_tensor)
                is_dst_fixed = torch.isin(dst_global, fixed_tensor)
                keep_mask = ~(is_src_fixed & is_dst_fixed)
                edge_index_ll = torch.stack([src_global[keep_mask], dst_global[keep_mask]], dim=0)
            else:
                edge_index_ll = torch.stack([src_global, dst_global], dim=0)

            # 【核心修订 2】：将硬编码的 5 改为 config.EDGE_ATTR_DIM
            attr_ll = torch.zeros((edge_index_ll.shape[1], config.EDGE_ATTR_DIM), device=self.device, dtype=torch.float)
        else:
            edge_index_ll = torch.empty((2, 0), device=self.device, dtype=torch.long)
            attr_ll = torch.empty((0, config.EDGE_ATTR_DIM), device=self.device, dtype=torch.float)

        # -----------------------------------------------------------
        # B. L-P / P-L 边
        # -----------------------------------------------------------
        if global_protein_indices.numel() > 0 and global_ligand_indices.numel() > 0:
            pos_protein = graph_data.pos[protein_mask]
            batch_protein = graph_data.batch[protein_mask]

            radius_output = radius(
                x=pos_protein,
                y=pos_ligand,
                r=config.DYNAMIC_GRAPH_MAX_RADIUS,
                batch_x=batch_protein,
                batch_y=batch_ligand,
                max_num_neighbors=64
            )
            ligand_local_idx = radius_output[0]
            protein_local_idx = radius_output[1]

            idx_lig_global = global_ligand_indices[ligand_local_idx]
            idx_prot_global = global_protein_indices[protein_local_idx]

            if fixed_tensor.numel() > 0:
                keep_mask_lp = ~torch.isin(idx_lig_global, fixed_tensor)
                idx_lig_filtered = idx_lig_global[keep_mask_lp]
                idx_prot_filtered = idx_prot_global[keep_mask_lp]
            else:
                idx_lig_filtered = idx_lig_global
                idx_prot_filtered = idx_prot_global

            edge_index_pl = torch.stack([idx_prot_filtered, idx_lig_filtered], dim=0)
            edge_index_lp = torch.stack([idx_lig_filtered, idx_prot_filtered], dim=0)
            edge_index_inter = torch.cat([edge_index_pl, edge_index_lp], dim=1)
            # 【核心修订 2】
            attr_inter = torch.zeros((edge_index_inter.shape[1], config.EDGE_ATTR_DIM), device=self.device, dtype=torch.float)
        else:
            edge_index_inter = torch.empty((2, 0), device=self.device, dtype=torch.long)
            attr_inter = torch.empty((0, config.EDGE_ATTR_DIM), device=self.device, dtype=torch.float)

        # -----------------------------------------------------------
        # C. P-P 边（静态复用）
        # -----------------------------------------------------------
        src_is_prot = protein_mask[graph_data.edge_index[0]]
        dst_is_prot = protein_mask[graph_data.edge_index[1]]
        is_pp_edge = src_is_prot & dst_is_prot
        edge_index_pp = graph_data.edge_index[:, is_pp_edge]
        edge_attr_pp = graph_data.edge_attr[is_pp_edge]

        # -----------------------------------------------------------
        # D. 注入固定边
        # -----------------------------------------------------------
        if fixed_ligand_edges is not None:
            edge_index_fixed = fixed_ligand_edges['edge_index']
            edge_attr_fixed = fixed_ligand_edges['edge_attr']
        else:
            edge_index_fixed = torch.empty((2, 0), device=self.device, dtype=torch.long)
            edge_attr_fixed = torch.empty((0, config.EDGE_ATTR_DIM), device=self.device, dtype=torch.float)

        # -----------------------------------------------------------
        # E. D-L (Dummy-Ligand) 边 (无视距离的上帝之手)
        # -----------------------------------------------------------
        if global_dummy_indices.numel() > 0 and global_ligand_indices.numel() > 0:
            batch_dummy = graph_data.batch[dummy_mask]
            
            # 创建映射: graph_id -> dummy_node_global_idx
            # 无论这是单图还是包含数百个图的 batch，我们都能找到该图专属的那个原点
            max_batch_idx = graph_data.batch.max() + 1
            dummy_map = torch.empty(max_batch_idx, dtype=torch.long, device=self.device)
            dummy_map[batch_dummy] = global_dummy_indices
            
            # 让每一个配体节点找到属于自己宇宙的那个原点
            matched_dummy_indices = dummy_map[batch_ligand]
            
            # 双向强连接
            edge_index_dl = torch.stack([matched_dummy_indices, global_ligand_indices], dim=0)
            edge_index_ld = torch.stack([global_ligand_indices, matched_dummy_indices], dim=0)
            edge_index_dummy = torch.cat([edge_index_dl, edge_index_ld], dim=1)
            
            # 专属边属性：标记为最后一位 (index 5)
            attr_dummy = torch.zeros((edge_index_dummy.shape[1], config.EDGE_ATTR_DIM), device=self.device, dtype=torch.float)
            attr_dummy[:, -1] = 1.0  
        else:
            edge_index_dummy = torch.empty((2, 0), device=self.device, dtype=torch.long)
            attr_dummy = torch.empty((0, config.EDGE_ATTR_DIM), device=self.device, dtype=torch.float)

        # -----------------------------------------------------------
        # F. 合并
        # -----------------------------------------------------------
        # 【核心修订 3】：将上帝之手缔造的 Dummy 边合入计算图
        final_edge_index = torch.cat([edge_index_pp, edge_index_fixed, edge_index_ll, edge_index_inter, edge_index_dummy], dim=1)
        final_edge_attr = torch.cat([edge_attr_pp, edge_attr_fixed, attr_ll, attr_inter, attr_dummy], dim=0)

        graph_data.edge_index = final_edge_index
        graph_data.edge_attr = final_edge_attr
        return graph_data

    # =========================================================================
    # 扩散采样逻辑 (Sampling Logic)
    # =========================================================================
    @staticmethod
    def _positive_radius(radius, pos):
        if radius is None:
            return None
        radius = torch.as_tensor(radius, device=pos.device, dtype=pos.dtype)
        return torch.clamp(radius, min=1e-6)

    @classmethod
    def _hard_radial_clamp_pos(cls, pos, radius):
        radius = cls._positive_radius(radius, pos)
        if radius is None or pos.numel() == 0:
            return pos

        pos_norm = torch.norm(pos, dim=-1, keepdim=True)
        scale = torch.clamp(radius / (pos_norm + 1e-8), max=1.0)
        return pos * scale

    @classmethod
    def _soft_radial_clamp_pos(cls, pos, radius, ratio):
        radius = cls._positive_radius(radius, pos)
        if radius is None or pos.numel() == 0:
            return pos

        ratio = max(float(ratio), 1e-6)
        soft_radius = radius * ratio
        pos_norm = torch.norm(pos, dim=-1, keepdim=True)
        scale = torch.where(
            pos_norm > soft_radius,
            soft_radius * torch.tanh(pos_norm / soft_radius) / (pos_norm + 1e-8),
            torch.ones_like(pos_norm)
        )
        return pos * scale

    @classmethod
    def _clamp_ligand_center(cls, pos, radius):
        radius = cls._positive_radius(radius, pos)
        if radius is None or pos.numel() == 0:
            return pos

        center = pos.mean(dim=0, keepdim=True)
        center_norm = torch.norm(center, dim=-1, keepdim=True)
        scale = torch.clamp(radius / (center_norm + 1e-8), max=1.0)
        target_center = center * scale
        return pos + (target_center - center)

    @classmethod
    def _apply_position_boundary(cls, pos, radius, soft_ratio=None):
        if radius is None:
            return pos
        if soft_ratio is not None:
            pos = cls._soft_radial_clamp_pos(pos, radius, soft_ratio)
        else:
            pos = cls._hard_radial_clamp_pos(pos, radius)

        pos = cls._clamp_ligand_center(pos, radius)
        return cls._hard_radial_clamp_pos(pos, radius)

    def q_sample(self, x0_feat, x0_pos, x0_ref_coords, t, batch_indices):
        noise_feat = torch.randn_like(x0_feat)
        noise_pos = torch.randn_like(x0_pos)
        noise_ref_coords = torch.randn_like(x0_ref_coords)
        t_nodes = t[batch_indices]
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_nodes]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_nodes]
        
        xt_feat = sqrt_alphas_cumprod_t.unsqueeze(-1) * x0_feat + sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * noise_feat
        xt_pos = sqrt_alphas_cumprod_t.unsqueeze(-1) * x0_pos + sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * noise_pos
        xt_ref_coords = sqrt_alphas_cumprod_t.view(-1, 1, 1) * x0_ref_coords + sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1) * noise_ref_coords
        return (xt_feat, xt_pos, xt_ref_coords), (noise_feat, noise_pos, noise_ref_coords)

    def predict_x0_from_noise(self, xt_tuple, pred_noise_tuple, t, batch_indices, 
                            pos_clamp_min=None, pos_clamp_max=None, pos_clamp_radius=None):
        xt_feat, xt_pos, xt_ref_coords = xt_tuple
        pred_noise_feat, pred_noise_pos, pred_noise_frame = pred_noise_tuple
        t_nodes = t[batch_indices]
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_nodes]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_nodes]
        
        safe_denom_t_node = sqrt_alphas_cumprod_t + 1e-8
        x0_pred_feat = (xt_feat - sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * pred_noise_feat) / safe_denom_t_node.unsqueeze(-1)
        x0_pred_pos = (xt_pos - sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * pred_noise_pos) / safe_denom_t_node.unsqueeze(-1)
        x0_pred_ref_coords = (xt_ref_coords - sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1) * pred_noise_frame) / safe_denom_t_node.view(-1, 1, 1)        
       
        # 裁剪逻辑
        if pos_clamp_radius is not None:
            x0_pred_pos = self._apply_position_boundary(x0_pred_pos, pos_clamp_radius)
        elif pos_clamp_min is not None and pos_clamp_max is not None:
            x0_pred_pos = torch.max(torch.min(x0_pred_pos, pos_clamp_max), pos_clamp_min)
        else:
            x0_pred_pos = torch.clamp(
                x0_pred_pos,
                -config.POS_ABS_SAFETY_CLAMP,
                config.POS_ABS_SAFETY_CLAMP
            )
        
        # --- 修复：区分 Embedding 和 Chem Props 的截断策略 ---
        # 1. 切片提取 Embedding (前 128 维) 和 化学属性 (后 14 维)
        pred_embeds = x0_pred_feat[:, :config.EMBEDDING_DIM_IN]
        pred_chem_props = x0_pred_feat[:, config.EMBEDDING_DIM_IN : config.EMBEDDING_DIM_IN + config.CHEM_PROPS_DIM_IN]
        
        # 2. 对 Embedding 放宽限制 (仅防 Inf 数值崩溃，不改变语义方向，这里设为 20.0 作为绝对安全网)
        pred_embeds = torch.clamp(
            pred_embeds,
            -config.EMBEDDING_SAFETY_CLAMP,
            config.EMBEDDING_SAFETY_CLAMP
        )
        
        # 3. 仅对 化学属性 执行严格的 FEAT_CLAMP_BUFFER (如 4.0)
        pred_chem_props = torch.clamp(pred_chem_props, -config.FEAT_CLAMP_BUFFER, config.FEAT_CLAMP_BUFFER)
        
        # 4. 重新拼接
        x0_pred_feat = torch.cat([pred_embeds, pred_chem_props], dim=-1)
        
        x0_pred_ref_coords = torch.clamp(x0_pred_ref_coords, -config.REF_COORDS_CLAMP_BUFFER, config.REF_COORDS_CLAMP_BUFFER)

        return x0_pred_feat, x0_pred_pos, x0_pred_ref_coords

    @torch.no_grad()
    def p_sample_ddim(self, xt_tuple, protein_data, t, t_prev, eta,
                      known_chem_props=None, fixed_ligand_edges=None):
        xt_feat, xt_pos, xt_ref_coords = xt_tuple
        num_ligand_nodes = xt_pos.shape[0]

        # 【核心修订 4】：Type Encoding 升至 3 维 [配体, 蛋白, 虚拟]
        ligand_type_encoding = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(num_ligand_nodes, 1)

        if known_chem_props is not None:
            assert known_chem_props.shape == (num_ligand_nodes, config.CHEM_PROPS_DIM_IN)
            current_chem_props = known_chem_props
        else:
            current_chem_props = xt_feat[:, config.EMBEDDING_DIM_IN:config.LIGAND_FEATURE_DIM]

        embedded_frags = xt_feat[:, :config.EMBEDDING_DIM_IN]
        xt_ligand_x = torch.cat([embedded_frags, current_chem_props, ligand_type_encoding], dim=1)

        protein_nodes_mask = ~protein_data.is_ligand

        temp_data = Data(
            x=torch.cat([protein_data.x[protein_nodes_mask], xt_ligand_x], dim=0),
            pos=torch.cat([protein_data.pos[protein_nodes_mask], xt_pos], dim=0),
            ref_coords=torch.cat([protein_data.ref_coords[protein_nodes_mask], xt_ref_coords], dim=0),
            is_ligand=torch.cat([
                torch.zeros(protein_nodes_mask.sum(), dtype=torch.bool, device=self.device),
                torch.ones(num_ligand_nodes, dtype=torch.bool, device=self.device)
            ]),
            # 【核心修订 5】：透传 is_dummy 掩码，保证虚拟锚点在推理时不丢失身份
            is_dummy=torch.cat([
                protein_data.is_dummy[protein_nodes_mask],
                torch.zeros(num_ligand_nodes, dtype=torch.bool, device=self.device)
            ]),
            edge_index=protein_data.edge_index,
            edge_attr=protein_data.edge_attr
        ).to(self.device)

        temp_data.batch = torch.zeros(temp_data.num_nodes, dtype=torch.long, device=self.device)

        data_t = self.rebuild_graph_with_dynamic_edges(temp_data, fixed_ligand_edges=fixed_ligand_edges)
        
        num_graphs = 1
        time_tensor = t.repeat(num_graphs)
        pred_noise_feat, pred_noise_pos, pred_noise_frame, edge_logits, filtered_edge_index = self.model(data_t, time_tensor)

        protein_pos = protein_data.pos[protein_nodes_mask]
        if protein_pos.shape[0] > 0:
            max_protein_dist = torch.norm(protein_pos, dim=1).max()
            clamp_radius = max_protein_dist + config.POS_CLAMP_BUFFER
        else:
            clamp_radius = None

        x0_pred_feat, x0_pred_pos, x0_pred_ref_coords = self.predict_x0_from_noise(
            (xt_feat[:, :config.LIGAND_FEATURE_DIM], xt_pos, xt_ref_coords),
            (pred_noise_feat, pred_noise_pos, pred_noise_frame),
            t, torch.zeros(xt_feat.shape[0], dtype=torch.long, device=self.device),
            pos_clamp_radius=clamp_radius
        )
        
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
        
        if t_prev < 0:
            x0_pred_pos = self._apply_position_boundary(x0_pred_pos, clamp_radius)
            return (x0_pred_feat, x0_pred_pos, x0_pred_ref_coords), (edge_logits, filtered_edge_index)

        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        variance = 1 - alpha_t_prev - sigma_t**2
        direction_term_coeff = torch.sqrt(torch.clamp(variance, min=0.0))
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        
        term1_feat = sqrt_alpha_t_prev * x0_pred_feat
        direction_pointing_to_xt_feat = direction_term_coeff * pred_noise_feat
        random_noise_feat = sigma_t * torch.randn_like(x0_pred_feat) if sigma_t > 0 else 0.
        xt_prev_feat = term1_feat + direction_pointing_to_xt_feat + random_noise_feat
        
        term1_pos = sqrt_alpha_t_prev * x0_pred_pos
        direction_pointing_to_xt_pos = direction_term_coeff * pred_noise_pos
        random_noise_pos = sigma_t * torch.randn_like(xt_pos) if sigma_t > 0 else 0.
        xt_prev_pos = term1_pos + direction_pointing_to_xt_pos + random_noise_pos
        xt_prev_pos = self._apply_position_boundary(
            xt_prev_pos,
            clamp_radius,
            soft_ratio=config.POS_SOFT_CLAMP_RATIO
        )
        term1_ref = sqrt_alpha_t_prev.view(-1, 1, 1) * x0_pred_ref_coords
        direction_pointing_to_xt_ref = direction_term_coeff.view(-1, 1, 1) * pred_noise_frame
        random_noise_ref = sigma_t * torch.randn_like(xt_ref_coords) if sigma_t > 0 else 0.
        xt_prev_ref_coords = term1_ref + direction_pointing_to_xt_ref + random_noise_ref
        
        ENABLE_LANGEVIN_GUIDANCE = getattr(config, 'ENABLE_LANGEVIN_GUIDANCE', True)
        
        if ENABLE_LANGEVIN_GUIDANCE and t_prev >= 0:
            # 【关键】：因为外层包裹在 no_grad() 中，我们需要为坐标临时开启梯度追踪
            with torch.enable_grad():
                # 剥离计算图并设置 requires_grad
                pos_for_grad = xt_prev_pos.detach().requires_grad_(True)
                
                # 临时组装单图 batch 索引
                num_lig_nodes = pos_for_grad.shape[0]
                temp_lig_batch = torch.zeros(num_lig_nodes, dtype=torch.long, device=self.device)
                
                # 提取蛋白坐标与参考系 (前面代码已定义 protein_pos)
                temp_prot_batch = torch.zeros(protein_pos.shape[0], dtype=torch.long, device=self.device)
                protein_ref = protein_data.ref_coords[protein_nodes_mask]
                
                # 引入物理计算模块
                from PhiSSE3TD_utils_geom import compute_physical_constraint_loss
                
                # 计算双重排斥能量
                energy_clash_ll, energy_clash_lp, _ = compute_physical_constraint_loss(
                    ligand_pos=pos_for_grad, 
                    ligand_ref=xt_prev_ref_coords.detach(), # 仅作为定位辅助
                    protein_pos=protein_pos, 
                    protein_ref=protein_ref,
                    batch_ligand=temp_lig_batch, 
                    batch_protein=temp_prot_batch,
                    mask_physics=torch.ones(num_lig_nodes, dtype=torch.bool, device=self.device)
                )
                
                # 合并能量 (同时撑开内部，并推离口袋壁)
                energy_total = energy_clash_ll + energy_clash_lp
                
                if energy_total > 1e-4:
                    # 求导！计算能量下降最快（推开原子）的梯度方向
                    grad_pos = torch.autograd.grad(outputs=energy_total, inputs=pos_for_grad)[0]
                    
                    # 动态时间退火: 保证早推晚不推
                    t_ratio = t.float() / self.num_timesteps
                    base_guidance = getattr(config, 'LANGEVIN_SCALE', 0.2)
                    
                    # 抛物线退火：早期(t_ratio接近1)推力大，后期(t_ratio接近0)迅速衰减为 0
                    current_scale = base_guidance * (t_ratio ** 2) 
                    
                    # 梯度裁剪，防止偶尔能量突变把原子炸飞到宇宙边缘
                    grad_pos = torch.clamp(grad_pos, min=-2.0, max=2.0)
                    
                    # 位移更新：向能量降低的方向（负梯度）推一步！
                    xt_prev_pos = pos_for_grad.detach() - current_scale * grad_pos
                    
                    # 推开后再次应用口袋边界约束，防止配体被强大的斥力弹出口袋外
                    xt_prev_pos = self._apply_position_boundary(
                        xt_prev_pos,
                        clamp_radius,
                        soft_ratio=config.POS_SOFT_CLAMP_RATIO
                    )

        return (xt_prev_feat, xt_prev_pos, xt_prev_ref_coords), (edge_logits, filtered_edge_index)
