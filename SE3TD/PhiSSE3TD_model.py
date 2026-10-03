import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax
from torch.utils.checkpoint import checkpoint

# 引入 e3nn 标准组件
from e3nn import o3, nn as e3nn_nn
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import BatchNorm, Gate
import math
import sys

# Local Imports
import PhiSSE3TD_config as config
from PhiSSE3TD_utils_geom import check_tensor_health

# =============================================================================
# 1. 核心组件 (Core Components)
# =============================================================================
class GatedEquivariantLayerNorm(nn.Module):
    """
    引入自适应模长缩放的等变层归一化 (Gated Vector Norm)。
    利用标量特征动态控制向量特征的模长，打破过度保守的均值化陷阱。
    """
    def __init__(self, irreps, eps=1e-5, affine=True):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        
        # 1. 识别标量部分
        self.num_scalar = sum(mul * ir.dim for mul, ir in self.irreps if ir.l == 0)
        if self.num_scalar > 0:
            self.scalar_norm = nn.LayerNorm(self.num_scalar, eps=eps, elementwise_affine=affine)
            
        # 2. 识别非标量部分
        self.ops = [] 
        self.affine_weights = nn.ParameterList()
        self.num_vector_channels = 0  # 统计向量通道总数
        
        current_idx = 0
        weight_idx_counter = 0
        
        for mul, ir in self.irreps:
            length = mul * ir.dim
            if ir.l == 0:
                pass
            else:
                self.num_vector_channels += mul
                if affine:
                    weight = nn.Parameter(torch.ones(mul))
                    self.affine_weights.append(weight)
                    w_idx = weight_idx_counter
                    weight_idx_counter += 1
                else:
                    w_idx = None
                
                self.ops.append({
                    'slice': slice(current_idx, current_idx + length),
                    'mul': mul,
                    'dim': ir.dim,
                    'w_idx': w_idx
                })
            current_idx += length

        # 3. [核心新增] 自适应模长门控网络 (Gate Network)
        # 仅在同时存在标量和向量时启用
        if self.num_scalar > 0 and self.num_vector_channels > 0:
            # 采用轻量级 MLP，将归一化后的标量特征映射为向量通道的缩放系数
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.num_scalar, self.num_scalar // 2),
                nn.SiLU(),
                nn.Linear(self.num_scalar // 2, self.num_vector_channels),
                nn.Sigmoid() 
            )
        else:
            self.gate_mlp = None

    def forward(self, x):
        out_chunks = []
        
        # 1. 处理标量块 (假设标量永远在最前面)
        if self.num_scalar > 0:
            scalars = x[:, :self.num_scalar]
            normed_scalars = self.scalar_norm(scalars)
            out_chunks.append(normed_scalars)
            
            # [核心新增] 计算动态门控系数
            if self.gate_mlp is not None:
                # 乘以 2.0 的巧思：Sigmoid 输出 (0, 1)。乘以 2 后范围变为 (0, 2)。
                # 这允许模型不仅能“抑制”噪声 (scale < 1)，还能“放大”关键信号 (scale > 1)。
                # 初始状态 Sigmoid(0)*2 = 1.0，正好等于原版不缩放的平稳状态。
                dynamic_gates = self.gate_mlp(normed_scalars) * 2.0
            else:
                dynamic_gates = None
        else:
            normed_scalars = None
            dynamic_gates = None
            
        # 2. 处理非标量块 (按照 ops 的顺序)
        vector_channel_offset = 0
        for op in self.ops:
            sl = op['slice']
            mul, d, w_idx = op['mul'], op['dim'], op['w_idx']
            
            chunk = x[:, sl]
            chunk_view = chunk.reshape(-1, mul, d) 
            
            # 计算各通道模长的均方根 (RMS)，将 EPS 移到根号内加法更稳定
            rms = torch.sqrt(chunk_view.pow(2).mean(dim=2, keepdim=True) + self.eps)
            chunk_normed = chunk_view / rms
            
            # 应用静态仿射变换
            if self.affine and w_idx is not None:
                gamma = self.affine_weights[w_idx].reshape(1, mul, 1)
                chunk_normed = chunk_normed * gamma
                
            # [核心新增] 应用动态自适应模长缩放
            if dynamic_gates is not None:
                # 提取当前操作对应的门控系数 [batch_size, mul]
                gate_slice = dynamic_gates[:, vector_channel_offset : vector_channel_offset + mul]
                # 调整形状为 [batch_size, mul, 1] 以便对三维向量进行广播相乘
                gate_view = gate_slice.unsqueeze(-1)
                
                # 实施物理放缩：方向不变，模长动态改变
                chunk_normed = chunk_normed * gate_view
                vector_channel_offset += mul
                
            out_chunks.append(chunk_normed.reshape(-1, mul * d))
            
        return torch.cat(out_chunks, dim=-1)  

class EquivariantMLP(torch.nn.Module):
    """
    基于 e3nn 标准组件重构的等变 MLP。
    流程: Linear -> Gate (Activation) -> Linear (已移除 BatchNorm)
    """
    def __init__(self, irreps_in, irreps_out, irreps_hidden=None):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)
        
        if irreps_hidden is None:
            irreps_hidden = irreps_in
        else:
            irreps_hidden = o3.Irreps(irreps_hidden)
            
        # --- 1. 构建 Gate 策略 ---
        hidden_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l == 0])
        hidden_vectors = o3.Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l > 0])
        hidden_gates = o3.Irreps([(mul, "0e") for mul, _ in hidden_vectors])
        
        self.gate = e3nn_nn.Gate(
            irreps_scalars=hidden_scalars, 
            act_scalars=[config.ACTIVATION_FUNCTION] * len(hidden_scalars),
            irreps_gates=hidden_gates,
            act_gates=[torch.sigmoid] * len(hidden_gates),
            irreps_gated=hidden_vectors
        )
        
        # --- 2. 定义层 ---
        self.linear1 = o3.Linear(irreps_in, self.gate.irreps_in)
        
        # 【火力全开】：启用你手写的 EquivariantLayerNorm 替代有毒的 BatchNorm！
        self.norm = GatedEquivariantLayerNorm(self.gate.irreps_in)  
        
        self.linear2 = o3.Linear(self.gate.irreps_out, irreps_out)

    def forward(self, x):
        x = self.linear1(x)
        
        # 稳定的归一化，保证深层 MLP 不会发生梯度消失或激活值饱和
        if self.norm is not None:
            x = self.norm(x)  
            
        x = self.gate(x)
        x = self.linear2(x)
        return x
# =============================================================================
# 2. Transformer Layer
# =============================================================================

class EquivariantTransformerLayer(torch.nn.Module):
    def __init__(self, irreps_node_in, irreps_node_out, irreps_sh):
        super().__init__()

        self.attn_scale = nn.Parameter(torch.tensor(1.0))
        self.dist_bias_weight = nn.Parameter(torch.tensor(1.0))

        # 1. 先定义不可约表示 (Irreps)
        self.irreps_node_in = o3.Irreps(irreps_node_in) 
        self.irreps_node_out = o3.Irreps(irreps_node_out) 
        self.irreps_sh = o3.Irreps(irreps_sh) 
        
        # 2. 然后立刻计算标量维度
        self.scalar_dim = self.irreps_node_out.count(o3.Irrep(0, 1))
        
        # 3. 再基于维度初始化 Norm 和 Dropout
        self.scalar_norm = nn.LayerNorm(self.scalar_dim) if self.scalar_dim > 0 else nn.Identity()
        self.scalar_dropout = nn.Dropout(config.SCALAR_DROPOUT) 
        self.density_proj = nn.Linear(1, self.scalar_dim) if self.scalar_dim > 0 else None
        
        # Q, K, V 投影
        self.linear_q = o3.Linear(self.irreps_node_in, self.irreps_node_in)
        self.tp_k = o3.FullyConnectedTensorProduct(self.irreps_node_in, self.irreps_sh, self.irreps_node_in, shared_weights=False)
        self.tp_v = o3.FullyConnectedTensorProduct(self.irreps_node_in, self.irreps_sh, self.irreps_node_in, shared_weights=False)
    
        combined_edge_dim = config.NUM_BASIS + config.EDGE_ATTR_DIM + 3
        
        # 边特征归一化层：专门压平 RBF 和相对坐标投影的狂野方差
        self.edge_emb_norm = nn.LayerNorm(combined_edge_dim)
        
        # 边属性 MLP
        self.fc_k = e3nn_nn.FullyConnectedNet([combined_edge_dim] + config.FC_NEURONS + [self.tp_k.weight_numel], act=config.ACTIVATION_FUNCTION)
        self.fc_v = e3nn_nn.FullyConnectedNet([combined_edge_dim] + config.FC_NEURONS + [self.tp_v.weight_numel], act=config.ACTIVATION_FUNCTION)

        # Attention Dot Product
        self.dot = o3.FullyConnectedTensorProduct(self.irreps_node_in, self.irreps_node_in, "1x0e")
        
        # Output Linear
        self.linear_out = o3.Linear(self.irreps_node_in, self.irreps_node_in)
        
        # FeedForward Network (FFN)
        self.ff_mlp = EquivariantMLP(
            irreps_in=self.irreps_node_in, 
            irreps_out=self.irreps_node_in, 
            irreps_hidden=self.irreps_node_in
        )
        
        # Update Heads
        self.coord_update_mlp = EquivariantMLP(self.irreps_node_in, o3.Irreps("1x1o"))
        self.frame_update_mlp = EquivariantMLP(self.irreps_node_in, o3.Irreps("3x1o"))
        self.final_mlp = EquivariantMLP(self.irreps_node_in, self.irreps_node_out)

    # 修改签名，接收外部计算好的几何特征变量，不再重复计算
    def forward(self, f_in, pos, ref_coords, edge_index, edge_attr, 
                edge_vec, edge_length_raw, edge_sh, edge_length_embedding, 
                is_last_layer=False):
        edge_src, edge_dst = edge_index
        
        # =================================================================
        # [AMP 精密控制区] 局部关闭 AMP，在 FP32 下处理几何防溢出
        # =================================================================
        with torch.amp.autocast('cuda', enabled=False):
            # 1. 只有特征张量需要提权，几何张量（edge_vec等）传入前已经是 FP32 了
            f_in_f32 = f_in.float()
            ref_coords_f32 = ref_coords.float()
            edge_attr_f32 = edge_attr.float()

            # 安全投影单位方向 (无论训练还是验证，统一截断防溢出)
            safe_edge_length = torch.clamp(edge_length_raw, min=1e-3)
            edge_dir = edge_vec / safe_edge_length.unsqueeze(-1)
                
            rel_pos_invariant = torch.einsum('ed, evd -> ev', edge_dir, ref_coords_f32[edge_src])
            combined_edge_emb = torch.cat([edge_length_embedding, edge_attr_f32, rel_pos_invariant], dim=-1)
            
            # 强行将拼接后的边特征归一化，切断梯度放大的乘数效应
            combined_edge_emb = self.edge_emb_norm(combined_edge_emb)
            
            # 3. 生成边权重，强制转回 FP32
            edge_scalars_k = self.fc_k(combined_edge_emb).float()
            edge_scalars_v = self.fc_v(combined_edge_emb).float()
            
            # 4. 核心注意力与张量积
            q_f32 = self.linear_q(f_in_f32)
            k_f32 = self.tp_k(f_in_f32[edge_src], edge_sh, edge_scalars_k)
            v_f32 = self.tp_v(f_in_f32[edge_src], edge_sh, edge_scalars_v)
            
            dot_scores = self.dot(q_f32[edge_dst], k_f32).squeeze(-1)
            scaling_factor = math.sqrt(self.irreps_node_in.dim) 
            
            # [核心修改 1]: Attention 缩放，融入可学习温度参数 attn_scale
            dot_scores = (dot_scores / scaling_factor) * self.attn_scale
            
            safe_dist_weight = F.softplus(self.dist_bias_weight)
            dot_scores = dot_scores - safe_dist_weight * edge_length_raw
            
            # 经过偏置惩罚后，只有真正近距离（或特征极其强烈）的节点，才能在 Softmax 中存活
            alpha = softmax(dot_scores, edge_dst, num_nodes=f_in_f32.shape[0])
            
            # 【关键修改 3】：聚合节点特征。
            f_agg = scatter(alpha.unsqueeze(-1) * v_f32, edge_dst, dim=0, reduce='sum', dim_size=f_in_f32.shape[0])

            # 【关键修改 4】：保留物理拥挤度特征 (density)，但**坚决删除** norm_factor 对 f_agg 的重复缩放
            num_nodes = f_in_f32.shape[0]
            ones = torch.ones_like(edge_dst, dtype=v_f32.dtype)
            degree = scatter(ones, edge_dst, dim=0, dim_size=num_nodes, reduce='sum')
            degree = degree.clamp(min=1.0)
            
            # 仅计算对数物理拥挤度特征，不再执行 f_agg = f_agg * (1.0 / sqrt(degree))
            density_feature = torch.log(degree).unsqueeze(-1)

        # =================================================================
        # ��️ [防线]: 离开 FP32 保护区，强制恢复原始数据类型
        # =================================================================
        f_agg = f_agg.to(f_in.dtype)
        density_feature = density_feature.to(f_in.dtype)
        
        # [核心修改 3]: 密度特征安全注入！
        if self.density_proj is not None:
            # 经过投影后，density_emb 的维度精确等于 scalar_dim
            density_emb = self.density_proj(density_feature)
            # 通过残差直接加到纯标量部分，完全不破坏旋转等变向量的几何性质
            f_agg[:, :self.scalar_dim] = f_agg[:, :self.scalar_dim] + density_emb
        
        # 5. 特征变换与更新头
        f_linear = self.linear_out(f_agg)
        f_activated = self.ff_mlp(f_linear)
        f_update = self.final_mlp(f_activated)
        
        # 截断机制：最后一层不计算坐标更新
        if not is_last_layer:
            pos_update = self.coord_update_mlp(f_activated).squeeze(1)
            frame_update = self.frame_update_mlp(f_activated).view(-1, 3, 3)
            pos_out = pos + pos_update
            ref_coords_out = ref_coords + frame_update
        else:
            pos_out = pos
            ref_coords_out = ref_coords
            
        # 6. 残差连接
        f_out = f_in + f_update
        
        # 7. 标量 LayerNorm 与 Dropout
        if self.scalar_dim > 0:
            # 拆分标量与向量
            scalars_out = f_out[:, :self.scalar_dim]
            vectors_out = f_out[:, self.scalar_dim:]
            
            # 第一道防线：不管训练还是验证，必须 LayerNorm
            scalars_out = self.scalar_norm(scalars_out)
            
            # 第二道防线：Dropout 仅在训练时生效
            if self.training:
                scalars_out = self.scalar_dropout(scalars_out)
                
            f_out = torch.cat([scalars_out, vectors_out], dim=-1)
            
        return f_out, pos_out, ref_coords_out

class EdgePredictor(nn.Module):
    """
    [新增] 图拓扑边预测器
    利用相连节点的标量特征和空间距离，预测它们在真实物理世界中的相互作用类型。
    """
    def __init__(self, node_scalar_dim, edge_basis_dim, fc_neurons, num_classes=6):
        super().__init__()
        in_dim = node_scalar_dim * 2 + edge_basis_dim
        
        layers = []
        current_dim = in_dim
        
        for hidden_dim in fc_neurons:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if isinstance(config.ACTIVATION_FUNCTION, type) and issubclass(config.ACTIVATION_FUNCTION, nn.Module):
                layers.append(config.ACTIVATION_FUNCTION())
            else:
                layers.append(nn.SiLU()) 
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, node_scalars, edge_index, edge_length_emb):
        src, dst = edge_index
        edge_feat = torch.cat([node_scalars[src], node_scalars[dst], edge_length_emb], dim=-1)
        logits = self.net(edge_feat)
        return logits
    
# =============================================================================
# 3. 主模型架构 (Main Model)
# =============================================================================

class E3NNTransformerDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Irreps Definitions ---
        self.TOTAL_HIDDEN_VEC = config.TOTAL_HIDDEN_VEC_CHANNELS
        self.REF_COORDS_HIDDEN = config.REF_COORDS_HIDDEN_CHANNELS
        self.FREE_HIDDEN = self.TOTAL_HIDDEN_VEC - self.REF_COORDS_HIDDEN
        
        self.irreps_hidden = o3.Irreps(f"{config.HIDDEN_SCALAR_CHANNELS}x0e + {self.TOTAL_HIDDEN_VEC}x1o")
        self.irreps_ref_coords_out = o3.Irreps(f"{self.REF_COORDS_HIDDEN}x1o")
        self.irreps_free_vectors = o3.Irreps(f"{self.FREE_HIDDEN}x1o")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        
        self.irreps_out_noise_feat = o3.Irreps(f"{config.LIGAND_FEATURE_DIM}x0e")
        self.irreps_out_noise_pos = o3.Irreps("1x1o")
        self.irreps_out_noise_frame = o3.Irreps("3x1o")

        # --- Embedding Layers ---
        self.time_emb_mlp = e3nn_nn.FullyConnectedNet([config.TIME_EMB_DIM] + config.FC_NEURONS, act=config.ACTIVATION_FUNCTION)
        self.embed_norm = nn.LayerNorm(config.EMBEDDING_DIM_IN)
        scalar_input_dim_processed = config.EMBEDDING_DIM_IN + config.CHEM_PROPS_DIM_IN + config.TYPE_ENCODING_DIM_IN
        scalar_input_dim_total = scalar_input_dim_processed + config.FC_NEURONS[-1]
        
        irreps_scalar_in = o3.Irreps(f"{scalar_input_dim_total}x0e")
        irreps_scalar_out = o3.Irreps(f"{config.HIDDEN_SCALAR_CHANNELS}x0e") + self.irreps_free_vectors
        self.scalar_embedding = o3.Linear(irreps_scalar_in, irreps_scalar_out)
        
        irreps_ref_coords_in = o3.Irreps("3x1o") 
        self.ref_coords_embedding = o3.Linear(irreps_ref_coords_in, self.irreps_ref_coords_out)

        # --- Transformer Layers ---
        self.transformer_layers = torch.nn.ModuleList([
            EquivariantTransformerLayer(self.irreps_hidden, self.irreps_hidden, self.irreps_sh) 
            for _ in range(config.NUM_TRANSFORMER_LAYERS)
        ])

        del self.transformer_layers[-1].coord_update_mlp
        del self.transformer_layers[-1].frame_update_mlp
        
        # --- Output Heads ---
        self.noise_feat_head = EquivariantMLP(self.irreps_hidden, self.irreps_out_noise_feat)
        self.noise_pos_head = EquivariantMLP(self.irreps_hidden, self.irreps_out_noise_pos)
        self.noise_frame_head = EquivariantMLP(self.irreps_hidden, self.irreps_out_noise_frame)

        self.edge_predictor = EdgePredictor(
            node_scalar_dim=config.HIDDEN_SCALAR_CHANNELS,
            edge_basis_dim=config.NUM_BASIS,
            fc_neurons=config.EDGE_PREDICTOR_NEURONS,
            num_classes=6 
        )

    def _compute_geometry(self, pos, edge_index, edge_attr):
        """统一计算几何特征 (FP32 保护下)，供拓扑边预测和各 Transformer 层共享"""
        with torch.amp.autocast('cuda', enabled=False):
            pos_f32 = pos.float()
            edge_src, edge_dst = edge_index
            
            edge_vec = pos_f32[edge_dst] - pos_f32[edge_src]
            edge_squared = torch.sum(edge_vec ** 2, dim=-1)
            edge_length_raw = torch.sqrt(edge_squared + 1e-8)
            
            # 计算球谐函数 (Spherical Harmonics)
            edge_sh = torch.zeros(edge_vec.shape[0], self.irreps_sh.dim, device=edge_vec.device, dtype=torch.float32)
            valid_edges_mask = edge_squared > 1e-6
            if valid_edges_mask.any():
                edge_sh[valid_edges_mask] = o3.spherical_harmonics(
                    self.irreps_sh, edge_vec[valid_edges_mask], normalize=True, normalization='component'
                )
                
            edge_length_embedding = soft_one_hot_linspace(
                edge_length_raw, start=0.0, end=config.MAX_EDGE_LENGTH, 
                number=config.NUM_BASIS, basis='smooth_finite', cutoff=True
            ).mul(config.NUM_BASIS**0.5)
            
            is_dummy_edge = edge_attr[:, -1] == 1.0
            if is_dummy_edge.any():
                continuous_dist = torch.log1p(edge_length_raw[is_dummy_edge]) 
                edge_length_embedding = edge_length_embedding.clone()
                edge_length_embedding[is_dummy_edge, -1] = continuous_dist
                
        return edge_vec, edge_length_raw, edge_sh, edge_length_embedding

    def forward(self, data, timestep):
        # ==================== Tensor Health Check ====================
        check_tensor_health(data.x, "data.x", "E3NN.forward - Inputs")
        check_tensor_health(data.pos, "data.pos", "E3NN.forward - Inputs")
        check_tensor_health(data.ref_coords, "data.ref_coords", "E3NN.forward - Inputs")
        
        # --- 1. Scalar Feature Preparation ---

        raw_embeds = self.embed_norm(data.x[:, :config.EMBEDDING_DIM_IN])
        raw_chem_props = data.x[:, config.EMBEDDING_DIM_IN : config.EMBEDDING_DIM_IN + config.CHEM_PROPS_DIM_IN]
        type_encoding = data.x[:, config.EMBEDDING_DIM_IN + config.CHEM_PROPS_DIM_IN:]

        # [修改] 直接使用纯净的 raw_chem_props 进行拼接
        processed_features = torch.cat([
            raw_embeds, 
            raw_chem_props,
            type_encoding
        ], dim=1)
        
        t_emb = self._timestep_embedding(timestep, config.TIME_EMB_DIM, data.batch.device)
        scalar_features_in = torch.cat([processed_features, self.time_emb_mlp(t_emb[data.batch])], dim=1)
        
        ref_coords_in = data.ref_coords.reshape(data.num_nodes, -1)
        ref_coords_embedded = self.ref_coords_embedding(ref_coords_in)
        check_tensor_health(ref_coords_embedded, "ref_coords_embedded", "E3NN.forward - Post Embedding")
        
        scalar_embedded = self.scalar_embedding(scalar_features_in)
        
        # --- 2. Construct Hidden State ---
        num_scalar_hidden = config.HIDDEN_SCALAR_CHANNELS
        hidden_scalars = scalar_embedded[:, :num_scalar_hidden]
        free_vectors = scalar_embedded[:, num_scalar_hidden:]
        
        f = torch.cat([hidden_scalars, ref_coords_embedded, free_vectors], dim=-1)
        
        # --- 3. Initial Geometry Computation ---
        # 【核心修订 2】：提取真实的 edge_attr 并传入 _compute_geometry
        injected_edge_attr = data.edge_attr.clone()
        edge_vec, edge_length_raw, edge_sh, edge_length_embedding = self._compute_geometry(data.pos, data.edge_index, injected_edge_attr)

        # --- 4. Transformer Loop ---
        pos, ref_coords = data.pos, data.ref_coords       
        num_layers = len(self.transformer_layers)
        
        for i, layer in enumerate(self.transformer_layers):
            is_last = (i == num_layers - 1)
            
            if self.training:
                f, pos, ref_coords = checkpoint(
                    layer, 
                    f, pos, ref_coords, data.edge_index, injected_edge_attr,
                    edge_vec, edge_length_raw, edge_sh, edge_length_embedding,
                    is_last,
                    use_reentrant=False 
                )
            else:
                f, pos, ref_coords = layer(
                    f, pos, ref_coords, data.edge_index, injected_edge_attr, 
                    edge_vec, edge_length_raw, edge_sh, edge_length_embedding,
                    is_last 
                )
                
            check_tensor_health(f, f"f_layer_{i}", "Transformer Loop")
            
            if not is_last:
                # 【核心修订 2】：同步补充 injected_edge_attr
                edge_vec, edge_length_raw, edge_sh, edge_length_embedding = self._compute_geometry(pos, data.edge_index, injected_edge_attr)
                
        # =================================================================
        # --- 5. Global Receptive Field Edge Prediction (显存优化重构) ---
        # =================================================================
        final_node_scalars = f[:, :config.HIDDEN_SCALAR_CHANNELS]
        
        # 【核心手术 3】：精准切除 PP 边 和 Dummy 边，挽救显存且防止乱预测
        src, dst = data.edge_index
        is_pp_edge = (~data.is_ligand[src]) & (~data.is_ligand[dst])
        
        # 识别 Dummy 边 (只要端点有一个是 dummy，就是辅助边)
        is_dummy_node = getattr(data, 'is_dummy', torch.zeros_like(data.is_ligand))
        is_dummy_edge = is_dummy_node[src] | is_dummy_node[dst]

        valid_edge_mask = (~is_pp_edge) & (~is_dummy_edge)  

        filtered_edge_index = data.edge_index[:, valid_edge_mask]
        filtered_edge_length_emb = edge_length_embedding[valid_edge_mask]

        if filtered_edge_index.shape[1] > 0:
            edge_logits = self.edge_predictor(final_node_scalars, filtered_edge_index, filtered_edge_length_emb)
        else:
            edge_logits = torch.empty((0, 6), device=pos.device)

        # --- 6. Prediction Heads ---
        ligand_features_hidden = f[data.is_ligand]
        predicted_noise_feat = self.noise_feat_head(ligand_features_hidden)
        predicted_noise_pos = self.noise_pos_head(ligand_features_hidden).squeeze(1)
        predicted_noise_frame = self.noise_frame_head(ligand_features_hidden).view(-1, 3, 3)
        
        check_tensor_health(predicted_noise_feat, "out_noise_feat", "Final Output")
        
        # 【关键修改】：必须把过滤后的边索引也传出去，否则 Loss 函数不知道 logits 对应谁
        return predicted_noise_feat, predicted_noise_pos, predicted_noise_frame, edge_logits, filtered_edge_index

    @staticmethod
    def _timestep_embedding(timesteps, embedding_dim, device):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

if __name__ == '__main__':
    print("Testing Updated Model (Standard e3nn components)...")
    try:
        model = E3NNTransformerDiffusion()
        print("Model Initialized Successfully.")
        print(f"MLP Gate Structure: {model.noise_feat_head.gate}")
        print(f"MLP Norm Structure: {model.noise_feat_head.norm}")
    except Exception as e:
        print(f"Initialization Failed: {e}")