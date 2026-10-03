# PhiSGATv2_model.py (修订版: 增加边属性支持与GlobalAttention池化)

import torch
import torch.nn.functional as F
from torch import nn
# [新增] 导入 GlobalAttention
from torch_geometric.nn import GATv2Conv, GlobalAttention 

# 导入配置文件
import PhiSGATv2_config as config


class GATv2Model(nn.Module):
    """
    基于 GATv2 的图神经网络模型 (支持边属性 & Global Attention Pooling)。
    """
    def __init__(self):
        super(GATv2Model, self).__init__()
        
        self.dropout = config.DROPOUT
        self.num_tasks = config.NUM_TASKS
        
        # ----- 1. 输入特征维度 -----
        self.combined_input_dim = config.INPUT_DIM_NUMERIC + config.EMBEDDING_DIM

        # ----- 2. GATv2 特征提取模块 (修订: 加入 edge_dim) -----
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层 GATv2
        self.convs.append(
            GATv2Conv(
                self.combined_input_dim, 
                config.HIDDEN_CHANNELS, 
                heads=config.HEADS, 
                concat=True,
                edge_dim=config.EDGE_DIM  # <--- [核心修改] 传入边特征维度
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(config.HIDDEN_CHANNELS * config.HEADS))

        # 中间层 GATv2
        for _ in range(config.NUM_LAYERS - 2):
            conv = GATv2Conv(
                config.HIDDEN_CHANNELS * config.HEADS, 
                config.HIDDEN_CHANNELS, 
                heads=config.HEADS, 
                concat=True,
                edge_dim=config.EDGE_DIM  # <--- [核心修改] 传入边特征维度
            )
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(config.HIDDEN_CHANNELS * config.HEADS))

        # 最后一层 GATv2
        self.convs.append(
            GATv2Conv(
                config.HIDDEN_CHANNELS * config.HEADS, 
                config.HIDDEN_CHANNELS, 
                heads=config.HEADS, 
                concat=True,
                edge_dim=config.EDGE_DIM  # <--- [核心修改] 传入边特征维度
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(config.HIDDEN_CHANNELS * config.HEADS))
        
        # ----- 3. 全局注意力池化模块 (Global Attention Pooling) -----
        # 计算 GAT 最后一层的输出维度
        final_node_dim = config.HIDDEN_CHANNELS * config.HEADS
        
        # 定义注意力门控网络 (Gate Neural Network)
        # 它为每个节点计算一个权重分数
        gate_nn = nn.Sequential(
            nn.Linear(final_node_dim, 1)
        )
        self.pool = GlobalAttention(gate_nn=gate_nn) # <--- [核心修改] 替换原来的 global_mean_pool

        # ----- 4. MLP 回归输出模块 -----
        self.mlp = nn.Sequential(
            nn.Linear(final_node_dim, config.HIDDEN_CHANNELS),
            nn.SiLU(), 
            nn.Dropout(p=self.dropout),
            nn.Linear(config.HIDDEN_CHANNELS, self.num_tasks) 
        )

    def forward(self, data):
        """
        定义模型的前向传播逻辑。
        """
        # [修订] 从 data 中额外提取 edge_attr
        required_attrs = ['x', 'edge_index', 'frag_embeds']
        if config.EDGE_DIM > 0:
            required_attrs.append('edge_attr')
        missing = [
            name for name in required_attrs
            if not hasattr(data, name)
        ]
        if missing:
            raise ValueError(f"Missing required graph attributes: {missing}")

        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(
            data,
            'batch',
            torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        frag_embeds = data.frag_embeds.float()

        if x.dim() != 2 or x.size(-1) != config.INPUT_DIM_NUMERIC:
            raise ValueError(
                f"data.x must have shape [num_nodes, {config.INPUT_DIM_NUMERIC}], "
                f"got {tuple(x.shape)}"
            )
        if frag_embeds.dim() != 2 or frag_embeds.size(-1) != config.EMBEDDING_DIM:
            raise ValueError(
                f"data.frag_embeds must have shape [num_nodes, {config.EMBEDDING_DIM}], "
                f"got {tuple(frag_embeds.shape)}"
            )
        if frag_embeds.size(0) != x.size(0):
            raise ValueError(
                "data.x and data.frag_embeds must describe the same number of nodes, "
                f"got {x.size(0)} and {frag_embeds.size(0)}"
            )
        
        # 确保 edge_attr 是浮点型 (如果是整数类型的 one-hot，必须转为 float)
        if edge_attr is not None:
            edge_attr = edge_attr.float()
            if edge_attr.dim() == 1:
                if edge_attr.numel() != edge_index.size(1) * config.EDGE_DIM:
                    raise ValueError(
                        f"1D edge_attr cannot be reshaped to [num_edges, {config.EDGE_DIM}], "
                        f"got {tuple(edge_attr.shape)} for {edge_index.size(1)} edges"
                    )
                edge_attr = edge_attr.view(edge_index.size(1), config.EDGE_DIM)
            if edge_attr.dim() != 2 or edge_attr.size(-1) != config.EDGE_DIM:
                raise ValueError(
                    f"edge_attr must have shape [num_edges, {config.EDGE_DIM}], "
                    f"got {tuple(edge_attr.shape)}"
                )
            if edge_attr.size(0) != edge_index.size(1):
                raise ValueError(
                    "edge_attr and edge_index disagree on edge count, "
                    f"got {edge_attr.size(0)} and {edge_index.size(1)}"
                )

        # --- 步骤 1: 拼接原始数值特征与片段嵌入 ---
        x = torch.cat([x, frag_embeds], dim=-1)

        # --- 步骤 2: GATv2 层信息传递 (携带边信息) ---
        for conv, bn in zip(self.convs, self.batch_norms):
            # [核心修改] 在卷积计算中传入 edge_attr
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.silu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # --- 步骤 3: 全局注意力池化 ---
        # [核心修改] 使用 GlobalAttention 进行聚合
        # 模型会自动学习每个原子的重要性权重
        x = self.pool(x, batch)

        # --- 步骤 4: MLP 回归输出 ---
        x = self.mlp(x)

        # 输出形状: (batch_size, num_tasks)，每个任务一个连续预测值
        return x
