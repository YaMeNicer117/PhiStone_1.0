import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset
import warnings
import concurrent.futures
import torch.nn.functional as F

# Local Imports
import PhiSSE3TD_config as config

def is_main_process():
    return int(os.environ.get('RANK', 0)) == 0

# =============================================================================
# 1. 核心数据变换 (Transform)
# =============================================================================
class LazyGraphDataset(TorchDataset):
    """
    针对大规模 3D 分子图数据的懒加载数据集。
    只保存文件路径列表，在 __getitem__ 时才实时读取和预处理数据。
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 扫描目录下所有的 .pt 文件
        search_pattern = os.path.join(data_dir, '**', '*.pt')
        self.file_list = sorted(glob.glob(search_pattern, recursive=True))
        
        if not self.file_list:
            raise FileNotFoundError(f"在 {data_dir} 中未找到任何 .pt 文件。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 稳健性升级：设置最大重试次数，防止无限死循环
        max_retries = 10 
        current_idx = idx
        
        for attempt in range(max_retries):
            f_path = self.file_list[current_idx]
            try:
                # 实时从硬盘加载数据
                data = torch.load(f_path, weights_only=False)
                
                # 实时应用预处理转换 (如拼接特征、计算相对坐标)
                if self.transform is not None:
                    data = self.transform(data)
                    
                return data
                
            except Exception as e:
                # 附带当前进程的 Rank，方便在多卡日志中精准定位是哪张卡在报错
                rank = os.environ.get('RANK', '0')
                warnings.warn(f"[Rank {rank}] 读取文件 {f_path} 失败 (尝试 {attempt+1}/{max_retries})。错误信息: {e}")
                
                # 【核心修复】：不再使用顺序递增 (current_idx + 1)
                # 改用全局随机重采样。这样可以立即跳出连续损坏的数据区块，
                # 且重复采样一两个样本对全局梯度的影响微乎其微，完美保证 DDP 同步不断流。
                current_idx = int(torch.randint(0, len(self.file_list), (1,)).item())
                
        # 如果连续失败达到上限，果断抛出严重异常，防止程序僵死
        raise RuntimeError(f"致命错误：连续 {max_retries} 次加载数据失败！请检查硬盘状态或数据集 {self.data_dir} 是否大面积损坏。")

class PreprocessNodeFeatures(T.BaseTransform):
    """
    [原模块一] 自定义的 PyG Transform。
    功能：
    1. 重构节点特征：拼接 [Embeddings, ChemProps, TypeEncoding]。
    2. 坐标相对化：将 ref_coords 转换为相对于节点 pos 的相对坐标。
    3. 创建 is_ligand 掩码。
    """
    def __call__(self, data):
        # 1. 解析原始特征 (旧数据只有 2 维 Type，需要取前两维)
        raw_type = data.x[:, :2]      
        chem_props = data.x[:, 2:2+config.CHEM_PROPS_DIM_IN]         

        # 2. 扩展为 3 维 Type (新增一维留给 Dummy)
        type_encoding = F.pad(raw_type, (0, 1), "constant", 0.0)

        # 3. 重构新的真实节点特征
        new_x = torch.cat([data.frag_embeds, chem_props, type_encoding], dim=1)
        
        # 4. 相对坐标转换
        center_pos = data.pos.unsqueeze(1) 
        new_ref = data.ref_coords - center_pos
        
        # ===========================================================
        # --- 5. 注入虚拟锚点 (Dummy Node) ---
        # ===========================================================
        device = data.x.device
        dummy_embed = torch.zeros((1, config.EMBEDDING_DIM_IN), device=device)
        dummy_chem = torch.zeros((1, config.CHEM_PROPS_DIM_IN), device=device)
        dummy_type = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        dummy_x = torch.cat([dummy_embed, dummy_chem, dummy_type], dim=1)
        
        # 原点坐标与空参考系
        dummy_pos = torch.zeros((1, 3), device=data.pos.device)
        dummy_ref = torch.zeros((1, 3, 3), device=data.pos.device)
        
        # 拼接到原图末尾
        data.x = torch.cat([new_x, dummy_x], dim=0)
        data.pos = torch.cat([data.pos, dummy_pos], dim=0)
        data.ref_coords = torch.cat([new_ref, dummy_ref], dim=0)
        
        # --- 6. 掩码更新 ---
        orig_is_ligand = (type_encoding[:, 0] == 1.0)
        data.is_ligand = torch.cat([orig_is_ligand, torch.tensor([False], device=device)], dim=0)
        
        # 新增专门的 is_dummy 掩码，方便后续过滤
        orig_is_dummy = torch.zeros(orig_is_ligand.size(0), dtype=torch.bool, device=device)
        data.is_dummy = torch.cat([orig_is_dummy, torch.tensor([True], device=device)], dim=0)

        # HAC 与原数据清理
        if not hasattr(data, 'hac'):
            orig_hac = torch.full((orig_is_ligand.size(0),), 20.0, dtype=torch.float32, device=device)
        else:
            orig_hac = data.hac.float()
        data.hac = torch.cat([orig_hac, torch.tensor([0.0], device=device)], dim=0)

        if hasattr(data, 'frag_embeds'):
            del data.frag_embeds
            
        # ===========================================================
        # --- 7. GT 边属性升维保护 ---
        # ===========================================================
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # 兼容老数据，将 5 维边属性扩展为 6 维
            if data.edge_attr.shape[1] == 5:
                data.edge_attr = F.pad(data.edge_attr, (0, 1), "constant", 0.0)
                
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.gt_edge_index = data.edge_index.clone()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.gt_edge_attr = data.edge_attr.clone()

        # 强行同步图节点数
        data.num_nodes = data.x.shape[0]

        return data

# 实例化全局 Transform 对象
preprocess_transform = PreprocessNodeFeatures()

class PreloadedGraphDataset(TorchDataset):
    """
    [极速全内存版] 将所有数据及预处理结果一次性加载到内存中。
    彻底消除每个 Epoch 训练时的磁盘 I/O 瓶颈和 CPU 预处理开销。
    """
    def __init__(self, data_dir, transform=None, max_workers=16):
        self.data_dir = data_dir
        self.transform = transform
        
        # 扫描目录下所有的 .pt 文件
        search_pattern = os.path.join(data_dir, '**', '*.pt')
        self.file_list = sorted(glob.glob(search_pattern, recursive=True))
        
        if not self.file_list:
            raise FileNotFoundError(f"在 {data_dir} 中未找到任何 .pt 文件。")

        if is_main_process():
            print(f"\n🚀 开始将 {len(self.file_list)} 个图数据预加载到物理内存...")
            print("注意：多卡 DDP 模式下，每个 GPU 进程会持有一份内存副本。")

        self.data_list = []
        
        # 定义单文件加载与预处理函数
        def load_and_transform(f_path):
            try:
                # 1. 从硬盘读取
                data = torch.load(f_path, weights_only=False)
                # 2. 提前在 CPU 上完成 Transform (特征拼接、相对坐标计算等)
                if self.transform is not None:
                    data = self.transform(data)
                return data
            except Exception as e:
                warnings.warn(f"读取或处理 {f_path} 失败，已跳过。错误: {e}")
                return None

        # 使用多线程极速并发读取
        disable_tqdm = not is_main_process()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务并显示进度条
            results = list(tqdm(
                executor.map(load_and_transform, self.file_list),
                total=len(self.file_list),
                desc="[RAM Loading]",
                disable=disable_tqdm
            ))
            
        # 过滤掉加载失败的 None 数据
        self.data_list = [d for d in results if d is not None]
        
        if is_main_process():
            print(f"✅ 预加载完成！成功驻留 {len(self.data_list)} 个图数据至内存。\n")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 训练时，直接以 O(1) 的光速从内存返回已经预处理好的对象
        return self.data_list[idx]     

# =============================================================================
# 2. 推理阶段专用预处理函数 (Inference Preprocessing)
# =============================================================================
def split_complex_data(complex_data):
    """
    [修订版] 安全拆分复合物，确保虚拟锚点被正确分配给口袋，而不是配体。
    """
    from torch_geometric.data import Data
    
    n_p = complex_data.n_pocket_nodes
    n_total = complex_data.num_nodes

    # 检查末尾是否被注入了 Dummy 节点
    has_dummy = getattr(complex_data, 'is_dummy', None) is not None and complex_data.is_dummy[-1].item()
    if has_dummy:
        n_ligand_end = n_total - 1
        dummy_x = complex_data.x[-1:]
        dummy_pos = complex_data.pos[-1:]
        dummy_embeds = complex_data.frag_embeds[-1:]
        dummy_ref = complex_data.ref_coords[-1:]
    else:
        n_ligand_end = n_total
        
    # --- 拆分节点属性 ---
    # 1. 提取纯净的口袋节点
    pocket_x = complex_data.x[:n_p]
    pocket_pos = complex_data.pos[:n_p]
    pocket_frag_embeds = complex_data.frag_embeds[:n_p]
    pocket_ref_coords = complex_data.ref_coords[:n_p]

    # 2. 提取纯净的配体节点 (排除末尾可能的 Dummy)
    ligand_x = complex_data.x[n_p:n_ligand_end]
    ligand_pos = complex_data.pos[n_p:n_ligand_end]
    ligand_frag_embeds = complex_data.frag_embeds[n_p:n_ligand_end]
    ligand_ref_coords = complex_data.ref_coords[n_p:n_ligand_end]

    # 3. 将 Dummy 节点强行归档到 Pocket 中
    if has_dummy:
        pocket_x = torch.cat([pocket_x, dummy_x], dim=0)
        pocket_pos = torch.cat([pocket_pos, dummy_pos], dim=0)
        pocket_frag_embeds = torch.cat([pocket_frag_embeds, dummy_embeds], dim=0)
        pocket_ref_coords = torch.cat([pocket_ref_coords, dummy_ref], dim=0)
        
    # --- 拆分边 (由于 Dummy 边是动态生成的，GT里没有，所以边拆分逻辑保持不变) ---
    edge_index = complex_data.edge_index
    edge_attr = complex_data.edge_attr
    if edge_attr is not None and edge_attr.shape[1] == 5:
        import torch.nn.functional as F
        edge_attr = F.pad(edge_attr, (0, 1), "constant", 0.0)

    pp_mask = (edge_index[0] < n_p) & (edge_index[1] < n_p)
    pp_edge_index = edge_index[:, pp_mask]
    pp_edge_attr = edge_attr[pp_mask]

    ll_mask = (edge_index[0] >= n_p) & (edge_index[1] >= n_p)
    ll_edge_index = edge_index[:, ll_mask] - n_p 
    ll_edge_attr = edge_attr[ll_mask]

    fixed_mask = (edge_index[0] >= n_p) | (edge_index[1] >= n_p)
    fixed_edge_index = edge_index[:, fixed_mask]
    fixed_edge_attr = edge_attr[fixed_mask]

    # --- 组装对象 ---
    pocket_data = Data(
        x=pocket_x, pos=pocket_pos,
        frag_embeds=pocket_frag_embeds, ref_coords=pocket_ref_coords,
        edge_index=pp_edge_index, edge_attr=pp_edge_attr,
        pdb_id=complex_data.pdb_id, meb_center=complex_data.meb_center
    )
    # 继承 Dummy 掩码
    if has_dummy:
        pocket_data.is_dummy = torch.cat([torch.zeros(n_p, dtype=torch.bool), torch.tensor([True])], dim=0)
        pocket_data.is_ligand = torch.zeros(n_p + 1, dtype=torch.bool)
    else:
        pocket_data.is_dummy = torch.zeros(n_p, dtype=torch.bool)
        pocket_data.is_ligand = torch.zeros(n_p, dtype=torch.bool)

    ligand_data = Data(
        x=ligand_x, pos=ligand_pos,
        frag_embeds=ligand_frag_embeds, ref_coords=ligand_ref_coords,
        edge_index=ll_edge_index, edge_attr=ll_edge_attr,
        pdb_id=complex_data.pdb_id, meb_center=complex_data.meb_center,
        is_ligand=torch.ones(n_ligand_end - n_p, dtype=torch.bool),
        is_dummy=torch.zeros(n_ligand_end - n_p, dtype=torch.bool)
    )

    return pocket_data, ligand_data, {'edge_index': fixed_edge_index, 'edge_attr': fixed_edge_attr}

def preprocess_pocket_data(protein_pocket_data):
    pocket_clone = protein_pocket_data.clone()

    # --- 第一部分：修复 data.x 并扩展 Type ---
    if hasattr(pocket_clone, 'frag_embeds'):
        raw_type = pocket_clone.x[:, :2]
        chem_props = pocket_clone.x[:, 2:]
        import torch.nn.functional as F
        type_encoding = F.pad(raw_type, (0, 1), "constant", 0.0)
        pocket_clone.x = torch.cat([pocket_clone.frag_embeds, chem_props, type_encoding], dim=1)
        del pocket_clone.frag_embeds
        
    if hasattr(pocket_clone, 'edge_attr') and pocket_clone.edge_attr is not None:
        if pocket_clone.edge_attr.shape[1] == 5:
            import torch.nn.functional as F
            pocket_clone.edge_attr = F.pad(pocket_clone.edge_attr, (0, 1), "constant", 0.0)
    
    # --- 第二部分：将 ref_coords 转换为相对坐标 ---
    if hasattr(pocket_clone, 'ref_coords') and hasattr(pocket_clone, 'pos'):
        center_pos = pocket_clone.pos.unsqueeze(1)
        pocket_clone.ref_coords = pocket_clone.ref_coords.to(center_pos.device) - center_pos
    
    # --- 第三部分：设置掩码 ---
    orig_num_nodes = pocket_clone.x.shape[0]
    pocket_clone.is_ligand = torch.zeros(orig_num_nodes, dtype=torch.bool, device=pocket_clone.x.device)
    pocket_clone.is_dummy = torch.zeros(orig_num_nodes, dtype=torch.bool, device=pocket_clone.x.device)

    # ===========================================================
    # --- 第四部分：注入虚拟锚点 (Dummy Node) ---
    # ===========================================================
    device = pocket_clone.x.device
    dummy_embed = torch.zeros((1, config.EMBEDDING_DIM_IN), device=device)
    dummy_chem = torch.zeros((1, config.CHEM_PROPS_DIM_IN), device=device)
    dummy_type = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    dummy_x = torch.cat([dummy_embed, dummy_chem, dummy_type], dim=1)
    
    dummy_pos = torch.zeros((1, 3), device=pocket_clone.pos.device)
    dummy_ref = torch.zeros((1, 3, 3), device=pocket_clone.pos.device)
    
    pocket_clone.x = torch.cat([pocket_clone.x, dummy_x], dim=0)
    pocket_clone.pos = torch.cat([pocket_clone.pos, dummy_pos], dim=0)
    if hasattr(pocket_clone, 'ref_coords'):
        pocket_clone.ref_coords = torch.cat([pocket_clone.ref_coords, dummy_ref], dim=0)
    
    pocket_clone.is_ligand = torch.cat([pocket_clone.is_ligand, torch.tensor([False], device=device)], dim=0)
    pocket_clone.is_dummy = torch.cat([pocket_clone.is_dummy, torch.tensor([True], device=device)], dim=0)
    
    pocket_clone.num_nodes = pocket_clone.x.shape[0]

    return pocket_clone

def preprocess_fixed_ligand_data(fixed_ligand_data):
    data = fixed_ligand_data.clone()
    num_nodes = data.num_nodes
    
    # 提取特征
    frag_embeds = data.frag_embeds
    
    # 因为旧版的 raw_type 被抛弃，我们需要适配维度
    if data.x.size(1) == 2 + config.CHEM_PROPS_DIM_IN: 
        chem_props = data.x[:, 2:]
    else:
        chem_props = data.x[:, :config.CHEM_PROPS_DIM_IN] 
    
    # 【核心修改】：统一配体类型编码为 3 维 [1.0, 0.0, 0.0]
    type_encoding = torch.tensor([1.0, 0.0, 0.0], device=data.x.device).repeat(num_nodes, 1)
    
    data.x = torch.cat([frag_embeds, chem_props, type_encoding], dim=1)
    
    # 坐标处理
    has_valid_pos = hasattr(data, 'pos') and data.pos is not None and data.pos.numel() > 0
    if has_valid_pos:
        if hasattr(data, 'ref_coords'):
            center_pos = data.pos.unsqueeze(1)
            data.ref_coords = data.ref_coords - center_pos
    else:
        data.pos = torch.zeros((num_nodes, 3), device=data.x.device)
        data.ref_coords = torch.zeros((num_nodes, 3, 3), device=data.x.device)
    
    # 掩码设置
    data.is_ligand = torch.ones(num_nodes, dtype=torch.bool, device=data.x.device)
    data.is_dummy = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    
    if hasattr(data, 'frag_embeds'):
        del data.frag_embeds
        
    return data

# =============================================================================
# 3. 数据集加载与划分 (DataLoader Factory)
# =============================================================================
def get_train_val_dataloaders(data_dir=config.PYG_DATA_DIR, 
                              batch_size=config.BATCH_SIZE, 
                              seed=config.SEED, 
                              distributed=False):
    if is_main_process():
        mode_str = "Lazy Loading (硬盘动态读取)" if config.USE_LAZY_DATASET else "Preloaded (内存常驻加速)"
        print(f"\n--- 开始初始化数据集 [{mode_str}] ---")
        print(f"数据源: {data_dir}")
    
    # 1. 根据全局配置，实例化对应的数据集
    if config.USE_LAZY_DATASET:
        full_dataset = LazyGraphDataset(data_dir=data_dir, transform=preprocess_transform)
    else:
        full_dataset = PreloadedGraphDataset(data_dir=data_dir, transform=preprocess_transform)
    
    if is_main_process():
        print(f"数据集初始化完成，共扫描到 {len(full_dataset)} 个复合物文件。")

    # 2. 按比例划分训练集和验证集
    num_data = len(full_dataset)
    num_train = int(num_data * config.TRAIN_VAL_SPLIT)
    num_val = num_data - num_train

    # 使用固定的 Generator 确保所有 GPU 上的划分一致
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(seed)
    )

    if is_main_process():
        print(f"数据集划分完成：训练集 {len(train_dataset)} 样本，验证集 {len(val_dataset)} 样本。")

    # =========================================================================
    # DDP Sampler 设置
    # =========================================================================
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    # =========================================================================
    # 创建 DataLoader
    # =========================================================================
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=train_shuffle, 
        sampler=train_sampler, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY,   
        follow_batch=['pos', 'ref_coords']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=val_sampler,   
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        follow_batch=['pos', 'ref_coords']
    )
    
    return train_loader, val_loader, train_sampler


def get_test_dataloader(data_dir=config.PYG_TEST_DATA_DIR, 
                        batch_size=config.BATCH_SIZE, 
                        distributed=False):
    if not os.path.exists(data_dir):
        if is_main_process():
            print(f"提示: 测试数据目录不存在: {data_dir}")
        return None

    if is_main_process():
        print(f"\n--- 开始初始化测试数据集 ---")
        
    try:
        if config.USE_LAZY_DATASET:
            test_dataset = LazyGraphDataset(data_dir=data_dir, transform=preprocess_transform)
        else:
            test_dataset = PreloadedGraphDataset(data_dir=data_dir, transform=preprocess_transform)
    except FileNotFoundError:
        if is_main_process():
            print(f"提示: 测试数据目录为空: {data_dir}")
        return None

    if is_main_process():
        print(f"测试集初始化完成，共 {len(test_dataset)} 个样本。")

    if distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = None

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=test_sampler, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        follow_batch=['pos', 'ref_coords']
    )
    
    return test_loader

if __name__ == '__main__':
    # 简单的测试逻辑
    try:
        train_l, val_l, _ = get_train_val_dataloaders()
        batch = next(iter(train_l))
        print(f"\nBatch Test Success:")
        print(f"Node Features: {batch.x.shape}")
        print(f"Positions: {batch.pos.shape}")
        print(f"Is Ligand: {batch.is_ligand.sum()}")
    except Exception as e:
        print(f"Test failed (Expected if data path is empty): {e}")