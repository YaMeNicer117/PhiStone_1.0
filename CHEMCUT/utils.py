# =============================================================================
#
#                               utils.py
#
# =============================================================================

# ----------------------------------------
# 导入所需的模块
# ----------------------------------------

# RDKit
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import ChemicalFeatures as Feat
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit import Chem, RDConfig, RDLogger

# Python Standard & Third-Party Libraries
import os
import math
import warnings
from itertools import combinations
import numpy as np
import pandas as pd

# 忽略 NumPy 的空切片警告，保持控制台绝对清爽
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# IPython (可选，用于Jupyter环境)
from IPython.display import display, HTML, clear_output


# ----------------------------------------
# 定义辅助函数
# ----------------------------------------

def get_num_non_dummy_atoms(mol):
    """计算一个片段中“真实”原子的数量（即不包括切割产生的虚拟原子*）。"""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0)

def reconstruct_mol_with_coords(original_mol):
    """
    [严格模式] 通过 SMILES 重建分子拓扑。
    如果 SMILES 生成失败、重建失败或子结构匹配失败，直接返回 None。
    """
    if not original_mol:
        return None

    try:
        # 1. 尝试生成规范化 SMILES
        smiles = Chem.MolToSmiles(original_mol, isomericSmiles=True, canonical=True)
        if not smiles: 
            return None
            
        new_mol = Chem.MolFromSmiles(smiles)
        if not new_mol:
            return None

        # --- 场景 A: 3D 分子 ---
        if original_mol.GetNumConformers() > 0:
            if new_mol.GetNumAtoms() != original_mol.GetNumAtoms():
                return None # 原子数不匹配，视为失败
            
            # 必须匹配成功
            match = original_mol.GetSubstructMatch(new_mol, useChirality=True)
            if not match or len(match) != new_mol.GetNumAtoms():
                match = original_mol.GetSubstructMatch(new_mol, useChirality=False)
                if not match or len(match) != new_mol.GetNumAtoms():
                    return None # 匹配失败，直接舍弃

            # 移植坐标与属性
            conf = Chem.Conformer(new_mol.GetNumAtoms())
            original_conf = original_mol.GetConformer(0)
            
            for new_idx, old_idx in enumerate(match):
                pos = original_conf.GetAtomPosition(old_idx)
                conf.SetAtomPosition(new_idx, pos)
                
                old_atom = original_mol.GetAtomWithIdx(old_idx)
                new_atom = new_mol.GetAtomWithIdx(new_idx)
                if old_atom.HasProp("_original_index"):
                    new_atom.SetIntProp("_original_index", old_atom.GetIntProp("_original_index"))
                if old_atom.HasProp("_GasteigerCharge"):
                    new_atom.SetProp("_GasteigerCharge", old_atom.GetProp("_GasteigerCharge"))

            new_mol.AddConformer(conf)
            return new_mol

        # --- 场景 B: 2D 分子 ---
        else:
             # 对于无3D坐标的情况，只要原子数一致且能匹配，就返回新拓扑
             if new_mol.GetNumAtoms() == original_mol.GetNumAtoms():
                 match = original_mol.GetSubstructMatch(new_mol, useChirality=True)
                 if match:
                     for new_idx, old_idx in enumerate(match):
                         old_atom = original_mol.GetAtomWithIdx(old_idx)
                         new_atom = new_mol.GetAtomWithIdx(new_idx)
                         if old_atom.HasProp("_original_index"):
                             new_atom.SetIntProp("_original_index", old_atom.GetIntProp("_original_index"))
                     return new_mol
             return None

    except Exception as e:
        # 发生任何异常，视为失败，返回 None
        return None

  
def would_create_isolated_atom(mol, bonds_to_cut_indices):
    """
    预测性检查：判断同时切割指定的键列表是否会导致某个 *碳* 原子被完全孤立。
    一个孤立的碳原子是指，它在切割后形成的片段里，是唯一的真实原子，且与其他两个或以上的虚拟原子相连。
    """
    temp_mol = Chem.FragmentOnBonds(mol, bonds_to_cut_indices, addDummies=True)
    for frag in Chem.GetMolFrags(temp_mol, asMols=True):
        # 至少需要3个原子才能构成 "1个真实原子 + 2个虚拟原子" 的情况
        if frag.GetNumAtoms() >= 3:
            non_dummy_atoms = [atom for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0]

            # 检查片段是否只含一个真实原子，且虚拟原子数>=2
            if len(non_dummy_atoms) == 1 and (frag.GetNumAtoms() - 1) >= 2:
                single_real_atom = non_dummy_atoms[0]

                if single_real_atom.GetAtomicNum() == 6:
                    return True
    return False

def identify_carbon_a_atoms(fragment):
    """
    (V2.0 修订版) 识别并返回片段中所有 "Carbon 'a'" 原子的索引集合。
    Carbon 'a' 新定义: 连接了任何杂原子 或 参与了不饱和键的碳。
    """
    carbon_a_indices = set()
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() != 6: continue

        # 条件1: 邻居中是否存在杂原子 (非碳、非氢、非虚拟原子)
        hetero_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() not in [6, 1, 0])
        # 条件2: 自身是否参与了任何非单键
        has_non_single_bond = any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())

        if hetero_neighbors > 0 or has_non_single_bond:
            carbon_a_indices.add(atom.GetIdx())

    return carbon_a_indices

def get_ring_bond_info(mol, eccentricities):
    """
    分析分子中的环系，并为每个键标记其在环系统中的拓扑角色。
    这是切割复杂环状结构（如稠环、桥环）的核心预处理步骤。
    """
    bond_info = {}
    ri = mol.GetRingInfo()

    # 建立一个从原子索引到其所属环索引的映射
    atom_to_rings_map = {i: [] for i in range(mol.GetNumAtoms())}
    for ring_idx, atom_indices in enumerate(ri.AtomRings()):
        for atom_idx in atom_indices:
            atom_to_rings_map[atom_idx].append(ring_idx)

    # 临时存储键的属性，并找出所有桥头原子
    temp_bond_props = {}
    b_bond_atom_indices = set() # 存储所有'b'类键的端点原子（桥头原子）

    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        # 'a'类键: 任何环中的键
        is_a = bond.IsInRing()
        if not is_a:
            temp_bond_props[idx] = {'is_a': False, 'is_b': False, 'is_c': False}
            continue

        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # 'b'类键: 桥头键，同时属于两个或以上环系的键
        is_b = len(set(atom_to_rings_map[begin_idx]).intersection(set(atom_to_rings_map[end_idx]))) >= 2
        # 'c'类键: 芳香键
        is_c = bond.GetIsAromatic()

        temp_bond_props[idx] = {'is_a': is_a, 'is_b': is_b, 'is_c': is_c}
        if is_b:
            b_bond_atom_indices.add(begin_idx)
            b_bond_atom_indices.add(end_idx)

    # 再次遍历，根据桥头原子信息确定'd'类键
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        props = temp_bond_props.get(idx, {'is_a': False})
        is_d = False
        # 'd'类键: 连接桥头原子的非桥头、非芳香的环键。是切割复杂环系的理想目标。
        if props.get('is_a') and not props.get('is_b') and not props.get('is_c'):
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin_idx in b_bond_atom_indices or end_idx in b_bond_atom_indices:
                is_d = True

        bond_info[idx] = {
            'is_a': props.get('is_a'), 'is_b': props.get('is_b'), 'is_c': props.get('is_c'), 'is_d': is_d,
            'ecc': (eccentricities[bond.GetBeginAtomIdx()] + eccentricities[bond.GetEndAtomIdx()]) / 2.0 if props.get('is_a') else -1.0
        }
    return bond_info

def find_type_a_rings(frag, main_mol_for_matching, match_indices, bond_info):
    """规则7的辅助函数：寻找含有两个或以上'd'类键的环。"""
    type_a_rings = []
    # 遍历片段中的每一个环
    for ring_atoms in frag.GetRingInfo().AtomRings():
        # 将片段环中的原子映射回原始分子，找到对应的化学键
        orig_bonds = [b.GetIdx() for i in range(len(ring_atoms)) if (b := main_mol_for_matching.GetBondBetweenAtoms(match_indices[ring_atoms[i]], match_indices[ring_atoms[(i + 1) % len(ring_atoms)]]))]
        
        # 检查这些原始化学键中有多少个是 'd' 类键
        d_bonds = [b_idx for b_idx in orig_bonds if bond_info.get(b_idx, {}).get('is_d')]
        
        # 如果一个环中'd'类键的数量大于等于2，则将其作为候选环
        if len(d_bonds) >= 2:
            type_a_rings.append({'atom_indices': ring_atoms, 'orig_d_bond_indices': d_bonds})
            
    return type_a_rings


def get_atom_coordinates(mol):
    """
    [已修改] 从一个RDKit分子中提取所有原子的3D坐标到一个numpy数组中。
    用于替代原Notebook中的get_atom_coords，并支持scipy的距离计算。
    """
    if not mol or mol.GetNumConformers() == 0:
        return np.array([])
    
    conf = mol.GetConformer(0)
    # 直接返回 (N, 3) 的 NumPy 数组
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

def calculate_centroid(mol_with_conf):
    """
    (增强版) 计算一个分子/碎片中所有重原子的几何中心（质心）坐标。
    
    改进点：
    显式忽略氢原子 (AtomicNum <= 1)。
    防止因去氢不彻底或虚原子残留导致的坐标偏差 (即忽略坐标为 0,0,0 的虚假原子)。
    """
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0:
        return None
        
    conf = mol_with_conf.GetConformer(0)
    
    # --- [关键修改] 筛选重原子 ---
    heavy_atom_indices = [a.GetIdx() for a in mol_with_conf.GetAtoms() if a.GetAtomicNum() > 1]
    num_heavy_atoms = len(heavy_atom_indices)
    
    if num_heavy_atoms == 0:
        return None    
        
    centroid = np.array([0.0, 0.0, 0.0])
    
    # 只累加重原子的坐标
    for i in heavy_atom_indices:
        pos = conf.GetAtomPosition(i)
        centroid += np.array([pos.x, pos.y, pos.z])
        
    # 除以重原子数量
    return tuple(centroid / num_heavy_atoms)


def calculate_reference_coordinate_system(mol_with_conf):
    """
    为分子/片段计算一个确定的、基于原子位置的3x3参考坐标矩阵。

    该坐标系由三个关键原子定义：
    1.  原子A: 分子中规范化排序 (Canonical Rank) 为0的原子。
        - 备用方案: 如果规范化失败，则选择原子质量最大、索引最小的原子。
    2.  原子B: 距离原子A最远的原子 (或在二原子情况下的另一个原子)。
    3.  原子C: 距离A-B连线垂直距离最远的原子 (或在二原子情况下的A-B中点)。

    在选择原子B和C时，如果出现距离并列的情况，则选择规范化排序最高的原子作为决胜者。
    函数会处理原子数少于3的特殊情况。

    Returns:
        np.ndarray: 一个[3, 3]的numpy数组，包含A, B, C的坐标；如果分子无效或无3D构象，则返回None。
    """
    # --- 初始有效性检查 ---
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0:
        return None
    
    # 获取构象
    conf = mol_with_conf.GetConformer(0)
    num_total_atoms = mol_with_conf.GetNumAtoms()
    
    # --- [关键步骤] 筛选重原子索引 ---
    # 只考虑原子序数 > 1 的原子
    heavy_atom_indices = [a.GetIdx() for a in mol_with_conf.GetAtoms() if a.GetAtomicNum() > 1]
    num_heavy_atoms = len(heavy_atom_indices)

    if num_heavy_atoms == 0:
        return None

    # 获取所有坐标 (用于后续索引访问)
    all_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_total_atoms)])
    
    # --- 尝试获取规范化原子排序 ---
    canonical_success = False
    try:
        # 计算所有原子的 rank
        full_ranks = list(Chem.rdmolfiles.CanonicalRankAtoms(mol_with_conf, breakTies=True))
        canonical_success = True
    except Exception:
        # 备用方案：使用索引本身
        full_ranks = list(range(num_total_atoms))
        
    # 建立重原子的 Rank 映射
    # 我们只关心重原子的 Rank 相对大小
    heavy_ranks_map = {idx: full_ranks[idx] for idx in heavy_atom_indices}
    
    # --- 确定原子A (Rank 最小 / 质量最大的重原子) ---
    idx_a = -1
    
    if canonical_success:
        # 方案A: 在重原子中找 Rank 最小的
        # 注意：不一定是全局 Rank 0 (因为 Rank 0 可能是 H)
        idx_a = min(heavy_atom_indices, key=lambda x: heavy_ranks_map[x])
    else:
        # 方案B: 质量最大、索引最小
        max_mass = -1.0
        for idx in heavy_atom_indices:
            atom = mol_with_conf.GetAtomWithIdx(idx)
            current_mass = atom.GetMass()
            if current_mass > max_mass:
                max_mass = current_mass
                idx_a = idx
            elif abs(current_mass - max_mass) < 1e-8:
                if idx < idx_a:
                    idx_a = idx
    
    coord_a = all_coords[idx_a]
    
    # --- 情况1: 只有一个重原子 ---
    if num_heavy_atoms == 1:
        # 赋予一个默认的局部正交标架 (偏移 1.0 埃)
        # 保证 B - A 和 C - A 是正交的非零向量
        coord_b = coord_a + np.array([1.0, 0.0, 0.0])
        coord_c = coord_a + np.array([0.0, 1.0, 0.0])
        return np.array([coord_a, coord_b, coord_c])

    # --- 情况2: 只有两个重原子 ---
    if num_heavy_atoms == 2:
        idx_b = [i for i in heavy_atom_indices if i != idx_a][0]
        coord_b = all_coords[idx_b]
        
        vec_ab = coord_b - coord_a
        vec_ab_len = np.linalg.norm(vec_ab)
        
        if vec_ab_len < 1e-6: # 极低概率的两个原子完全重合
            coord_b = coord_a + np.array([1.0, 0.0, 0.0])
            coord_c = coord_a + np.array([0.0, 1.0, 0.0])
            return np.array([coord_a, coord_b, coord_c])
            
        # 寻找一个与 A-B 向量不共线的随机/默认正交向量，用于生成点 C
        # 找出一个绝对值最小的分量，将其对应的基向量作为扰动，保证叉乘不为 0
        min_axis = np.argmin(np.abs(vec_ab))
        perturb = np.array([0.0, 0.0, 0.0])
        perturb[min_axis] = 1.0
        
        # 叉乘两次得到一个完美的、与 A-B 垂直的法向量，赋予点 C
        ortho_vec = np.cross(vec_ab, np.cross(vec_ab, perturb))
        ortho_vec = ortho_vec / np.linalg.norm(ortho_vec)
        
        coord_c = coord_a + ortho_vec
        return np.array([coord_a, coord_b, coord_c])

    # --- 情况3: 大于等于三个重原子 (主要逻辑) ---
    
    # 2. 确定原子B (距离A最远的重原子)
    max_dist_sq = -1.0
    idx_b = -1
    
    for i in heavy_atom_indices:
        if i == idx_a: continue
        
        dist_sq = np.sum((all_coords[i] - coord_a)**2)
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            idx_b = i
        elif abs(dist_sq - max_dist_sq) < 1e-8:
            # 决胜: Rank 较高者
            if heavy_ranks_map[i] > heavy_ranks_map[idx_b]:
                idx_b = i
                
    coord_b = all_coords[idx_b]

    # 3. 确定原子C (垂直距离A-B连线最远的重原子)
    max_perp_dist_sq = -1.0
    idx_c = -1
    vec_ab = coord_b - coord_a
    vec_ab_len_sq = np.sum(vec_ab**2)
    
    if vec_ab_len_sq < 1e-8:
        # A B 重合 (极罕见)
        for i in heavy_atom_indices:
            if i != idx_a:
                idx_c = i
                break
    else:
        for i in heavy_atom_indices:
            if i in [idx_a, idx_b]: continue
            
            vec_ap = all_coords[i] - coord_a
            projection_len = np.dot(vec_ap, vec_ab) / vec_ab_len_sq
            closest_point = coord_a + projection_len * vec_ab
            perp_dist_sq = np.sum((all_coords[i] - closest_point)**2)

            if perp_dist_sq > max_perp_dist_sq:
                max_perp_dist_sq = perp_dist_sq
                idx_c = i
            elif abs(perp_dist_sq - max_perp_dist_sq) < 1e-8:
                if heavy_ranks_map[i] > heavy_ranks_map[idx_c]:
                    idx_c = i

    # 兜底: 依然没找到C (共线)
    if idx_c == -1:
        # 取 Rank 最高的非 A, B 重原子
        sorted_indices = sorted(heavy_atom_indices, key=lambda x: heavy_ranks_map[x], reverse=True)
        for i in sorted_indices:
            if i not in [idx_a, idx_b]:
                idx_c = i
                break
                
    coord_c = all_coords[idx_c]

    return np.array([coord_a, coord_b, coord_c])

def calculate_max_distance(mol_with_conf):
    """计算分子中任意两个原子间的最大3D距离。"""
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0: return 0.0
    return AllChem.Get3DDistanceMatrix(mol_with_conf).max()

def calculate_max_angle(mol_with_conf):
    """计算分子中任意三个原子形成的最大键角。"""
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0: return 0.0
    conf, num_atoms = mol_with_conf.GetConformer(), mol_with_conf.GetNumAtoms()
    if num_atoms < 3: return 0.0
    positions, max_angle = [conf.GetAtomPosition(i) for i in range(num_atoms)], 0.0
    for p_a, p_b, p_c in combinations(positions, 3):
        v_ba, v_bc = p_a - p_b, p_c - p_b
        len_ba, len_bc = v_ba.Length(), v_bc.Length()
        if len_ba < 1e-6 or len_bc < 1e-6: continue
        cos_angle = v_ba.DotProduct(v_bc) / (len_ba * len_bc)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
        if angle > max_angle: max_angle = angle
    return max_angle

def calculate_max_plane_angle(mol_with_conf):
    """计算分子中任意四个原子形成的最大平面/二面角。"""
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0: return 0.0
    conf, num_atoms = mol_with_conf.GetConformer(), mol_with_conf.GetNumAtoms()
    if num_atoms < 4: return 0.0
    positions, max_angle_val = [conf.GetAtomPosition(i) for i in range(num_atoms)], 0.0
    for p1, p2, p3, p4 in combinations(positions, 4):
        normals_to_check = [((p2 - p1).CrossProduct(p3 - p1), (p2 - p1).CrossProduct(p4 - p1)), ((p2 - p1).CrossProduct(p3 - p1), (p3 - p2).CrossProduct(p4 - p2))]
        for n1, n2 in normals_to_check:
            if n1.Length() > 1e-6 and n2.Length() > 1e-6:
                angle_rad = n1.AngleTo(n2)
                angle_deg = min(math.degrees(angle_rad), 180 - math.degrees(angle_rad))
                if angle_deg > max_angle_val: max_angle_val = angle_deg
    return max_angle_val

def fragment_on_bonds_stable(mol, bond_indices):
    """
    通过手动移除化学键并添加带唯一、持久化标签的虚原子来切割分子。
    标签是一个字符串，格式为 "cut-{bond_idx}"，确保了全局唯一性和确定性。
    该方法解决了 RDKit 内置函数在处理多个切割时标签顺序不确定的问题。
    """
    if not bond_indices:
        return mol

    rw_mol = Chem.RWMol(mol)
    
    # 为了防止索引变化，我们从大到小处理要删除的键
    bonds_to_process = sorted(list(set(bond_indices)), reverse=True)

    for bond_idx in bonds_to_process:
        bond = rw_mol.GetBondWithIdx(bond_idx)
        if not bond: continue

        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        
        # --- 创建一个基于键索引的、全局唯一的字符串标签 ---
        unique_label = f"cut-{bond_idx}"
        
        # 1. 移除原始的化学键
        rw_mol.RemoveBond(begin_atom_idx, end_atom_idx)
        
        # 2. 添加第一个虚原子，连接到起始原子，并打上标签
        dummy1_idx = rw_mol.AddAtom(Chem.Atom(0))
        dummy1 = rw_mol.GetAtomWithIdx(dummy1_idx)
        dummy1.SetProp("_cut_id", unique_label) # 使用 SetProp 存储自定义属性
        rw_mol.AddBond(begin_atom_idx, dummy1_idx, Chem.BondType.SINGLE)
        
        # 3. 添加第二个虚原子，连接到结束原子，并打上相同的标签
        dummy2_idx = rw_mol.AddAtom(Chem.Atom(0))
        dummy2 = rw_mol.GetAtomWithIdx(dummy2_idx)
        dummy2.SetProp("_cut_id", unique_label)
        rw_mol.AddBond(end_atom_idx, dummy2_idx, Chem.BondType.SINGLE)
        
    final_mol = rw_mol.GetMol()
    # 使用SANITIZE_NONE来避免RDKit对我们手动创建的结构进行化学合理性检查
    try:
        Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
    except Exception:
        pass # 容忍可能出现的净化错误
        
    return final_mol

# ----------------------------------------
#  分子预处理与属性计算
# ----------------------------------------
def prepare_molecule_for_cutting(mol):
    # 1. 函数一开头：打上 _original_index (给主脚本建边用)
    if not mol.GetAtomWithIdx(0).HasProp("_original_index"):
        for atom in mol.GetAtoms():
            atom.SetIntProp("_original_index", atom.GetIdx())

    # 1. 先进行分离 (分离出的碎片会继承刚才打上的 _original_index)
    main_mol, ion_fragments = rule1_separate_ions(mol)
    
    result_data = {
        "main_mol_for_matching": None, 
        "ion_fragments": ion_fragments,
        "original_eccentricities": [],
        "original_logp_contribs": [],
        "original_tpsa_contribs": [],
        "original_charges": [],
        "bond_info": {}
    }

    if main_mol.GetNumAtoms() == 0:
        return result_data

    try:
        main_mol_for_matching = Chem.RemoveHs(main_mol)
        reconstructed_mol = reconstruct_mol_with_coords(main_mol_for_matching)
        if reconstructed_mol is not None:
            main_mol_for_matching = reconstructed_mol

        # 2. 拓扑重建后：打上 _internal_index (给当前特征计算查表用)
        for atom in main_mol_for_matching.GetAtoms():
            atom.SetIntProp("_internal_index", atom.GetIdx())
            if not atom.HasProp("_original_index"):
                atom.SetIntProp("_original_index", atom.GetIdx())

        result_data["main_mol_for_matching"] = main_mol_for_matching

        # ★ 关键修复点 2：将各项属性计算拆分，单独进行容错。算不出就跳过，绝不牵连分子本身。
        try:
            distance_matrix = GetDistanceMatrix(main_mol_for_matching)
            result_data["original_eccentricities"] = distance_matrix.max(axis=1)
        except: pass

        try:
            logp_contribs = rdMolDescriptors._CalcCrippenContribs(main_mol_for_matching)
            result_data["original_logp_contribs"] = [x[0] for x in logp_contribs]
        except: pass

        try:
            tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(main_mol_for_matching)
            result_data["original_tpsa_contribs"] = list(tpsa_contribs)
        except: pass

        try:
            AllChem.ComputeGasteigerCharges(main_mol_for_matching)
            charges_raw = [atom.GetProp('_GasteigerCharge') for atom in main_mol_for_matching.GetAtoms()]
            result_data["original_charges"] = [float(c) for c in charges_raw]
        except: pass

        try:
            if len(result_data["original_eccentricities"]) > 0:
                result_data["bond_info"] = get_ring_bond_info(main_mol_for_matching, result_data["original_eccentricities"])
        except: pass

        return result_data

    except Exception as e:
        # 如果最开始的 RemoveHs 等基础操作就崩溃了，只能放弃
        return result_data
    
def calculate_minimum_enclosing_ball(points: np.ndarray):
    """
    使用 Badoiu 和 Clarkson 的迭代算法计算点云的最小包围球。

    Args:
        points (np.ndarray): 一个形状为 (N, 3) 的numpy数组，代表N个点的3D坐标。

    Returns:
        tuple[np.ndarray, float]: 返回一个元组，包含：
            - center (np.ndarray): 球心的3D坐标。
            - radius (float): 球的半径。
    """
    # --- 边缘情况处理 ---
    num_points = points.shape[0]
    if num_points == 0:
        return np.zeros(3), 0.0
    if num_points == 1:
        return points[0], 0.0

    # --- 使用迭代算法 ---
    # 1. 初始化: 使用平均中心作为一个好的起点
    center = np.mean(points, axis=0)
    
    # 2. 迭代更新中心点
    for i in range(1, 151): # 150 次迭代对于典型的口袋来说绰绰有余
        # 找到当前距离中心最远的点
        dists_sq = np.sum((points - center)**2, axis=1)
        furthest_idx = np.argmax(dists_sq)
        p_i = points[furthest_idx]
        
        # 将中心点向最远点移动一小步
        step = 1.0 / (i + 1)
        center = (1.0 - step) * center + step * p_i

    # 3. 确定最终半径
    #    在找到最终的中心后，半径就是该中心到所有点的最大距离
    final_dists = np.linalg.norm(points - center, axis=1)
    radius = np.max(final_dists)
    
    return center, radius


# --- 数据汇总与描述符计算函数 (2D/3D 兼容版) ---

METAL_SYMBOLS = {
    'FE', 'CU', 'MN', 'CO', 'NI', 'MO', 'V', 'CR', 'ZN', 'MG', 'CA', 
    'CD', 'HG', 'PB', 'NA', 'K', 'LI', 'RB', 'CS', 'AL', 'GA', 
    'EU', 'GD', 'LA', 'YB', 'SM', 'AG', 'BI', 'IN', 'TL', 'CE', 'SR', 'BA', 'SN'
}

def get_standardized_metal_smiles(mol_obj):
    """
    (V1.1 已修正大小写)
    检查一个分子片段是否为单原子金属。
    如果是，则返回其不带电荷和方括号的、保持正确大小写的标准SMILES（例如 "[Mg]"）。
    否则返回 None。
    """
    if mol_obj.GetNumAtoms() == 1 and mol_obj.GetAtomWithIdx(0).GetAtomicNum() != 0:
        atom = mol_obj.GetAtomWithIdx(0)
        
        # --- 修正点: 先获取正确的大小写，再进行不区分大小写的比较 ---
        symbol_correct_case = atom.GetSymbol() # 例如 "Mg", "Fe"
        
        # 使用 .upper() 只是为了在 METAL_SYMBOLS 集合中进行查找
        if symbol_correct_case.upper() in METAL_SYMBOLS:
            # 返回时，使用原始的、大小写正确的符号
            return f"[{symbol_correct_case}]"
            
    return None

def _bfs_find_nearest_single_bond(mol, start_atom_idx, blocked_parent_idx):
    """
    辅助函数：BFS 搜索最近的单键。
    已包含对虚原子的检查，防止搜索穿过虚原子。
    """
    queue = [(start_atom_idx, 0)]
    visited = {start_atom_idx, blocked_parent_idx}
    MAX_SEARCH_DEPTH = 5 

    while queue:
        curr_idx, depth = queue.pop(0)
        if depth >= MAX_SEARCH_DEPTH: continue

        curr_atom = mol.GetAtomWithIdx(curr_idx)
        
        # 遇到虚原子直接停止该路径
        if curr_atom.GetAtomicNum() == 0:
            continue

        if curr_atom.IsInRing() and curr_idx != start_atom_idx:
            continue

        for bond in curr_atom.GetBonds():
            neighbor = bond.GetOtherAtom(curr_atom)
            nbr_idx = neighbor.GetIdx()
            
            # 忽略虚原子邻居
            if neighbor.GetAtomicNum() == 0:
                continue

            if nbr_idx in visited: continue

            if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                return bond.GetIdx()

            if not neighbor.IsInRing():
                visited.add(nbr_idx)
                queue.append((nbr_idx, depth + 1))
    return None

def is_fragment_valid(mol):
    """
    检查片段是否符合保留条件：
    1. 重原子数 <= 24
    2. 环数量 < 6
    3. 单个环的键数量 < 12 (无特大环)
    """
    # 1. 重原子检查
    if mol.GetNumHeavyAtoms() > 24:
        return False
    
    ri = mol.GetRingInfo()
    # 2. 环数量检查
    if ri.NumRings() >= 6:
        return False
        
    # 3. 大环检查 (AtomRings 返回每个环的原子索引元组，长度即为环大小)
    for ring in ri.AtomRings():
        if len(ring) >= 12:
            return False
            
    return True

def advanced_rescue_valence_errors(mol):
    """
    [增强版] 结晶学伪影精确抢救机制：
    直接捕获 RDKit 抛出的 Explicit valence 异常，通过正则解析出具体出问题的原子 ID。
    然后针对性地切断该原子连接的最长的一根键。循环直到分子通过检查。
    """
    if not mol or mol.GetNumConformers() == 0:
        return mol

    import re
    rw_mol = Chem.RWMol(mol)
    conf = rw_mol.GetConformer(0)
    
    max_attempts = 50  # 设置最大抢救次数，防止死循环
    
    for attempt in range(max_attempts):
        try:
            # 每次测试都需要在一个全新的拷贝上进行，防止 RDKit 内部状态在失败时被彻底破坏
            test_mol = rw_mol.GetMol()
            test_mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(test_mol, sanitizeOps=Chem.SANITIZE_SYMMRINGS|Chem.SANITIZE_SETCONJUGATION|Chem.SANITIZE_SETHYBRIDIZATION)
            # 如果能顺利走到这里，说明分子已经完全健康了！
            return test_mol
            
        except Exception as e:
            error_msg = str(e)
            # 尝试从错误信息中抓取出问题的原子 ID
            # 典型的报错如: "Explicit valence for atom # 1061 C, 5, is greater than permitted"
            match = re.search(r"atom #\s*(\d+)", error_msg)
            
            if match:
                bad_atom_idx = int(match.group(1))
                bad_atom = rw_mol.GetAtomWithIdx(bad_atom_idx)
                
                max_len = -1.0
                bond_to_break = None
                
                # 遍历这个错误原子的所有键，找到 3D 距离最长的那一根 (最可疑的虚假键)
                for bond in bad_atom.GetBonds():
                    neighbor = bond.GetOtherAtom(bad_atom)
                    pos1 = np.array(conf.GetAtomPosition(bad_atom.GetIdx()))
                    pos2 = np.array(conf.GetAtomPosition(neighbor.GetIdx()))
                    dist = np.linalg.norm(pos1 - pos2)
                    
                    if dist > max_len:
                        max_len = dist
                        bond_to_break = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                
                # 切断这根最长的键
                if bond_to_break:
                    rw_mol.RemoveBond(bond_to_break[0], bond_to_break[1])
                else:
                    # 如果找不到键可以切（极端情况），只能放弃抢救
                    break
            else:
                # 如果抛出的不是化合价溢出异常，或者正则没匹配上，停止抢救
                break
                
    # 返回尽力抢救后的结果
    return rw_mol.GetMol()

def calculate_descriptors_and_store(storage, mol_data, original_mol_with_conf, main_mol_for_matching, final_covalent_fragments, ion_fragments, original_eccentricities, original_logp_contribs, original_tpsa_contribs, original_charges):
    """
    (V3.0 修复版)
    - 逻辑修正: 二次精修队列判定改为检测虚原子，彻底解决 [5*] 等开环残留问题。
    - 兼容性: 支持仅处理离子片段（当主分子重构失败时）。
    """
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = Feat.BuildFeatureFactory(fdef_name)

    # === 新增：特征硬截断边界与辅助函数 ===
    FEATURE_BOUNDS = {
        'avg_ecc': (0.0, 100.0),
        'ecc_range': (0.0, 50.0),
        'avg_logp': (-15.0, 15.0),
        'avg_tpsa': (0.0, 500.0),
        'avg_charge': (-5.0, 5.0),
        'HBA': (0.0, 20.0),
        'HBD': (0.0, 20.0),
        'Aromatic': (0.0, 10.0),
        'Hydrophobe': (0.0, 20.0),
        'PosIonizable': (0.0, 10.0),
        'NegIonizable': (0.0, 10.0),
        'max_dist_3d': (0.0, 100.0),
        'max_angle_3d': (0.0, 180.0),
        'max_plane_angle_3d': (0.0, 180.0)
    }

    def enforce_chemical_bounds(data_dict):
        """对字典中的化学特征进行强制截断，防止极端异常值毁灭全局方差"""
        for prop, bounds in FEATURE_BOUNDS.items():
            if prop in data_dict and data_dict[prop] is not None:
                val = data_dict[prop]
                if isinstance(val, (int, float)) and np.isfinite(val):
                    data_dict[prop] = float(np.clip(val, bounds[0], bounds[1]))
        return data_dict

    single_atom_ion_charges = {
        'FE': 2.5, 'CU': 1.5, 'MN': 3, 'CO': 2.5, 'NI': 2.5,'MO': 5, 'V': 4, 'CR': 3, 'ZN': 2, 'MG': 2, 
        'CA': 2, 'CD': 2, 'HG': 2, 'PB': 2, 'NA': 1, 'K': 1, 'LI': 1, 'RB': 1, 'CS': 1, 'AL': 3, 'GA': 3, 
        'EU': 3, 'GD': 3, 'LA': 3, 'YB': 3, 'SM': 3, 'CL': -1, 'BR': -1, 'I': -1, 'F': -1,
        '[O-]P(=O)([O-])[O-]': -3.0, 'O=P([O-])([O-])O': -2.0, 'O=P([O-])(O)O': -1.0, 
        '[O-]S(=O)(=O)[O-]': -2.0, '[O-][N+](=O)[O-]': -1.0, '[O-]C(=O)[O-]': -2.0, 
        'O=C([O-])O': -1.0, 'CC(=O)[O-]': -1.0, '[O-]C=O': -1.0, 'O=C([O-])C(=O)[O-]': -2.0, 
        'O=C([O-])CC(=O)[O-]': -2.0, 'O=C([O-])CCC(=O)[O-]': -2.0, 
        'O=C([O-])CC(O)(C(=O)[O-])C(=O)[O-]': -3.0, 'O=C([O-])[C@H](O)[C@@H](O)C(=O)[O-]': -2.0, 
        'O=P([O-])([O-])OCC(O)CO': -2.0, '[O-]P(=O)([O-])OP(=O)([O-])[O-]': -4.0, 
        'O=C([O-])C(F)(F)F': -1.0, '[S-]C#N': -1.0, '[O-]Cl(=O)(=O)=O': -1.0
    }
    
    has_3d_conformer = original_mol_with_conf.GetNumConformers() > 0
    original_conformer = original_mol_with_conf.GetConformer(0) if has_3d_conformer else None
    
    # 丢弃计数器初始化
    dropped_fragment_count = 0

    def get_pharmacophore_features(mol_obj):
        feats = factory.GetFeaturesForMol(mol_obj)
        feature_counts = {'HBA': 0, 'HBD': 0, 'Aromatic': 0, 'Hydrophobe': 0, 'PosIonizable': 0, 'NegIonizable': 0}
        for f in feats:
            family = f.GetFamily()
            if family == 'Acceptor': feature_counts['HBA'] += 1
            elif family == 'Donor': feature_counts['HBD'] += 1
            elif family == 'Aromatic': feature_counts['Aromatic'] += 1
            elif family == 'Hydrophobe': feature_counts['Hydrophobe'] += 1
            elif family == 'PosIonizable': feature_counts['PosIonizable'] += 1
            elif family == 'NegIonizable': feature_counts['NegIonizable'] += 1
        return feature_counts

    # --- 处理共价片段 (重构版：队列+二次切割) ---
    current_mol_results_data = []
    
    # 1. 初始化队列
    processing_queue = list(final_covalent_fragments)
    max_iterations = 1000 
    iterations = 0

    while processing_queue and iterations < max_iterations:
        fragment = processing_queue.pop(0)
        iterations += 1
        
        # === A. 深度清洗 (Dummy -> H -> Sanitize -> RemoveHs) ===
        rw_mol = Chem.RWMol(fragment)
        dummies = [a for a in rw_mol.GetAtoms() if a.GetAtomicNum() == 0]
        
        for atom in dummies:
            # 重置键类型
            for bond in atom.GetBonds():
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    bond.SetBondType(Chem.BondType.SINGLE)
                    bond.SetIsAromatic(False)
            
            # [关键] 将虚原子变为 H
            atom.SetAtomicNum(1)        
            atom.SetIsotope(0)          
            atom.SetAtomMapNum(0)
            if atom.HasProp("_original_index"): atom.ClearProp("_original_index")
            if atom.HasProp("_cut_id"): atom.ClearProp("_cut_id")
            # [核心修复 3]: 同步清理内部索引，防止虚原子干扰特征提取
            if atom.HasProp("_internal_index"): atom.ClearProp("_internal_index")

            # 激活邻居原子的隐式氢重算
            for neighbor in atom.GetNeighbors():
                neighbor.SetNoImplicit(False) 
                neighbor.SetNumExplicitHs(0) 
                neighbor.UpdatePropertyCache(strict=False)

        clean_temp = rw_mol.GetMol()
        
        # 3. 结构修复 (Sanitize) - 分级尝试与 3D 距离抢救
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        rescued_by_cutting = False  
        
        try:
            Chem.SanitizeMol(clean_temp)
        except Exception:
            try:
                Chem.SanitizeMol(clean_temp, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                # 触发 3D 键长抢救机制
                clean_temp = advanced_rescue_valence_errors(clean_temp) # <-- 修复：同步为新函数名
                rescued_by_cutting = True  # 标记可能发生了断键
                try:
                    Chem.SanitizeMol(clean_temp, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    pass 
        finally:
            lg.setLevel(RDLogger.CRITICAL)

        # 4. 去氢 (RemoveHs) - 保护有机立体氢，定点清除金属/类金属上的畸形假氢
        try:
            # 第一步：先执行 RDKit 默认去氢（安全，保留有机片段的立体氢）
            clean_fragment = Chem.RemoveHs(clean_temp)
            
            # 第二步：针对遗留的金属/类金属氢进行定点暴力清除
            rw_mol = Chem.RWMol(clean_fragment)
            h_to_remove = []
            
            for atom in rw_mol.GetAtoms():
                if atom.GetAtomicNum() == 1: # 找到残留的氢原子
                    neighbor = atom.GetNeighbors()[0]
                    # 常见有机重原子白名单: B(5), C(6), N(7), O(8), F(9), Si(14), P(15), S(16), Cl(17), Se(34), Br(35), I(53)
                    organic_heavy_nums = {5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
                    
                    if neighbor.GetAtomicNum() not in organic_heavy_nums:
                        h_to_remove.append(atom.GetIdx())
            
            # 从大到小删除原子，防止 RWMol 内部索引错乱
            for idx in sorted(h_to_remove, reverse=True):
                rw_mol.RemoveAtom(idx)
                
            clean_fragment = rw_mol.GetMol()
        except:
            clean_fragment = clean_temp

        if rescued_by_cutting:
            split_frags = list(Chem.GetMolFrags(clean_fragment, asMols=True))
            if len(split_frags) > 1:
                # 如果断裂成了多个独立碎片，将它们放回队列头部，重新进行标准化清洗
                processing_queue[0:0] = split_frags
                continue 
        
        # =========================================================================
        # B. 二次精修流水线 (Rule 3 -> Rule 5 连续执行)
        # =========================================================================
        pipeline_frags = [clean_fragment]
        
        # --- 步骤 1: 准备拓扑信息 (为 Rule 3) ---
        try:
            temp_dist_mat = GetDistanceMatrix(clean_fragment)
            temp_eccs = temp_dist_mat.max(axis=1)
            temp_bond_info = get_ring_bond_info(clean_fragment, temp_eccs)
        except Exception:
            temp_bond_info = {} 

        # --- 步骤 2: 执行 Rule 3 (大环切割) ---
        try:
            r3_results = rule3_cut_large_rings(pipeline_frags, temp_bond_info)
        except Exception:
            r3_results = pipeline_frags
        
        pipeline_frags = r3_results
        
        # --- 步骤 3: 执行 Rule 5 (无环链切割) ---
        try:
            r5_results = rule5_cut_acyclic_chains(pipeline_frags)
        except Exception:
            r5_results = pipeline_frags
            
        # =========================================================================
        # [核心修订] C. 判定结果与递归
        # =========================================================================
        # 只要结果中出现了虚原子 (AtomicNum == 0)，说明发生了切割或开环
        has_new_dummy = False
        for res_mol in r5_results:
            for atom in res_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    has_new_dummy = True
                    break
            if has_new_dummy: break
            
        if has_new_dummy:
            # 如果产生了新的切割痕迹，放回队列重新清洗 (将虚原子转H)
            processing_queue[0:0] = r5_results 
            continue 
            
        final_processed_mol = r5_results[0]

        # =========================================================================
        # D. 最终阶段过滤检查
        # =========================================================================
        if not is_fragment_valid(final_processed_mol):
            dropped_fragment_count += 1
            continue 

        # === E. 生成最终 SMILES 和属性  ===
        standardized_metal_smi = get_standardized_metal_smiles(final_processed_mol)
        if standardized_metal_smi:
            final_smiles = standardized_metal_smi
        else:
            try:
                final_smiles = Chem.MolToSmiles(final_processed_mol, canonical=True)
            except:
                try:
                    final_smiles = Chem.MolToSmiles(clean_temp, canonical=True)
                except:
                    final_smiles = "<ERROR>"

        if final_smiles == "<ERROR>":
            dropped_fragment_count += 1
            continue

        match_indices = []
        for atom in final_processed_mol.GetAtoms():
            if atom.HasProp("_internal_index"):
                match_indices.append(atom.GetIntProp("_internal_index"))
        
        feature_counts = get_pharmacophore_features(final_processed_mol)
        r = {
            "mol": final_processed_mol, 
            "clean_mol": final_processed_mol, 
            "smiles": final_smiles,
            "avg_ecc": -1.0, "ecc_range": 0.0, "avg_logp": -1.0, "avg_tpsa": -1.0, "avg_charge": -1.0,
            "HBA": float(feature_counts['HBA']), "HBD": float(feature_counts['HBD']),
            "Aromatic": float(feature_counts['Aromatic']), "Hydrophobe": float(feature_counts['Hydrophobe']),
            "PosIonizable": float(feature_counts['PosIonizable']), "NegIonizable": float(feature_counts['NegIonizable']),
        }

        # ==========================================================
        # 属性赋值：Plan A (全局映射) -> Plan B (局部计算) -> Plan C (处决)
        # ==========================================================
        properties_calculated = False

        def _is_healthy(*values):
            for v in values:
                if not np.isfinite(v): 
                    return False
                if abs(v) > 1e6: 
                    return False
            return True
        
        # === Plan A: 优先尝试使用高质量的预计算全局属性映射 ===
        if match_indices and original_eccentricities is not None and len(original_eccentricities) > 0:
            try:
                frag_eccentricities = [original_eccentricities[i] for i in match_indices if i < len(original_eccentricities)]
                if frag_eccentricities:
                    avg_ecc = np.nanmean(frag_eccentricities)
                    ecc_range = np.ptp(frag_eccentricities)
                    avg_logp = np.nanmean([original_logp_contribs[i] for i in match_indices])
                    avg_tpsa_rdkit = np.nanmean([original_tpsa_contribs[i] for i in match_indices])
                    avg_charge = np.nanmean([original_charges[i] for i in match_indices])
                    
                    if abs(avg_charge) >= 1:
                        avg_tpsa_adjusted = avg_tpsa_rdkit + (abs(avg_charge) * 25.0)
                    else:
                        avg_tpsa_adjusted = avg_tpsa_rdkit
                        
                    # 【核心修订】：只有所有数值都健康，才认可 Plan A 成功！
                    if _is_healthy(avg_ecc, ecc_range, avg_logp, avg_tpsa_adjusted, avg_charge):
                        r.update({
                            "avg_ecc": avg_ecc, "ecc_range": ecc_range, "avg_logp": avg_logp,
                            "avg_tpsa": avg_tpsa_adjusted, "avg_charge": avg_charge
                        })
                        properties_calculated = True
                    else:
                        # 如果不健康，静默跳过，让程序自然进入 Plan B
                        pass
            except Exception:
                pass # Plan A 代码崩溃，静默放行，交给 Plan B 兜底

        # === Plan B: 全局预计算如果失败，立刻对切好的碎片进行局部实时计算 ===
        if not properties_calculated:
            try:
                # 1. 算局部 LogP 和 TPSA
                frag_logp = rdMolDescriptors.CalcCrippenDescriptors(final_processed_mol)[0]
                frag_tpsa_rdkit = rdMolDescriptors.CalcTPSA(final_processed_mol)
                
                # 2. 算局部电荷
                AllChem.ComputeGasteigerCharges(final_processed_mol)
                charges_raw = [atom.GetProp('_GasteigerCharge') for atom in final_processed_mol.GetAtoms() if atom.HasProp('_GasteigerCharge')]
                charges_float = [float(c) for c in charges_raw if np.isfinite(float(c))]
                frag_avg_charge = np.nanmean(charges_float) if charges_float else 0.0
                
                # 3. 算局部拓扑偏心率
                dist_mat = GetDistanceMatrix(final_processed_mol)
                eccs = dist_mat.max(axis=1)
                if len(eccs) == 0: raise ValueError("拓扑畸形，无法计算偏心率")
                frag_avg_ecc = np.nanmean(eccs)
                frag_ecc_range = np.ptp(eccs)
                
                # TPSA 补正逻辑
                if abs(frag_avg_charge) >= 1:
                    frag_tpsa_adjusted = frag_tpsa_rdkit + (abs(frag_avg_charge) * 25.0)
                else:
                    frag_tpsa_adjusted = frag_tpsa_rdkit
                
                # 【核心修订】：对 Plan B 的结果也进行健康体检
                if not _is_healthy(frag_avg_ecc, frag_ecc_range, frag_logp, frag_tpsa_adjusted, frag_avg_charge):
                    raise ValueError("局部属性计算结果包含不合法数值 (如 NaN/Inf 或 RDKit孤岛惩罚值)")
                
                # 更新到字典中
                r.update({
                    "avg_ecc": frag_avg_ecc, "ecc_range": frag_ecc_range, "avg_logp": frag_logp,
                    "avg_tpsa": frag_tpsa_adjusted, "avg_charge": frag_avg_charge
                })
                properties_calculated = True
                
            except Exception as e:
                # === Plan C: 畸形到连局部属性也算不出，坚决放弃该复合物！ ===
                # 抛出异常被外层捕获
                raise ValueError(f"碎片极度畸形，无法计算化学属性，触发复合物丢弃: {e}")

        # --- 兜底判断：即使 _internal_index 丢失，只要有 _original_index 就能恢复坐标 ---
        has_original_index = any(a.HasProp("_original_index") for a in final_processed_mol.GetAtoms())

        if has_3d_conformer and (match_indices or has_original_index):
            new_conf = Chem.Conformer(final_processed_mol.GetNumAtoms())
            try:
                for atom in final_processed_mol.GetAtoms():
                    if atom.HasProp("_original_index"):
                        orig_idx = atom.GetIntProp("_original_index")
                        if orig_idx < original_conformer.GetNumAtoms():
                            orig_pos = original_conformer.GetAtomPosition(orig_idx)
                            new_conf.SetAtomPosition(atom.GetIdx(), orig_pos)
                        else:
                            new_conf.SetAtomPosition(atom.GetIdx(), (0.0, 0.0, 0.0))
                    else:
                        new_conf.SetAtomPosition(atom.GetIdx(), (0.0, 0.0, 0.0))

                final_processed_mol.AddConformer(new_conf, assignId=True)
                r['centroid_3d'] = calculate_centroid(final_processed_mol)
                r['ref_coord_3d'] = calculate_reference_coordinate_system(final_processed_mol)
                r['max_dist_3d'] = calculate_max_distance(final_processed_mol)
                r['max_angle_3d'] = calculate_max_angle(final_processed_mol)
                r['max_plane_angle_3d'] = calculate_max_plane_angle(final_processed_mol)
            except Exception:
                 r['centroid_3d'] = None
                 r['ref_coord_3d'] = None
        else:
            r['max_dist_3d'] = -1.0; r['max_angle_3d'] = -1.0; r['max_plane_angle_3d'] = -1.0
            r['centroid_3d'] = None
            r['ref_coord_3d'] = None
            
        r = enforce_chemical_bounds(r)
        current_mol_results_data.append(r)

    # --- 处理离子片段 ---
    current_mol_ion_data = []
    for ion in ion_fragments:
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        try:
             Chem.SanitizeMol(ion)
        except:
             pass
        finally:
            lg.setLevel(RDLogger.CRITICAL)
            
        type_tag = ""
        if ion.HasProp("_Name"):
            name_prop = ion.GetProp("_Name")
            if name_prop.endswith("**"): type_tag = "**"
            elif name_prop.endswith("***"): type_tag = "***"
            elif name_prop.endswith("*"): type_tag = "*"
        is_neutral_agent = (type_tag == "**")
        
        final_charge = float(Chem.GetFormalCharge(ion))
        if not is_neutral_agent and final_charge == 0:
            ion_smiles_lookup = Chem.MolToSmiles(ion)
            try:
                ion_symbol = ion.GetAtomWithIdx(0).GetSymbol().upper() if ion.GetNumAtoms() == 1 else None
            except:
                ion_symbol = None
                
            if ion_smiles_lookup in single_atom_ion_charges:
                final_charge = single_atom_ion_charges[ion_smiles_lookup]
            elif ion_symbol and ion_symbol in single_atom_ion_charges:
                final_charge = single_atom_ion_charges[ion_symbol]

        if int(final_charge) != Chem.GetFormalCharge(ion) and ion.GetNumAtoms() == 1:
            atom = ion.GetAtomWithIdx(0)
            atom.SetFormalCharge(int(final_charge))
            atom.UpdatePropertyCache(strict=False)

        standardized_metal_smi = get_standardized_metal_smiles(ion)
        
        if standardized_metal_smi:
            if type_tag:
                 smiles = standardized_metal_smi + type_tag
            else:
                 smiles = standardized_metal_smi + "***"
        elif is_neutral_agent:
            smiles = ion.GetProp("_Name")
        else:
            try:
                base_smiles = Chem.MolToSmiles(ion, canonical=True)
                if type_tag:
                    smiles = base_smiles + type_tag
                else:
                    smiles = base_smiles + "***"
            except:
                smiles = "[ERROR]*"

        feature_counts = get_pharmacophore_features(ion)
        
        # 离子属性计算
        avg_logp = 0.0
        try:
            avg_logp = rdMolDescriptors.CalcCrippenDescriptors(ion)[0]
        except: pass
        
        ion_data = {
            "smiles": smiles, "avg_logp": avg_logp,
            "HBA": float(feature_counts['HBA']), "HBD": float(feature_counts['HBD']),
            "Aromatic": float(feature_counts['Aromatic']), "Hydrophobe": float(feature_counts['Hydrophobe']),
            "PosIonizable": float(feature_counts['PosIonizable']), "NegIonizable": float(feature_counts['NegIonizable']),
            "avg_charge": final_charge, "avg_ecc": -25.0, "ecc_range": 0.0
        }
        
        tpsa_rdkit = 0.0
        try: tpsa_rdkit = rdMolDescriptors.CalcTPSA(ion)
        except: pass
        
        if abs(final_charge) >= 1:
            tpsa_adjusted = tpsa_rdkit + (abs(final_charge) * 25.0)
        else:
            tpsa_adjusted = tpsa_rdkit
        ion_data['avg_tpsa'] = tpsa_adjusted
        
        if ion.GetNumConformers() > 0:
            try:
                ion_data['centroid_3d'] = calculate_centroid(ion)
                ion_data['ref_coord_3d'] = calculate_reference_coordinate_system(ion)
                ion_data['max_dist_3d'] = calculate_max_distance(ion)
                ion_data['max_angle_3d'] = calculate_max_angle(ion)
                ion_data['max_plane_angle_3d'] = calculate_max_plane_angle(ion)
            except:
                ion_data['max_dist_3d'] = -1.0; ion_data['max_angle_3d'] = -1.0; ion_data['max_plane_angle_3d'] = -1.0
                ion_data['centroid_3d'] = None
                ion_data['ref_coord_3d'] = None
        else:
            ion_data['max_dist_3d'] = -1.0; ion_data['max_angle_3d'] = -1.0; ion_data['max_plane_angle_3d'] = -1.0
            ion_data['centroid_3d'] = None
            ion_data['ref_coord_3d'] = None
        ion_data = enforce_chemical_bounds(ion_data)
        current_mol_ion_data.append(ion_data)

    storage.append({
        'id': mol_data['IDs'], 'smiles': mol_data['Smiles'],
        'results': current_mol_results_data, 'ions': current_mol_ion_data
    })
    
    return dropped_fragment_count

def update_online_stats(existing_stats, new_data_point):
    """
    [新移入] 使用Welford在线算法，根据单个新数据点来精确更新统计量。
    """
    count = existing_stats.get('count', 0)
    mean = existing_stats.get('mean', 0.0)
    M2 = existing_stats.get('M2', 0.0)
    
    count += 1
    delta = new_data_point - mean
    mean += delta / count
    M2 += delta * (new_data_point - mean)
    
    return {'count': count, 'mean': mean, 'M2': M2}

# ----------------------------------------
# 主切割流程
# ----------------------------------------

def execute_fragmentation_pipeline(main_mol_for_matching, bond_info):
    """
    碎片化流水线。
    1. Rule 2: 切断金属 (Metals)
    2. Rule 3: 打开大环 (Large Rings, 含隐式大环)
    3. Rule 6: 拆解螺环中心 (Spiro Centers)
    4. Rule 7: 拆解稠环/多环系统 (Multi-ring Systems)
    5. Rule 4: 切割环上的取代基/连接链 (Ring Substituents) - 此时处理的是已被 R6/R7 简化过的环系统
    6. Rule 9: 切割类酸官能团 (Acid-like Groups) - 优先进行化学特异性切割
    7. Rule 5: 切割无环长链 (Acyclic Chains) - 处理所有剩余的链，含增强的杂原子链检查
    8. Rule 8: 清理大的纯碳脂肪片段 (Large Carbon Fragments)
    """

    fragments_in_pipeline = [main_mol_for_matching]
    fragments_in_pipeline = rule2_cut_metals(fragments_in_pipeline)
    fragments_in_pipeline = rule3_cut_large_rings(fragments_in_pipeline, bond_info)
    fragments_in_pipeline = rule6_cut_spiro_centers(fragments_in_pipeline, main_mol_for_matching, bond_info)
    fragments_in_pipeline = rule7_cut_multi_ring_fragments(fragments_in_pipeline, main_mol_for_matching, bond_info)
    fragments_in_pipeline = rule4_cut_ring_substituents(fragments_in_pipeline)
    fragments_in_pipeline = rule9_cut_acid_like_groups(fragments_in_pipeline)
    fragments_in_pipeline = rule5_cut_acyclic_chains(fragments_in_pipeline)
    final_covalent_fragments = rule8_cut_large_carbon_fragments(fragments_in_pipeline, main_mol_for_matching)

    return final_covalent_fragments

def rule1_separate_ions(mol):
    """
    1. 获取所有不连通片段。
    2. 检查每个片段的 SMILES 是否在预定义的离子/溶剂列表中。
    3. 如果是已知离子 -> 移入 ion_fragments。
    4. 如果不是已知离子 (即使很小) -> 保留为 main_mol 的一部分。
    5. 最后将所有保留的片段合并回一个 Mol 对象。
    """
    
    # --- 1. 定义白名单 (保持原有列表，可根据需要扩充) ---
    NEUTRAL_COMPLEXING_AGENTS = {
        'O': 'H2O', 'Cl': 'HCl', 'Br': 'HBr', 'I': 'HI', 'F': 'HF',
        'OS(=O)(=O)O': 'H2SO4', 'O=[N+]([O-])O': 'HNO3', 'OP(=O)(O)O': 'H3PO4',
        'CC(=O)O': 'Acetic Acid', 'C(=O)O': 'Formic Acid',
        'N': 'Ammonia'
    }

    POTENTIAL_IONIC_SPECIES = {
        # 单原子离子 (符号)
        'FE', 'CU', 'MN', 'CO', 'NI', 'MO', 'V', 'CR', 'ZN', 'MG', 'CA', 'CD', 'HG', 'PB', 'SN',
        'NA', 'K', 'LI', 'RB', 'CS', 'AG',
        'AL', 'GA', 'BI', 'IN', 'TL',
        'EU', 'GD', 'LA', 'YB', 'SM', 'CE', 'SR', 'BA',
        'CL', 'BR', 'I', 'F',
        
        # 多原子离子 (SMILES)
        'O', 'N',
        'O=S(=O)=O', 'O=[N+]=O', 'O=P(O)(O)O', 'O=P(O)O', 
        'O=C=O', 'CC(=O)=O', 'C(#N)[S]', 'O=C(C=O)=O', 
        'O=Cl(=O)=O', 'O=C(CC=O)=O', 'O=C(CCC=O)=O', 
        'O=C(O)C(O)(C=O)C=O', 'O=C([C@H](O)[C@@H](O)C=O)=O', 
        'O=P(O)(O)OCC(O)CO', 'O=P(O)(O)OP(=O)(O)O', 'O=C(C(F)(F)F)=O'
    }

    # --- 2. 辅助函数：判断是否为离子 ---
    def is_known_ion(fragment):
        # 移除氢以匹配标准 SMILES
        temp_mol = Chem.RemoveHs(fragment)
        try:
            # 1. 尝试标准 SMILES 匹配
            smi = Chem.MolToSmiles(temp_mol, canonical=True)
            if smi in POTENTIAL_IONIC_SPECIES or smi in NEUTRAL_COMPLEXING_AGENTS:
                return True, smi
            
            # 2. 尝试单原子符号匹配 (针对带电荷的金属离子，如 [Fe+3])
            if temp_mol.GetNumAtoms() == 1:
                symbol = temp_mol.GetAtomWithIdx(0).GetSymbol().upper()
                if symbol in POTENTIAL_IONIC_SPECIES:
                    return True, symbol
            
            return False, smi
        except:
            return False, ""

    # --- 3. 主流程 ---
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    
    if not fragments:
        return mol, []

    organic_parts = []
    ion_parts = []

    for frag in fragments:
        is_ion, lookup_key = is_known_ion(frag)
        
        if is_ion:
            # === 是已知离子，进行标记并移入 ion_parts ===
            
            # 标记逻辑 (原逻辑保留)
            if lookup_key in NEUTRAL_COMPLEXING_AGENTS:
                # 中性溶剂/络合剂 -> **
                frag.SetProp("_Name", lookup_key + "**")
            else:
                # 普通离子 -> *** (如果原来是单原子可能需要特殊处理，这里简化)
                try:
                    original_smiles = Chem.MolToSmiles(frag)
                    frag.SetProp("_Name", original_smiles + "***")
                except:
                    pass
            
            ion_parts.append(frag)
        
        else:
            # === 不是已知离子，归入主结构 ===
            organic_parts.append(frag)

    # --- 4. 结果组装 ---
    
    # 情况 A: 所有的片段都被识别为离子 (极端情况，比如全是水)
    if not organic_parts:
        # 这种情况下，取最大的那个“离子”作为主分子，其他的作为离子
        # 或者返回空主分子。这里为了稳健，如果全被识别为离子，
        # 我们把最大的那个“捞回来”当主分子
        ion_parts.sort(key=lambda m: m.GetNumAtoms(), reverse=True)
        main_mol = ion_parts.pop(0)
        # 清除标记
        if main_mol.HasProp("_Name"): main_mol.ClearProp("_Name")
        return main_mol, ion_parts

    # 情况 B: 有多个有机片段 (你的案例)
    # 使用 CombineMols 将它们合并为一个 Mol 对象
    # 这样它们虽然不共价连接，但会作为同一个 Mol 对象进入后续的 Rule 2-9 切割流程
    main_mol = organic_parts[0]
    for i in range(1, len(organic_parts)):
        main_mol = Chem.CombineMols(main_mol, organic_parts[i])

    return main_mol, ion_parts

def rule2_cut_metals(fragments):
    """规则2: 切断金属连接。将有机金属化合物中的金属-配体键切断。"""
    metal_nums = {3, 4, 11, 12, 13, 19, 20, *range(21, 32), *range(37, 51), *range(55, 84)}
    output = []
    for frag in fragments:
        bonds_to_cut = [b.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() in metal_nums for b in a.GetBonds()]
        if bonds_to_cut:
            newly_fragmented = Chem.FragmentOnBonds(frag, bonds_to_cut, addDummies=True)
            new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
            output.extend([f for f in new_frags if get_num_non_dummy_atoms(f) > 0])
        else:
            output.append(frag)
    return output

def rule3_cut_large_rings(fragments, bond_info):
    """
    规则3 (增强版): 切割大环。
    逻辑更新：
    1. 优先处理显式的 8 元及以上大环（保持原有逻辑）。
    2. 如果存在“隐式大环”（即键在环上，但不属于任何 < 8 的小环），也将其视为大环的一部分进行切割。
       这解决了如环糊精（Cyclodextrin）等由小环串联成的笼状/大环结构无法被切割的问题。
    """
    output = []
    for frag in fragments:
        # --- 步骤 1: 收集小环信息 ---
        # 获取所有环的键列表
        all_ring_bonds = frag.GetRingInfo().BondRings()
        
        # 记录所有属于 "小环" (size < 8) 的键的索引
        # 如果一个键属于 6元环，它就被保护起来，除非它是大环唯一的切点
        bonds_in_small_rings = set()
        for ring in all_ring_bonds:
            if len(ring) < 8:
                bonds_in_small_rings.update(ring)

        # --- 步骤 2: 寻找显式大环 (原有逻辑) ---
        large_rings_bonds = [ring for ring in all_ring_bonds if len(ring) >= 8]
        
        candidate_to_cut = None # 存储最佳切割键 {'idx': int, 'ecc': float}

        if large_rings_bonds:
            # 策略 A: 存在显式大环 (如 12-crown-4)
            largest_ring_bonds = max(large_rings_bonds, key=len)
            
            # 优先找 d 类键 (连接桥头但不共用)
            candidates = [{'idx': idx, 'ecc': bond_info.get(idx, {}).get('ecc', 999)} 
                          for idx in largest_ring_bonds 
                          if (info := bond_info.get(idx)) and info['is_d']]
            
            # 其次找普通单环键 (非桥头)
            if not candidates:
                candidates = [{'idx': idx, 'ecc': bond_info.get(idx, {}).get('ecc', 999)} 
                              for idx in largest_ring_bonds 
                              if (info := bond_info.get(idx)) and info['is_a'] and not info['is_b']]
            
            if candidates:
                candidate_to_cut = min(candidates, key=lambda x: x['ecc'])

        # --- 步骤 3: 寻找隐式大环 (新增逻辑) ---
        # 如果策略 A 没有找到切点 (或者根本没有显式大环)，尝试策略 B
        if not candidate_to_cut:
            implicit_candidates = []
            for bond in frag.GetBonds():
                idx = bond.GetIdx()
                # 核心逻辑：键在环内，但不在任何小环内，且是单键
                if (bond.IsInRing() and 
                    idx not in bonds_in_small_rings and 
                    bond.GetBondType() not in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]):
                    
                    # 获取该键的偏心率
                    ecc = bond_info.get(idx, {}).get('ecc', 999)
                    implicit_candidates.append({'idx': idx, 'ecc': ecc})
            
            if implicit_candidates:
                # 同样选择最中心的键进行切割
                candidate_to_cut = min(implicit_candidates, key=lambda x: x['ecc'])

        # --- 步骤 4: 执行切割 ---
        if candidate_to_cut:
            try:
                # 执行切割
                newly_fragmented = Chem.FragmentOnBonds(frag, [candidate_to_cut['idx']], addDummies=True)
                new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                # 过滤掉只含虚原子的碎片
                valid_frags = [f for f in new_frags if get_num_non_dummy_atoms(f) > 0]
                if valid_frags:
                    output.extend(valid_frags)
                else:
                    output.append(frag)
            except Exception:
                output.append(frag)
        else:
            # 既没有显式大环，也没有隐式大环结构，保持原样
            output.append(frag)
            
    return output

def rule4_cut_ring_substituents(fragments):
    """
    规则4 (修复版): 切割环上取代基。
    
    修复内容：
    在遍历环原子的邻居时，显式排除原子序数为 0 的虚原子 (Dummy Atoms)。
    防止将上一轮切割产生的 '*' 标记误判为新的取代基从而导致无限切割或核心丢失。
    """
    final_fragments = []
    to_process = list(fragments)
    iteration_limit = 100 
    
    while to_process and iteration_limit > 0:
        frag = to_process.pop(0)
        
        if frag.GetRingInfo().NumRings() == 0:
            final_fragments.append(frag)
            continue
            
        bonds_to_cut = set()
        ring_atom_indices = {a.GetIdx() for a in frag.GetAtoms() if a.IsInRing()}
        
        for r_idx in ring_atom_indices:
            r_atom = frag.GetAtomWithIdx(r_idx)
            
            for bond in r_atom.GetBonds():
                neighbor = bond.GetOtherAtom(r_atom)
                
                # [关键修复]：如果邻居是虚原子，直接跳过！
                # 这阻止了对切割位点的二次切割
                if neighbor.GetAtomicNum() == 0:
                    continue
                    
                n_idx = neighbor.GetIdx()
                
                # 情况 A: 内部环键 -> 跳过
                if n_idx in ring_atom_indices and bond.IsInRing():
                    continue
                
                # 情况 B: 环间连接子
                if n_idx in ring_atom_indices and not bond.IsInRing():
                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        bonds_to_cut.add(bond.GetIdx())
                    continue
                
                # 情况 C: 环-取代基 连接
                if n_idx not in ring_atom_indices:
                    # 再次确保取代基不是虚原子 (双重保险)
                    if neighbor.GetAtomicNum() == 0:
                        continue

                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        bonds_to_cut.add(bond.GetIdx())
                    else:
                        target_bond_idx = _bfs_find_nearest_single_bond(frag, n_idx, r_idx)
                        if target_bond_idx is not None:
                            bonds_to_cut.add(target_bond_idx)

        if bonds_to_cut:
            try:
                newly_fragmented = Chem.FragmentOnBonds(frag, list(bonds_to_cut), addDummies=True)
                new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                valid_frags = [f for f in new_frags if get_num_non_dummy_atoms(f) > 0]
                
                if valid_frags:
                    to_process.extend(valid_frags)
                    iteration_limit -= 1
                else:
                    final_fragments.append(frag)
            except Exception:
                final_fragments.append(frag)
        else:
            final_fragments.append(frag)
            
    return final_fragments

def rule5_cut_acyclic_chains(fragments):
    """
    规则5 (安全修复版): 切割无环长链。
    
    修复内容：
    在 Check 1 (P2逻辑) 和 Check 2 (杂原子检查) 中，显式排除连接虚原子(AtomicNum=0)的键。
    防止程序试图切断 "C-*" 这种连接标记，避免死循环或错误碎片。
    """
    final_fragments = []
    fragments_to_process = list(fragments)
    processed_smiles = set()

    while fragments_to_process:
        current_fragment = fragments_to_process.pop(0)        
        try:
            # 1. 更新属性缓存，允许 RDKit 重新计算价态
            current_fragment.UpdatePropertyCache(strict=False)            
            # 2. 尝试快速修复 (Fast Sanitize)
            Chem.SanitizeMol(current_fragment, 
                             sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP)
        except Exception:
            pass
        try:
            current_smi = Chem.MolToSmiles(current_fragment, canonical=True)
            if current_smi in processed_smiles:
                final_fragments.append(current_fragment)
                continue
            processed_smiles.add(current_smi)
        except Exception:
            final_fragments.append(current_fragment)
            continue
        
        if current_fragment.GetRingInfo().NumRings() > 0 or get_num_non_dummy_atoms(current_fragment) < 3:
            final_fragments.append(current_fragment)
            continue
            
        # ==================================================================
        # Check 1: 原始 Rule 5 逻辑 (基于 Carbon 'a' 的切割)
        # ==================================================================
        bonds_to_cut_this_iteration = set()
        carbon_a_indices = identify_carbon_a_atoms(current_fragment)

        # P1 Bonds: 连接两个 Carbon 'a' 的单键 (优先切断官能团之间的连接)
        p1_bonds = {b.GetIdx() for b in current_fragment.GetBonds() 
                    if b.GetBondType() == Chem.BondType.SINGLE 
                    and b.GetBeginAtomIdx() in carbon_a_indices 
                    and b.GetEndAtomIdx() in carbon_a_indices}
        
        if p1_bonds:
            bonds_to_cut_this_iteration.update(p1_bonds)
        else:
            # P2 Bonds: Carbon 'a' 周围的其他特定键
            p2_bonds = set()
            for a_idx in carbon_a_indices:
                atom_a = current_fragment.GetAtomWithIdx(a_idx)
                
                # --- [逻辑重构] ---
                # 不使用模糊的 is_unsaturated，而是显式检查是否存在刚性双/三键
                has_rigid_unsaturation = False
                for bond in atom_a.GetBonds():
                    if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                        has_rigid_unsaturation = True
                        break
                
                if has_rigid_unsaturation:
                    # === 分支 A: 真正的官能团中心 (如 C=O, C=N, C=C) ===
                    # 这里的逻辑是为了保护官能团不被拆散
                    
                    unsaturated_partner_is_heteroatom = False
                    for bond in atom_a.GetBonds():
                        if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                            partner_atom = bond.GetOtherAtom(atom_a)
                            if partner_atom.GetAtomicNum() not in [6, 1, 0]:
                                unsaturated_partner_is_heteroatom = True
                                break
                    
                    if unsaturated_partner_is_heteroatom:
                        # 场景: 保护 C=O, C=N 等杂原子双键
                        # 策略: 切断连接该碳原子的"碳-碳单键"，保留 C-Het 连接
                        for bond in atom_a.GetBonds():
                            other = bond.GetOtherAtom(atom_a)
                            if other.GetAtomicNum() == 0: continue # 避开虚原子
                            
                            # 只切断连接 Carbon 的单键 (隔离整个官能团)
                            if bond.GetBondType() == Chem.BondType.SINGLE and other.GetAtomicNum() == 6:
                                p2_bonds.add(bond.GetIdx())
                    else:
                        # 场景: 纯碳双键 (C=C)
                        # 策略: 切断该碳原子的所有单键
                        for bond in atom_a.GetBonds():
                            other = bond.GetOtherAtom(atom_a)
                            if other.GetAtomicNum() == 0: continue
                            
                            if bond.GetBondType() == Chem.BondType.SINGLE:
                                p2_bonds.add(bond.GetIdx())

                else:
                    # === 分支 B: 表面饱和 (或仅含 Aromatic 伪影) 的 Carbon 'a' ===
                    # 用户需求: 只要没有双键/三键，就强制切断与杂原子的连接
                    # 这能有效处理 CC(C)CCNC=O 中的 C-N 键，无论它是 Single 还是 Aromatic
                    
                    for bond in atom_a.GetBonds():
                        other_atom = bond.GetOtherAtom(atom_a)
                        
                        # 1. 安全检查: 忽略虚原子
                        if other_atom.GetAtomicNum() == 0: continue

                        # 2. 目标检查: 对方必须是杂原子 (非C, 非H)
                        if other_atom.GetAtomicNum() not in [6, 1, 0]:
                            
                            # 3. 动作: 切断连接
                            # 只要不是双键/三键 (前面已经 check 过了，这里是 else 分支)，统统切断
                            # 包含了 SINGLE 和 AROMATIC (伪影)
                            if bond.GetBondType() not in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                                p2_bonds.add(bond.GetIdx())
            
            bonds_to_cut_this_iteration.update(p2_bonds)

        if bonds_to_cut_this_iteration:
            newly_fragmented_mol = Chem.FragmentOnBonds(current_fragment, list(bonds_to_cut_this_iteration), addDummies=True)
            new_frags = Chem.GetMolFrags(newly_fragmented_mol, asMols=True)
            fragments_to_process.extend([frag for frag in new_frags if get_num_non_dummy_atoms(frag) > 0])
            continue

        # ==================================================================
        # Check 2: 新增逻辑 (杂原子长链检查)
        # ==================================================================
        
        heteroatom_count = sum(1 for a in current_fragment.GetAtoms() if a.GetAtomicNum() not in (6, 1, 0))
        
        if heteroatom_count >= 5:
            try:
                d_mat = GetDistanceMatrix(current_fragment)
                atom_eccentricities = d_mat.max(axis=1)
                
                best_cut_bond_idx = None
                min_bond_ecc = float('inf')
                
                for bond in current_fragment.GetBonds():
                    # [修复点 4] 关键：如果键的任一端是虚原子，跳过！
                    if bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0:
                        continue

                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        b_idx = bond.GetBeginAtomIdx()
                        e_idx = bond.GetEndAtomIdx()
                        bond_ecc = (atom_eccentricities[b_idx] + atom_eccentricities[e_idx]) / 2.0
                        
                        if bond_ecc < min_bond_ecc:
                            min_bond_ecc = bond_ecc
                            best_cut_bond_idx = bond.GetIdx()
                
                if best_cut_bond_idx is not None:
                    newly_fragmented = Chem.FragmentOnBonds(current_fragment, [best_cut_bond_idx], addDummies=True)
                    new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                    fragments_to_process.extend([frag for frag in new_frags if get_num_non_dummy_atoms(frag) > 0])
                    continue

            except Exception:
                pass

        final_fragments.append(current_fragment)
            
    return final_fragments

def rule6_cut_spiro_centers(fragments, main_mol_for_matching, bond_info):
    """规则6: 切割螺环中心。将通过单个螺原子连接的两个环分离开。"""
    to_process = list(fragments)
    while True:
        cut_made, next_process, processed = False, [], []
        for frag in to_process:
            best_cut = {'bonds': [], 'priority_score': float('inf')}
            clean = Chem.RWMol(frag); [clean.RemoveAtom(i) for i in sorted([a.GetIdx() for a in clean.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)]
            match = main_mol_for_matching.GetSubstructMatch(clean)
            if not match: processed.append(frag); continue
            
            b_head_atoms = {idx for b_idx, info in bond_info.items() if info['is_b'] for idx in (main_mol_for_matching.GetBondWithIdx(b_idx).GetBeginAtomIdx(), main_mol_for_matching.GetBondWithIdx(b_idx).GetEndAtomIdx()) if idx in match}
            spiro_atoms = [a for a in frag.GetAtoms() if a.GetDegree() >= 4 and frag.GetRingInfo().NumAtomRings(a.GetIdx()) >= 2 and (match[a.GetIdx()] if a.GetIdx() < len(match) else -1) not in b_head_atoms]
            if not spiro_atoms: processed.append(frag); continue
            
            for atom in spiro_atoms:
                for pair in combinations(atom.GetBonds(), 2):
                    indices = [b.GetIdx() for b in pair]
                    temp_frag_mol = Chem.FragmentOnBonds(frag, indices, addDummies=True)
                    subs = list(Chem.GetMolFrags(temp_frag_mol, asMols=True))
                    if len(subs) == 2:
                        score = abs(rdMolDescriptors.CalcExactMolWt(subs[0]) - rdMolDescriptors.CalcExactMolWt(subs[1]))
                        if score < best_cut['priority_score']:
                            best_cut = {'bonds': indices, 'priority_score': score}
            
            if best_cut['bonds']:
                newly_fragmented = Chem.FragmentOnBonds(frag, best_cut['bonds'], addDummies=True)
                new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                next_process.extend([f for f in new_frags if get_num_non_dummy_atoms(f) > 0])
                cut_made = True
            else: processed.append(frag)
        
        to_process = processed + next_process
        if not cut_made: break
    return to_process

def rule7_cut_multi_ring_fragments(fragments, main_mol_for_matching, bond_info):
    """规则7: 切割多环片段。处理复杂的稠环或桥环体系。"""
    to_process = list(fragments)
    while True:
        cut_made, next_process, processed = False, [], []
        for frag in to_process:
            if frag.GetRingInfo().NumRings() < 3:
                processed.append(frag); continue
            
            clean = Chem.RWMol(frag); [clean.RemoveAtom(i) for i in sorted([a.GetIdx() for a in clean.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)]
            match = main_mol_for_matching.GetSubstructMatch(clean)
            if not match: processed.append(frag); continue
            
            cand_rings = find_type_a_rings(frag, main_mol_for_matching, match, bond_info)
            if not cand_rings: processed.append(frag); continue
            
            ecc_frag = GetDistanceMatrix(frag).max(axis=1)
            best_ring = min(cand_rings, key=lambda r: np.mean([ecc_frag[idx] for idx in r['atom_indices']]))
            
            if best_ring:
                map_o2f = {orig_idx: frag_idx for frag_idx, orig_idx in enumerate(match)}
                frag_indices = [b.GetIdx() for o_idx in best_ring['orig_d_bond_indices'] if (o_bond := main_mol_for_matching.GetBondWithIdx(o_idx)) and (f_a1 := map_o2f.get(o_bond.GetBeginAtomIdx())) is not None and (f_a2 := map_o2f.get(o_bond.GetEndAtomIdx())) is not None and (b := frag.GetBondBetweenAtoms(f_a1, f_a2))]
                if frag_indices:
                    newly_fragmented = Chem.FragmentOnBonds(frag, frag_indices, addDummies=True)
                    new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                    next_process.extend([f for f in new_frags if get_num_non_dummy_atoms(f) > 0])
                    cut_made = True
                else: processed.append(frag)
            else: processed.append(frag)
        to_process = processed + next_process
        if not cut_made: break
    return to_process

def rule8_cut_large_carbon_fragments(fragments, main_mol_for_matching):
    """规则8: 切割大的纯碳片段。处理残留的、大于6个碳的脂肪片段。"""
    final_fragments = []
    to_process = list(fragments)
    while to_process:
        frag = to_process.pop(0)
        
        is_all_carbon = all(a.GetAtomicNum() == 6 for a in frag.GetAtoms() if a.GetAtomicNum() != 0)
        num_carbons = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() == 6)
        
        if not is_all_carbon or num_carbons < 4:
            final_fragments.append(frag); continue
            
        best_cut = {'bond_idx': -1, 'mass_diff': float('inf'), 'avg_idx': float('inf')}
        clean_frag = Chem.RWMol(frag); [clean_frag.RemoveAtom(i) for i in sorted([a.GetIdx() for a in clean_frag.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)]
        match_indices = main_mol_for_matching.GetSubstructMatch(clean_frag.GetMol())
        if not match_indices:
            final_fragments.append(frag); continue

        for bond in frag.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE: continue
            
            try:
                temp_frag_mol = Chem.FragmentOnBonds(frag, [bond.GetIdx()], addDummies=True)
                subs = list(Chem.GetMolFrags(temp_frag_mol, asMols=True))
                if len(subs) == 2:
                    mass_diff = abs(rdMolDescriptors.CalcExactMolWt(subs[0]) - rdMolDescriptors.CalcExactMolWt(subs[1]))
                    avg_idx = (match_indices[bond.GetBeginAtomIdx()] + match_indices[bond.GetEndAtomIdx()]) / 2.0
                    
                    if mass_diff < best_cut['mass_diff'] or (mass_diff == best_cut['mass_diff'] and avg_idx < best_cut['avg_idx']):
                        best_cut = {'bond_idx': bond.GetIdx(), 'mass_diff': mass_diff, 'avg_idx': avg_idx}
            except Exception:
                continue

        if best_cut['bond_idx'] != -1:
            newly_fragmented = Chem.FragmentOnBonds(frag, [best_cut['bond_idx']], addDummies=True)
            new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
            to_process.extend([f for f in new_frags if get_num_non_dummy_atoms(f) > 0])
        else:
            final_fragments.append(frag)
    return final_fragments

def rule9_cut_acid_like_groups(fragments):
    """
    规则9: 切割含多个类酸官能团的片段。
    """
    final_fragments = []
    to_process = list(fragments)

    while to_process:
        frag = to_process.pop(0)

        def is_target_heteroatom(atom):
            if atom.GetAtomicNum() == 0: return False # 虚原子不是目标
            if atom.GetSymbol() not in ['P', 'S', 'N', 'B', 'C', 'Si', 'Se', 'Mn', 'As', 'Cr', 'Cl', 'Br', 'I']:
                return False
            # 检查双键氧和单键氧 (忽略连接到虚原子的键)
            has_double_bond_O = False
            has_single_bond_O = False
            for b in atom.GetBonds():
                other = b.GetOtherAtom(atom)
                if other.GetAtomicNum() == 0: continue # 忽略虚原子
                if other.GetAtomicNum() == 8:
                    if b.GetBondType() == Chem.BondType.DOUBLE: has_double_bond_O = True
                    if b.GetBondType() == Chem.BondType.SINGLE: has_single_bond_O = True
            
            return has_double_bond_O and has_single_bond_O

        current_frag_to_process = frag
        while True:
            target_atoms = [a for a in current_frag_to_process.GetAtoms() if is_target_heteroatom(a)]
            if len(target_atoms) < 2:
                final_fragments.append(current_frag_to_process)
                break

            candidate_bonds = []
            try:
                ecc_frag = GetDistanceMatrix(current_frag_to_process).max(axis=1)
            except:
                final_fragments.append(current_frag_to_process)
                break

            target_atom_indices = {a.GetIdx() for a in target_atoms}

            for atom in target_atoms:
                for bond in atom.GetBonds():
                    if not (bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()): continue
                    
                    bridge_atom = bond.GetOtherAtom(atom)
                    
                    # [修复点 1] 忽略虚原子桥
                    if bridge_atom.GetAtomicNum() == 0: continue

                    is_bridge = False
                    for neighbor_of_bridge in bridge_atom.GetNeighbors():
                        if neighbor_of_bridge.GetAtomicNum() == 0: continue # 忽略虚原子邻居
                        
                        if neighbor_of_bridge.GetIdx() in target_atom_indices and neighbor_of_bridge.GetIdx() != atom.GetIdx():
                            is_bridge = True
                            break
                    if is_bridge:
                        bond_ecc = (ecc_frag[bond.GetBeginAtomIdx()] + ecc_frag[bond.GetEndAtomIdx()]) / 2.0
                        candidate_bonds.append({'idx': bond.GetIdx(), 'ecc': bond_ecc})
            
            # 如果没找到 P-O-P 这种桥，找 P-O-C
            if not candidate_bonds:
                for atom in target_atoms:
                    for bond in atom.GetBonds():
                        other_atom = bond.GetOtherAtom(atom)
                        # [修复点 2] 忽略虚原子
                        if other_atom.GetAtomicNum() == 0: continue
                        
                        if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                            if other_atom.GetAtomicNum() == 6:
                                bond_ecc = (ecc_frag[bond.GetBeginAtomIdx()] + ecc_frag[bond.GetEndAtomIdx()]) / 2.0
                                candidate_bonds.append({'idx': bond.GetIdx(), 'ecc': bond_ecc})

            if not candidate_bonds:
                final_fragments.append(current_frag_to_process)
                break

            bond_to_cut = min(candidate_bonds, key=lambda x: x['ecc'])
            
            try:
                newly_fragmented = Chem.FragmentOnBonds(current_frag_to_process, [bond_to_cut['idx']], addDummies=True)
                new_frags_list = Chem.GetMolFrags(newly_fragmented, asMols=True)
                filtered_frags = [f for f in new_frags_list if get_num_non_dummy_atoms(f) > 0]
                
                if filtered_frags:
                    filtered_frags.sort(key=get_num_non_dummy_atoms, reverse=True)
                    current_frag_to_process = filtered_frags.pop(0)
                    to_process.extend(filtered_frags)
                else:
                    final_fragments.append(current_frag_to_process)
                    break
            except Exception:
                final_fragments.append(current_frag_to_process)
                break

    return final_fragments


# ----------------------------------------
# PDB 解析与口袋截取
# ----------------------------------------

# 已知溶剂/缓冲液/离子残基名称（从配体候选中排除）
KNOWN_SOLVENT_RESIDUES = {
    'HOH', 'WAT', 'H2O', 'DOD', 'D2O',
    'SO4', 'PO4', 'NO3', 'CO3', 'CL', 'BR', 'IOD', 'FLC',
    'GOL', 'EDO', 'PEG', 'PGE', 'MPD', 'DMS', 'ACE', 'NH2',
    'BME', 'MES', 'TRS', 'EPE', 'FMT', 'ACT', 'IMD', 'IPA',
    'CA', 'ZN', 'MG', 'NA', 'K', 'MN', 'FE', 'NI', 'CU', 'CO',
    'CD', 'HG', 'PB', 'SR', 'BA', 'CS', 'RB', 'LI'
}


def parse_hetatm_residues(pdb_path, exclude_residues=None):
    """
    解析 PDB 文件，识别所有非溶剂的 HETATM 残基作为配体候选。
    同时收集所有 ATOM（蛋白）记录以备后用。

    Args:
        pdb_path: PDB 文件路径
        exclude_residues: 要排除的残基名称集合，默认使用 KNOWN_SOLVENT_RESIDUES

    Returns:
        ligand_candidates: list[dict]，每个元素包含:
            - 'resName', 'chainID', 'resSeq', 'label'
            - 'atom_lines': 原始 PDB 行列表
            - 'coords': np.ndarray (N, 3)
        protein_lines: list[str]，所有 ATOM 记录行
    """
    if exclude_residues is None:
        exclude_residues = KNOWN_SOLVENT_RESIDUES

    hetatm_groups = {}  # key=(chainID, resName, resSeq) -> {'lines':[], 'coords':[]}
    protein_lines = []

    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            record = line[:6].strip()
            if record == 'ATOM':
                protein_lines.append(line)
            elif record == 'HETATM':
                try:
                    resName = line[17:20].strip()
                    chainID = line[21].strip() if len(line) > 21 else ''
                    resSeq = line[22:26].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue

                key = (chainID, resName, resSeq)
                if key not in hetatm_groups:
                    hetatm_groups[key] = {'lines': [], 'coords': []}
                hetatm_groups[key]['lines'].append(line)
                hetatm_groups[key]['coords'].append([x, y, z])

    # 过滤：排除溶剂/离子
    ligand_candidates = []
    for (chainID, resName, resSeq), data in hetatm_groups.items():
        if resName.upper() in exclude_residues:
            continue
        # 至少2个原子才可能是配体（排除单原子金属残基）
        if len(data['coords']) < 2:
            continue
        ligand_candidates.append({
            'resName': resName,
            'chainID': chainID,
            'resSeq': resSeq,
            'label': f"{resName}:{chainID}:{resSeq} ({len(data['coords'])} atoms)",
            'atom_lines': data['lines'],
            'coords': np.array(data['coords'])
        })

    return ligand_candidates, protein_lines


def extract_pocket_around_ligand(pdb_path, ligand_atom_coords, protein_lines=None, radius=12.0):
    """
    根据配体原子坐标，从蛋白 ATOM 记录中截取 radius 范围内的完整残基。

    Args:
        pdb_path: PDB 文件路径（当 protein_lines 为 None 时用于读取）
        ligand_atom_coords: np.ndarray (M, 3)，配体原子坐标
        protein_lines: list[str]，预解析的 ATOM 行（可选）
        radius: 截取半径（Å），默认 12.0

    Returns:
        pocket_mol: RDKit Mol（口袋分子，已做容错处理）
        pocket_pdb_lines: list[str]，用于保存的 PDB 行
    """
    import scipy.spatial.distance

    # 如果没有预传入蛋白行，从文件读取
    if protein_lines is None:
        protein_lines = []
        with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line[:6].strip() == 'ATOM':
                    protein_lines.append(line)

    # 按残基分组 (chainID, resSeq, iCode)
    residue_groups = {}
    for line in protein_lines:
        try:
            chainID = line[21] if len(line) > 21 else ''
            resSeq = line[22:26].strip()
            iCode = line[26] if len(line) > 26 else ''
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue

        key = (chainID, resSeq, iCode)
        if key not in residue_groups:
            residue_groups[key] = {'lines': [], 'coords': []}
        residue_groups[key]['lines'].append(line)
        residue_groups[key]['coords'].append([x, y, z])

    # 判断每个残基是否在 radius 范围内
    pocket_lines = []
    for key, data in residue_groups.items():
        res_coords = np.array(data['coords'])
        dists = scipy.spatial.distance.cdist(res_coords, ligand_atom_coords)
        if dists.min() < radius:
            pocket_lines.extend(data['lines'])

    if not pocket_lines:
        return None, []

    # 用 PDB block 构建 RDKit Mol
    pdb_block = ''.join(pocket_lines) + 'END\n'
    pocket_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)

    if pocket_mol is not None:
        pocket_mol = advanced_rescue_valence_errors(pocket_mol)
        if pocket_mol is not None:
            try:
                pocket_mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(pocket_mol,
                                 sanitizeOps=Chem.SANITIZE_SYMMRINGS |
                                             Chem.SANITIZE_SETCONJUGATION |
                                             Chem.SANITIZE_SETHYBRIDIZATION)
            except:
                pass

    return pocket_mol, pocket_lines


def build_ligand_mol_from_pdb_lines(ligand_lines):
    """
    从 HETATM 行构建配体的 RDKit Mol 对象。
    用 OpenBabel 推断键级，再映射回 RDKit 的 PDB mol（保留原始坐标和原子顺序）。
    """
    pdb_block = ''.join(ligand_lines) + 'END\n'

    # 第一步：RDKit 读 PDB（保留完整的 3D 坐标和 MonomerInfo）
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)
    if mol is None:
        return None

    # 第二步：用 OpenBabel 推断键级，映射回 mol
    try:
        from openbabel import openbabel as ob
        ob_conv = ob.OBConversion()
        ob_conv.SetInFormat("pdb")
        ob_conv.SetOutFormat("sdf")
        ob_mol = ob.OBMol()
        ob_conv.ReadString(ob_mol, pdb_block)
        ob_mol.PerceiveBondOrders()
        sdf_block = ob_conv.WriteString(ob_mol)

        ref_mol = Chem.MolFromMolBlock(sdf_block, removeHs=True, sanitize=False)

        if ref_mol is not None and ref_mol.GetNumAtoms() == mol.GetNumAtoms():
            # 原子顺序一致，直接按索引复制键级
            rw = Chem.RWMol(mol)
            for bond in rw.GetBonds():
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                ref_bond = ref_mol.GetBondBetweenAtoms(idx1, idx2)
                if ref_bond is not None:
                    bond.SetBondType(ref_bond.GetBondType())
                    bond.SetIsAromatic(ref_bond.GetIsAromatic())
            # 同步原子芳香性标记
            for i in range(rw.GetNumAtoms()):
                rw.GetAtomWithIdx(i).SetIsAromatic(
                    ref_mol.GetAtomWithIdx(i).GetIsAromatic()
                )
            mol = rw.GetMol()
    except Exception:
        pass  # OpenBabel 不可用，保持原始 PDB 键级

    # 第三步：正常的抢救和 sanitize 流程
    mol = advanced_rescue_valence_errors(mol)
    if mol is not None:
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol,
                            sanitizeOps=Chem.SANITIZE_SYMMRINGS |
                                        Chem.SANITIZE_SETCONJUGATION |
                                        Chem.SANITIZE_SETHYBRIDIZATION |
                                        Chem.SANITIZE_SETAROMATICITY)
        except:
            pass

    return mol

# ----------------------------------------
# Kabsch 反映射与 PyG 节点操作
# ----------------------------------------

def kabsch_align_fragment_to_reference(smiles, target_centroid, target_ref_coords):
    """
    Kabsch 反映射：从 SMILES 生成 3D conformer，对齐到存储的参考坐标系。

    原理：
      正向编码: mol -> centroid + ref_coords[A,B,C] + SMILES
      反向解码: SMILES + centroid + ref_coords -> aligned 3D mol

    步骤：
      1. SMILES -> 3D conformer (EmbedMolecule)
      2. 计算 conformer 的 canonical 参考系和质心
      3. Kabsch SVD 求旋转矩阵 R
      4. 所有原子: new_pos = R @ (pos - centroid_gen) + target_centroid

    Args:
        smiles: 碎片的 canonical SMILES
        target_centroid: (3,) 目标质心坐标
        target_ref_coords: (3, 3) 目标框架坐标 [A, B, C]

    Returns:
        RDKit Mol（带对齐后的 3D conformer）或 None
    """
    from rdkit.Geometry import Point3D

    target_centroid = np.asarray(target_centroid, dtype=np.float64)
    target_ref_coords = np.asarray(target_ref_coords, dtype=np.float64)

    # 清理 SMILES（去除离子后缀标记）
    clean_smiles = smiles.rstrip('*')
    if clean_smiles in ('<UNK>', '<ERROR>', '') or not clean_smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(clean_smiles)
        if not mol:
            return None
        mol = Chem.AddHs(mol)

        # 生成 3D conformer
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            pass

        mol = Chem.RemoveHs(mol)

        if mol.GetNumConformers() == 0:
            return None

        num_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

        # --- 单原子：直接放在目标质心 ---
        if num_heavy <= 1:
            conf = mol.GetConformer(0)
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(
                    float(target_centroid[0]),
                    float(target_centroid[1]),
                    float(target_centroid[2])
                ))
            return mol

        # --- 多原子：Kabsch 对齐 ---
        gen_centroid = np.array(calculate_centroid(mol))
        gen_ref = calculate_reference_coordinate_system(mol)  # (3, 3)

        if gen_ref is None:
            return None

        # 中心化
        gen_ref_centered = gen_ref - gen_centroid
        target_ref_centered = target_ref_coords - target_centroid

        # Kabsch SVD
        H = gen_ref_centered.T @ target_ref_centered
        U, S, Vt = np.linalg.svd(H)

        # 处理反射
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, d])
        R = Vt.T @ sign_matrix @ U.T

        # 应用变换
        conf = mol.GetConformer(0)
        for i in range(mol.GetNumAtoms()):
            pos = np.array(list(conf.GetAtomPosition(i)))
            new_pos = R @ (pos - gen_centroid) + target_centroid
            conf.SetAtomPosition(i, Point3D(
                float(new_pos[0]), float(new_pos[1]), float(new_pos[2])
            ))

        return mol

    except Exception:
        return None


def remove_nodes_from_pyg_data(data, nodes_to_delete):
    """
    从 PyG Data 中移除指定节点及其关联的边，并重新映射边索引。

    Args:
        data: torch_geometric.data.Data
        nodes_to_delete: set[int]，要删除的节点索引集合

    Returns:
        新的 Data 对象
    """
    import torch
    from torch_geometric.data import Data

    total_nodes = data.x.size(0)
    kept_indices = sorted(set(range(total_nodes)) - set(nodes_to_delete))

    if not kept_indices:
        return None

    old_to_new = {old: new for new, old in enumerate(kept_indices)}

    kept_tensor = torch.tensor(kept_indices, dtype=torch.long)
    new_x = data.x[kept_tensor]
    new_pos = data.pos[kept_tensor] if hasattr(data, 'pos') and data.pos is not None else None
    new_frag_embeds = data.frag_embeds[kept_tensor] if hasattr(data, 'frag_embeds') and data.frag_embeds is not None else None
    new_ref_coords = data.ref_coords[kept_tensor] if hasattr(data, 'ref_coords') and data.ref_coords is not None else None

    if data.edge_index is not None and data.edge_index.size(1) > 0:
        edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
        for i in range(data.edge_index.size(1)):
            u = data.edge_index[0, i].item()
            v = data.edge_index[1, i].item()
            if u in old_to_new and v in old_to_new:
                edge_mask[i] = True

        new_edge_index = data.edge_index[:, edge_mask].clone()
        for i in range(new_edge_index.size(1)):
            new_edge_index[0, i] = old_to_new[new_edge_index[0, i].item()]
            new_edge_index[1, i] = old_to_new[new_edge_index[1, i].item()]

        new_edge_attr = data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = torch.empty((0, 5), dtype=torch.float)

    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        pdb_id=data.pdb_id if hasattr(data, 'pdb_id') else None,
        meb_center=data.meb_center if hasattr(data, 'meb_center') else None,
        n_pocket_nodes=data.n_pocket_nodes if hasattr(data, 'n_pocket_nodes') else None, # <-- 新增这一行
    )
    if new_pos is not None:
        new_data.pos = new_pos
    if new_frag_embeds is not None:
        new_data.frag_embeds = new_frag_embeds
    if new_ref_coords is not None:
        new_data.ref_coords = new_ref_coords

    return new_data