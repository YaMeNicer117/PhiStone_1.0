import os
import glob
import shutil
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import pdist
from tqdm import tqdm

# =============================================================================
# 抑制第三方库的底层警告，并修复导入方式
# =============================================================================
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

# 修复 OpenBabel 导入警告
from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)
openbabel.obErrorLog.StopLogging()
from openbabel import pybel

# =============================================================================
# 配置路径与超参数
# =============================================================================
INPUT_DATA_ROOT = 'data'             
OUTPUT_DATA_ROOT = 'datas_BioLiP'    

CUTOFF_RADIUS = 8.0                  
MAX_LIGAND_SPAN = 12.0                
NUM_WORKERS = min(os.cpu_count(), 384)

# =============================================================================
# 核心辅助函数
# =============================================================================
def calculate_minimum_enclosing_ball(mol_with_conf):
    heavy_atom_indices = [a.GetIdx() for a in mol_with_conf.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy_atom_indices:
        return None
        
    conf = mol_with_conf.GetConformer(0)
    points = np.array([list(conf.GetAtomPosition(i)) for i in heavy_atom_indices])
    
    center = np.mean(points, axis=0)
    for i in range(1, 151):
        dists_sq = np.sum((points - center)**2, axis=1)
        furthest_idx = np.argmax(dists_sq)
        p_i = points[furthest_idx]
        step = 1.0 / (i + 1)
        center = (1.0 - step) * center + step * p_i
        
    return center

def parse_pdb_lines_and_crop(pdb_lines, center_coord, cutoff=8.0):
    rec_atoms = []
    
    for idx, line in enumerate(pdb_lines):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                chain = line[21]
                res_seq = line[22:26]
                icode = line[26]
                res_id = (chain, res_seq, icode)
                
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                
                rec_atoms.append({'res_id': res_id, 'coord': [x, y, z]})
            except ValueError:
                continue

    if not rec_atoms:
        return pdb_lines

    coords = np.array([atom['coord'] for atom in rec_atoms])
    dists = np.linalg.norm(coords - center_coord, axis=1)
    
    within_cutoff_indices = np.where(dists <= cutoff)[0]
    keep_res_ids = set([rec_atoms[i]['res_id'] for i in within_cutoff_indices])
    
    new_pdb_lines = []
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            chain = line[21]
            res_seq = line[22:26]
            icode = line[26]
            res_id = (chain, res_seq, icode)
            if res_id in keep_res_ids:
                new_pdb_lines.append(line)
        elif line.startswith("TER") or line.startswith("END"):
            pass 
        else:
            new_pdb_lines.append(line)
            
    new_pdb_lines.append("END\n")
    return new_pdb_lines

def analyze_ligand_status(sdf_path):
    """
    分析并返回配体状态的标签
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
    try:
        mol = next(suppl)
    except StopIteration:
        return "rdkit_parse_error", None

    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return "empty_or_no_heavy_atoms", None

    # 1. 连通性检查 (单一片段)
    if len(Chem.GetMolFrags(mol)) > 1:
        return "multiple_fragments", None

    # 2. 有机物检查 (至少含有一个碳原子)
    has_carbon = any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    if not has_carbon:
        return "no_carbon_inorganic", None

    # 3. 几何跨度检查
    conf = mol.GetConformer(0)
    pos = conf.GetPositions()
    if len(pos) > 1:
        max_dist = np.max(pdist(pos))
        if max_dist > MAX_LIGAND_SPAN:
            return "span_too_large", None

    return "success", mol

# =============================================================================
# 多进程 Worker 函数
# =============================================================================
def process_single_complex(complex_dir):
    stats = {
        "total_scanned": 0,
        "success": 0,
        "obabel_fail": 0,
        "rdkit_parse_error": 0,
        "empty_or_no_heavy_atoms": 0,
        "multiple_fragments": 0,
        "no_carbon_inorganic": 0,
        "span_too_large": 0,
        "meb_calc_fail": 0
    }
    
    complex_id = os.path.basename(complex_dir)
    receptor_files = glob.glob(os.path.join(complex_dir, "*_receptor.pdb"))
    ligand_files = glob.glob(os.path.join(complex_dir, "*_ligand.pdb"))
    
    if not receptor_files or not ligand_files:
        return stats
        
    with open(receptor_files[0], 'r', encoding='utf-8') as f:
        receptor_lines = f.readlines()
        
    for i, lig_pdb_path in enumerate(ligand_files):
        stats["total_scanned"] += 1
        sdf_path = lig_pdb_path.replace('.pdb', '.sdf')
        
        # 步骤 A: OpenBabel 转换
        try:
            ob_mol = next(pybel.readfile("pdb", lig_pdb_path))
            ob_mol.write("sdf", sdf_path, overwrite=True)
        except Exception:
            stats["obabel_fail"] += 1
            continue 
            
        # 步骤 B: RDKit 状态分析
        status_code, lig_mol = analyze_ligand_status(sdf_path)
        if status_code != "success":
            stats[status_code] += 1
            if os.path.exists(sdf_path): 
                os.remove(sdf_path)
            continue
            
        # 步骤 C: 计算球心
        center_coord = calculate_minimum_enclosing_ball(lig_mol)
        if center_coord is None:
            stats["meb_calc_fail"] += 1
            if os.path.exists(sdf_path): 
                os.remove(sdf_path)
            continue
            
        # 步骤 D & E: 裁剪与保存
        cropped_receptor_lines = parse_pdb_lines_and_crop(receptor_lines, center_coord, CUTOFF_RADIUS)
        
        new_complex_dir_name = f"{complex_id}_{i}"
        out_dir = os.path.join(OUTPUT_DATA_ROOT, new_complex_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        
        new_receptor_path = os.path.join(out_dir, f"{new_complex_dir_name}_receptor.pdb")
        with open(new_receptor_path, 'w', encoding='utf-8') as f:
            f.writelines(cropped_receptor_lines)
            
        new_ligand_sdf_path = os.path.join(out_dir, f"{new_complex_dir_name}_ligand.sdf")
        shutil.move(sdf_path, new_ligand_sdf_path)
        
        stats["success"] += 1
        
    return stats

# =============================================================================
# 主程序
# =============================================================================
if __name__ == '__main__':
    print(f"1. 开始扫描输入目录: {INPUT_DATA_ROOT}")
    # 确保输出目录干净，避免旧文件干扰
    if os.path.exists(OUTPUT_DATA_ROOT):
        print(f"检测到已存在的输出文件夹 {OUTPUT_DATA_ROOT}，正在重置以防混淆...")
        shutil.rmtree(OUTPUT_DATA_ROOT)
    os.makedirs(OUTPUT_DATA_ROOT, exist_ok=True)
    
    all_complex_dirs = [os.path.join(INPUT_DATA_ROOT, d) for d in os.listdir(INPUT_DATA_ROOT) 
                        if os.path.isdir(os.path.join(INPUT_DATA_ROOT, d))]
    
    print(f"2. 找到 {len(all_complex_dirs)} 个复合物文件夹。")
    print(f"3. 核心执行策略: 转换SDF -> 提取 {CUTOFF_RADIUS}Å 受体口袋，并记录清洗原因")
    
    # 全局统计字典
    global_stats = {
        "total_scanned": 0,
        "success": 0,
        "obabel_fail": 0,
        "rdkit_parse_error": 0,
        "empty_or_no_heavy_atoms": 0,
        "multiple_fragments": 0,
        "no_carbon_inorganic": 0,
        "span_too_large": 0,
        "meb_calc_fail": 0
    }
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_complex, d) for d in all_complex_dirs]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            try:
                local_stats = future.result()
                # 累加统计结果
                for key in global_stats:
                    global_stats[key] += local_stats.get(key, 0)
            except Exception as e:
                pass
                
    print("\n" + "="*70)
    print("�� 数据清洗报告 (过滤详情分析)")
    print("="*70)
    print(f"总计扫描的配体数目      : {global_stats['total_scanned']}")
    print(f"最终保留的高质量数目    : {global_stats['success']}  (✅ 核心输出)")
    print("-" * 70)
    print("❌ 过滤掉的配体分类及数量:")
    print(f"1. 几何跨度大于 {MAX_LIGAND_SPAN}Å  : {global_stats['span_too_large']} (超大分子、多肽等)")
    print(f"2. 断裂/多片段组成的配体: {global_stats['multiple_fragments']} (网络结构、分散溶剂)")
    print(f"3. 无机物/不含碳原子    : {global_stats['no_carbon_inorganic']} (游离金属簇、无机盐)")
    print(f"4. OpenBabel SDF转换失败: {global_stats['obabel_fail']} (底层 PDB 拓扑不完整)")
    print(f"5. RDKit 加载失败       : {global_stats['rdkit_parse_error']} (化学价态不合法)")
    print(f"6. 空白或无重原子分子   : {global_stats['empty_or_no_heavy_atoms']} (如仅含有H或解析为空)")
    print(f"7. 中心计算失败(MEB)    : {global_stats['meb_calc_fail']}")
    print("="*70)
    print(f"�� 最终数据已安全保存在: {os.path.abspath(OUTPUT_DATA_ROOT)}")