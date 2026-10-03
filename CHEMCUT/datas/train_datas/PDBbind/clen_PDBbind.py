import os
import glob
import shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import pdist, cdist
from tqdm import tqdm

# =============================================================================
# 抑制第三方库的底层警告
# =============================================================================
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# =============================================================================
# 配置路径与超参数
INPUT_DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(INPUT_DATA_ROOT)
OUTPUT_DATA_ROOT = os.path.join(parent_dir, 'PDBbind_Processed')

CUTOFF_RADIUS = 12.0          
MAX_LIGAND_SPAN = 18.0        
NUM_WORKERS = 28

# =============================================================================
# 核心辅助函数
# =============================================================================
def parse_pdb_lines_and_crop(pdb_lines, ligand_coords, cutoff=12.0):
    """读取口袋 PDB 行，并基于两两最短距离(pairwise shortest distance)进行裁剪"""
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

    rec_coords = np.array([atom['coord'] for atom in rec_atoms])

    # 使用 cdist 计算受体原子 (M) 到配体原子 (N) 的距离矩阵 (M x N)
    dist_matrix = cdist(rec_coords, ligand_coords)

    # 找出每个受体原子到配体所有原子中最短的那个距离 (M,)
    min_dists = np.min(dist_matrix, axis=1)

    # 筛选出最短距离 <= cutoff 的受体原子索引
    within_cutoff_indices = np.where(min_dists <= cutoff)[0]

    # 只要该残基有一个原子落在 cutoff 范围内，就将其完整记录下来
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
    except Exception:
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
        "rdkit_parse_error": 0,
        "empty_or_no_heavy_atoms": 0,
        "multiple_fragments": 0,
        "no_carbon_inorganic": 0,
        "span_too_large": 0,
        "coord_extract_fail": 0,
        "missing_pocket_pdb": 0
    }

    complex_id = os.path.basename(complex_dir)

    # 获取 SDF 格式的配体文件 (PDBbind 预先提供)
    sdf_files = glob.glob(os.path.join(complex_dir, "*_ligand.sdf"))
    if not sdf_files:
        return stats

    for i, sdf_path in enumerate(sdf_files):
        stats["total_scanned"] += 1

        # 寻找对应的 pocket.pdb
        pocket_path = sdf_path.replace('_ligand.sdf', '_pocket.pdb')
        if not os.path.exists(pocket_path):
            stats["missing_pocket_pdb"] += 1
            continue

        with open(pocket_path, 'r', encoding='utf-8') as f:
            pocket_lines = f.readlines()

        # 步骤 A: RDKit 状态分析
        status_code, lig_mol = analyze_ligand_status(sdf_path)
        if status_code != "success":
            stats[status_code] += 1
            continue

        # 步骤 B: 提取配体重原子坐标 (替代原先的 MEB 球心计算)
        heavy_atom_indices = [a.GetIdx() for a in lig_mol.GetAtoms() if a.GetAtomicNum() > 1]
        if not heavy_atom_indices:
            stats["coord_extract_fail"] += 1
            continue

        conf = lig_mol.GetConformer(0)
        ligand_coords = np.array([list(conf.GetAtomPosition(idx)) for idx in heavy_atom_indices])

        # 步骤 C & D: 以配体所有重原子为基准对官方 pocket 进行两两最短距离截取并保存
        cropped_pocket_lines = parse_pdb_lines_and_crop(pocket_lines, ligand_coords, CUTOFF_RADIUS)

        new_complex_dir_name = f"{complex_id}" if len(sdf_files) == 1 else f"{complex_id}_{i}"
        out_dir = os.path.join(OUTPUT_DATA_ROOT, new_complex_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        new_pocket_path = os.path.join(out_dir, f"{new_complex_dir_name}_pocket_{int(CUTOFF_RADIUS)}A.pdb")
        with open(new_pocket_path, 'w', encoding='utf-8') as f:
            f.writelines(cropped_pocket_lines)

        new_ligand_sdf_path = os.path.join(out_dir, f"{new_complex_dir_name}_ligand.sdf")
        # 拷贝而不是移动，保护原始数据集
        shutil.copy2(sdf_path, new_ligand_sdf_path)

        stats["success"] += 1

    return stats

# =============================================================================
# 主程序
# =============================================================================
if __name__ == '__main__':
    print(f"1. 开始扫描输入目录: {INPUT_DATA_ROOT}")
    if os.path.exists(OUTPUT_DATA_ROOT):
        print(f"检测到已存在的输出文件夹 {OUTPUT_DATA_ROOT}，正在重置以防混淆...")
        shutil.rmtree(OUTPUT_DATA_ROOT)
    os.makedirs(OUTPUT_DATA_ROOT, exist_ok=True)

    # 递归查找所有包含 _ligand.sdf 的文件夹
    print("正在递归检索 PDBbind 数据集...")
    all_sdf_files = glob.glob(os.path.join(INPUT_DATA_ROOT, '**', '*_ligand.sdf'), recursive=True)
    # 提取所有去重的父文件夹路径
    all_complex_dirs = list(set([os.path.dirname(f) for f in all_sdf_files]))

    print(f"2. 找到 {len(all_complex_dirs)} 个有效的复合物文件夹。")
    print(f"3. 核心执行策略: 读取SDF质检 -> 基于两两最短距离提取 {CUTOFF_RADIUS}Å 受体口袋，使用 {NUM_WORKERS} 线程并发")

    global_stats = {
        "total_scanned": 0,
        "success": 0,
        "rdkit_parse_error": 0,
        "empty_or_no_heavy_atoms": 0,
        "multiple_fragments": 0,
        "no_carbon_inorganic": 0,
        "span_too_large": 0,
        "coord_extract_fail": 0,
        "missing_pocket_pdb": 0
    }

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_complex, d) for d in all_complex_dirs]

        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            try:
                local_stats = future.result()
                for key in global_stats:
                    global_stats[key] += local_stats.get(key, 0)
            except Exception as e:
                pass

    print("\n" + "="*70)
    print("📋 PDBbind 数据清洗报告 (过滤详情分析)")
    print("="*70)
    print(f"总计扫描的配体数目      : {global_stats['total_scanned']}")
    print(f"最终保留的高质量数目    : {global_stats['success']}  (✅ 核心输出)")
    print("-" * 70)
    print("❌ 过滤掉的配体分类及数量:")
    print(f"1. 几何跨度大于 {MAX_LIGAND_SPAN}Å  : {global_stats['span_too_large']} (超大分子、多肽等)")
    print(f"2. 断裂/多片段组成的配体: {global_stats['multiple_fragments']} (网络结构、分散溶剂)")
    print(f"3. 无机物/不含碳原子    : {global_stats['no_carbon_inorganic']} (游离金属簇、无机盐)")
    print(f"4. 缺失对应的 pocket 文件: {global_stats['missing_pocket_pdb']} (数据不完整)")
    print(f"5. RDKit 加载失败       : {global_stats['rdkit_parse_error']} (化学价态不合法)")
    print(f"6. 空白或无重原子分子   : {global_stats['empty_or_no_heavy_atoms']} (如仅含有H或解析为空)")
    print(f"7. 坐标提取失败         : {global_stats['coord_extract_fail']} (罕见重原子缺失)")
    print("="*70)
    print(f"📁 最终清洗好的数据已保存在: {os.path.abspath(OUTPUT_DATA_ROOT)}")