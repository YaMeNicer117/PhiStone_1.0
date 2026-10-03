import os
import glob
import shutil
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ==========================================
# 1. 超参数与配置区 (Hyperparameters)
# ==========================================
# 路径配置
INPUT_DIR = "processed_data/generated_molecules"
OUTPUT_DIR = "processed_data/filtered_molecules"

# 成药性与合成难度阈值
MIN_QED = 0.3          # QED下限 (越高越好，0.3属宽松先导物标准)
MAX_SASCORE = 6.0      # SAscore上限 (越低越好，>6通常极难合成)

# 环系统与结构复杂度阈值
MAX_RING_SIZE = 8          # 允许的最大单环尺寸 (防畸变大环)
MAX_SPIRO_ATOMS = 1        # 允许的最大螺原子数量 (防极度扭曲)
MAX_BRIDGEHEAD_ATOMS = 2   # 允许的最大桥头原子数量 (防复杂立体网状)
MAX_TOTAL_RINGS = 5        # 单个分子允许的最大环总数 (防无限稠合)

# ==========================================
# 2. 尝试导入 SAscore 计算模块
# ==========================================
try:
    # 通常 RDkit 的 contrib 目录下会有 SA_Score
    from rdkit.Chem import RDConfig
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    HAS_SASCORER = True
except ImportError:
    print("[警告] 未找到 RDKit 的 sascorer 模块。请确保 RDKit 完整安装。")
    print("如缺失，可从 RDKit GitHub 的 Contrib/SA_Score 下载 sascorer.py 和 fpscores.pkl.gz")
    HAS_SASCORER = False

# ==========================================
# 3. 核心过滤函数
# ==========================================
def check_chemical_validity(mol):
    """检查基础化学合理性、化合价与自由基"""
    try:
        # RDKit 原生 Sanitize 会捕捉绝大部分化合价错误，并且原生允许 [N+] 季铵离子
        Chem.SanitizeMol(mol)
    except Exception:
        return False, "Sanitize_Failed"
    
    # 拒绝自由基
    if Descriptors.NumRadicalElectrons(mol) > 0:
        return False, "Has_Radicals"
    
    # 额外拦截异常高配位原子 (例如度数 > 4 的碳)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetDegree() > 4:
            return False, "Texas_Carbon"
        # 如果是 N，度数大于4绝对错误；度数为4且不带正电，也是错误的
        if atom.GetSymbol() == 'N':
            if atom.GetDegree() > 4:
                return False, "Hypervalent_Nitrogen"
            if atom.GetDegree() == 4 and atom.GetFormalCharge() <= 0:
                return False, "Neutral_4_Coord_Nitrogen"
                
    return True, "OK"

def check_ring_complexity(mol):
    """检查环系统复杂度(稠环、桥环、螺环、大环)"""
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    
    # 1. 检查总环数
    if len(atom_rings) > MAX_TOTAL_RINGS:
        return False, f"Too_Many_Rings({len(atom_rings)})"
        
    # 2. 检查最大环尺寸
    if any(len(ring) > MAX_RING_SIZE for ring in atom_rings):
        return False, "Macro_Ring_Exceeded"
        
    # 3. 检查螺环复杂度
    spiro_count = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    if spiro_count > MAX_SPIRO_ATOMS:
        return False, f"Too_Many_Spiro({spiro_count})"
        
    # 4. 检查桥环复杂度
    bridge_count = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    if bridge_count > MAX_BRIDGEHEAD_ATOMS:
        return False, f"Too_Many_Bridgeheads({bridge_count})"
        
    return True, "OK"

# ==========================================
# 4. 主干流水线
# ==========================================
def run_pipeline():
    if not os.path.exists(INPUT_DIR):
        print(f"[错误] 输入目录 {INPUT_DIR} 不存在！")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_counts = {}
    report_data = [["Original_Folder", "Pocket_ID", "New_Prefix", "QED", "SAscore"]]
    
    # 获取所有的 2_mmff94_optimized_ligand.sdf
    search_pattern = os.path.join(INPUT_DIR, "*", "2_mmff94_optimized_ligand.sdf")
    ligand_files = glob.glob(search_pattern)
    
    print(f"找到 {len(ligand_files)} 个待过滤的配体任务...")
    
    for sdf_path in ligand_files:
        run_folder = os.path.dirname(sdf_path)
        folder_name = os.path.basename(run_folder) # 例如: F1_run3
        
        # 解析靶点ID (假设命名为 {pocket_id}_run{X})
        pocket_id = folder_name.split('_run')[0] if '_run' in folder_name else folder_name
        
        complex_pdb_path = os.path.join(run_folder, "3_mmff94_complex.pdb")
        
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
        if len(suppl) == 0:
            continue
        mol = suppl[0]
        if mol is None:
            continue
            
        # --- 漏斗 1: 化学合理性验证 ---
        is_valid, chem_reason = check_chemical_validity(mol)
        if not is_valid:
            print(f"[-] {folder_name} 被过滤: {chem_reason}")
            continue
            
        # --- 漏斗 2: 环系统限制 ---
        is_valid, ring_reason = check_ring_complexity(mol)
        if not is_valid:
            print(f"[-] {folder_name} 被过滤: {ring_reason}")
            continue
            
        # --- 漏斗 3: QED 计算 ---
        qed_score = Descriptors.qed(mol)
        if qed_score < MIN_QED:
            print(f"[-] {folder_name} 被过滤: QED太低 ({qed_score:.2f})")
            continue
            
        # --- 漏斗 4: SAscore 计算 ---
        sa_score = 0.0
        if HAS_SASCORER:
            sa_score = sascorer.calculateScore(mol)
            if sa_score > MAX_SASCORE:
                print(f"[-] {folder_name} 被过滤: SAscore太高 ({sa_score:.2f})")
                continue
        
        # ==========================================
        # 通过所有过滤！执行保存逻辑
        # ==========================================
        if pocket_id not in success_counts:
            success_counts[pocket_id] = 0
        success_counts[pocket_id] += 1
        current_idx = success_counts[pocket_id]
        
        # 构建新名称和目标路径
        new_prefix = f"{pocket_id}_{current_idx}"
        target_folder = os.path.join(OUTPUT_DIR, new_prefix)
        os.makedirs(target_folder, exist_ok=True)
        
        new_sdf_path = os.path.join(target_folder, f"{new_prefix}.sdf")
        new_pdb_path = os.path.join(target_folder, f"{new_prefix}_complex.pdb")
        
        # 将分数写入分子的 Property 中保存
        mol.SetProp("QED", f"{qed_score:.3f}")
        if HAS_SASCORER:
            mol.SetProp("SAscore", f"{sa_score:.3f}")
        mol.SetProp("_Name", new_prefix)
        
        with Chem.SDWriter(new_sdf_path) as writer:
            writer.write(mol)
            
        # 复制对应的 PDB 文件 (如果存在)
        if os.path.exists(complex_pdb_path):
            shutil.copy2(complex_pdb_path, new_pdb_path)
            
        print(f"[+] 成功: {folder_name} -> {new_prefix} (QED: {qed_score:.2f}, SA: {sa_score:.2f})")
        report_data.append([folder_name, pocket_id, new_prefix, f"{qed_score:.3f}", f"{sa_score:.3f}"])

    # 写入摘要报告
    report_path = os.path.join(OUTPUT_DIR, "filter_report.csv")
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(report_data)
    
    print("\n" + "="*40)
    print(f"过滤完成！共保留 {sum(success_counts.values())} 个有效分子。")
    print(f"报告已保存至: {report_path}")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()