import os
from tqdm import tqdm

# ==================== 配置区域 ====================
BIOLIP_TXT_PATH = "metadate/BioLiP.txt"  # 使用你当前目录下的相对路径
RAW_LIGAND_DIR = "raw_ligand/ligand"            # 你的配体解压目录

# 1. 大分子类
MACROMOLECULES = {'dna', 'rna', 'peptide', 'nuc'}

# 2. 常见金属离子 (Metal Ions)
METAL_IONS = {
    'ZN', 'MG', 'CA', 'FE', 'FE2', 'CU', 'CU1', 'MN', 'CO', 'NI',
    'K', 'NA', 'CD', 'HG', 'PT', 'AU', 'SR', 'BA', 'PB', 'AL',
    'V', 'MO', 'W', 'AG', 'LI', 'CS', 'RB', 'TL', 'SM', 'HO',
    'YB', 'LU', 'GD', 'TB', 'EU', 'PR', 'ND', 'CE', 'LA', 'U'
}

# 3. 单原子阴离子、无机酸根及常见小有机酸根/溶剂 (Anions & Solvents)
ANIONS_AND_SMALL_MOLS = {
    'CL', 'BR', 'I', 'F',        # 卤素阴离子
    'SO4', 'PO4', 'NO3', 'CO3',  # 硫酸根、磷酸根、硝酸根、碳酸根
    'SO3', 'NO2', 'BO3',         # 亚硫酸根、亚硝酸根、硼酸根
    'ACT', 'FMT', 'MLI',         # 乙酸根(Acetate)、甲酸根(Formate)、丙二酸根
    'CN', 'SCN', 'AZI', 'N3',    # 氰根、硫氰酸根、叠氮根
    'OH', 'O', 'O2', 'H2O', 'DOD', 'NH4', 'NH3' # 氢氧根、水、重水、氨等
}

# 合并所有黑名单，并统一转换为小写，方便后续进行不区分大小写的比对
EXCLUDE_LIGAND_TYPES = {x.lower() for x in (MACROMOLECULES | METAL_IONS | ANIONS_AND_SMALL_MOLS)}

# ！！！安全开关：设为 False 则只打印信息不删文件；确认无误后改为 True 执行真实物理删除 ！！！
ACTUALLY_DELETE = True
# ==================================================

def clean_biolip_ligands_only():
    if not os.path.exists(BIOLIP_TXT_PATH):
        print(f"❌ 找不到注释文件: {BIOLIP_TXT_PATH}")
        return

    print("1. 正在解析 BioLiP.txt 生成严格的小分子配体白名单 ...")
    valid_ligands = set()

    with open(BIOLIP_TXT_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析每一行
    for line in tqdm(lines, desc="读取注释文件"):
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        
        if len(cols) < 7:
            continue

        pdb_id = cols[0]           # 第1列: PDB ID 
        lig_id = cols[4]           # 第5列: 配体代号 (例如 HEM, ZN, SO4)
        lig_chain = cols[5]        # 第6列: 配体链
        lig_serial = cols[6]       # 第7列: 配体序号

        # 判断是否为真实的小分子（不在任何黑名单剔除列表中）
        if lig_id.lower() not in EXCLUDE_LIGAND_TYPES:
            # 构造文件名主体：pdbId_ligId_chain_serial
            ligand_name = f"{pdb_id}_{lig_id}_{lig_chain}_{lig_serial}"
            valid_ligands.add(ligand_name)

    print(f"✅ 解析完毕！剔除大分子和离子后，白名单共保留: {len(valid_ligands)} 个高质量小分子配体。")

    # -----------------
    # 2. 仅清理配体文件
    # -----------------
    if os.path.exists(RAW_LIGAND_DIR):
        print(f"\n2. 正在扫描配体目录: {RAW_LIGAND_DIR} ...")
        ligand_files = os.listdir(RAW_LIGAND_DIR)
        deleted_ligands = 0

        for fname in tqdm(ligand_files, desc="比对并清理无效配体"):
            # 取文件名主体（去掉可能存在的 .pdb 或 .mol2 后缀）
            name_body = os.path.splitext(fname)[0] 
            
            if name_body not in valid_ligands:
                file_path = os.path.join(RAW_LIGAND_DIR, fname)
                if ACTUALLY_DELETE:
                    os.remove(file_path)
                deleted_ligands += 1

        action_str = "物理删除了" if ACTUALLY_DELETE else "发现需要删除"
        print(f"-> 配体清理结果: 共 {action_str} {deleted_ligands} 个非目标配体文件。")
    else:
        print(f"\n⚠️ 配体目录不存在: {RAW_LIGAND_DIR}，请检查路径。")

    if not ACTUALLY_DELETE:
        print("\n" + "="*50)
        print("�� 当前为试运行模式（ACTUALLY_DELETE = False），未删除任何文件。")
        print("请确认数字合理后，将脚本第 34 行的 ACTUALLY_DELETE 改为 True 再次运行。")
        print("="*50)

if __name__ == "__main__":
    clean_biolip_ligands_only()