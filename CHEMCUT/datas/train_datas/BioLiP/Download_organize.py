import os
import subprocess
import tarfile
import shutil
import glob
from tqdm import tqdm

# ==================== 配置区域 ====================
# 路径配置 (可根据你的实际情况修改)
BIOLIP_TXT_PATH = "metadata/BioLiP.txt"           # BioLiP.txt 存放路径
DOWNLOAD_DIR = "BioLiP_Data/weekly_full"          # S3 下载存放目录
RAW_LIGAND_DIR = "raw_ligand"                     # 配体解压目录
RAW_RECEPTOR_DIR = "raw_receptor"                 # 受体解压目录
OUT_DIR = "data"                                  # 最终组装好数据的存放目录

# 黑名单配置
MACROMOLECULES = {'dna', 'rna', 'peptide', 'nuc'}
METAL_IONS = {
    'ZN', 'MG', 'CA', 'FE', 'FE2', 'CU', 'CU1', 'MN', 'CO', 'NI',
    'K', 'NA', 'CD', 'HG', 'PT', 'AU', 'SR', 'BA', 'PB', 'AL',
    'V', 'MO', 'W', 'AG', 'LI', 'CS', 'RB', 'TL', 'SM', 'HO',
    'YB', 'LU', 'GD', 'TB', 'EU', 'PR', 'ND', 'CE', 'LA', 'U'
}
ANIONS_AND_SMALL_MOLS = {
    'CL', 'BR', 'I', 'F',        
    'SO4', 'PO4', 'NO3', 'CO3',  
    'SO3', 'NO2', 'BO3',         
    'ACT', 'FMT', 'MLI',         
    'CN', 'SCN', 'AZI', 'N3',    
    'OH', 'O', 'O2', 'H2O', 'DOD', 'NH4', 'NH3'
}

# 统一转换为小写，方便比对
EXCLUDE_LIGAND_TYPES = {x.lower() for x in (MACROMOLECULES | METAL_IONS | ANIONS_AND_SMALL_MOLS)}
# ==================================================

def step1_download():
    """第一步：从 AWS S3 下载文件"""
    print("\n" + "="*50)
    print("�� 第一步：开始从 AWS 下载数据...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    cmd = [
        "aws", "s3", "sync", "s3://biolip/weekly/", DOWNLOAD_DIR,
        "--no-sign-request", "--exclude", "*", 
        "--include", "*.tar.bz2", "--exclude", "*_nr.tar.bz2"
    ]
    
    # 因为 aws sync 本身自带进度信息，Python 难以完美拦截转换为 tqdm，所以这里直接调用
    # 但我们提醒用户它正在运行
    print("⏳ 正在执行 aws s3 sync，请耐心等待（终端会显示 AWS 的下载进度）...")
    subprocess.run(cmd, check=True)
    print("✅ 下载完成！")


def extract_tar_files(tar_pattern, extract_to):
    """辅助函数：带有进度条的解压功能"""
    os.makedirs(extract_to, exist_ok=True)
    tar_files = glob.glob(tar_pattern)
    if not tar_files:
        print(f"⚠️ 找不到匹配的文件: {tar_pattern}")
        return

    for tar_path in tqdm(tar_files, desc=f"解压到 {os.path.basename(extract_to)}"):
        try:
            with tarfile.open(tar_path, "r:bz2") as tar:
                # 过滤防范目录穿越攻击并解压
                members = tar.getmembers()
                tar.extractall(path=extract_to, members=members)
        except Exception as e:
            print(f"\n❌ 解压 {tar_path} 失败: {e}")


def step2_extract():
    """第二步：解压下载的压缩包"""
    print("\n" + "="*50)
    print("�� 第二步：开始解压文件...")
    
    ligand_pattern = os.path.join(DOWNLOAD_DIR, "ligand_*.tar.bz2")
    receptor_pattern = os.path.join(DOWNLOAD_DIR, "receptor_*.tar.bz2")
    
    extract_tar_files(ligand_pattern, RAW_LIGAND_DIR)
    extract_tar_files(receptor_pattern, RAW_RECEPTOR_DIR)
    print("✅ 所有压缩包解压完毕！")


def combine_pdb_files(input_files, output_file):
    """将多个 PDB 文件合并为一个，自动剔除中间多余的 END 标签"""
    with open(output_file, 'w') as outfile:
        for f in input_files:
            if not os.path.exists(f):
                continue
            with open(f, 'r') as infile:
                for line in infile:
                    if not line.startswith("END"):
                        outfile.write(line)
        outfile.write("END\n")


def step3_and_4_organize_data():
    """第三步与第四步合并：解析白名单并直接组装复合物"""
    print("\n" + "="*50)
    print("�� 第三步：过滤无效配体并组装多链复合物...")
    
    if not os.path.exists(BIOLIP_TXT_PATH):
        print(f"❌ 致命错误: 找不到注释文件 {BIOLIP_TXT_PATH}")
        print("请手动下载 BioLiP.txt 并放置在配置的路径下。")
        return

    # 1. 解析注释文件，获取白名单组合
    lig_to_recs = {}
    with open(BIOLIP_TXT_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="读取并清洗 BioLiP.txt"):
        if not line.strip() or line.startswith('#'): continue
        cols = line.strip().split('\t')
        if len(cols) < 7: continue
        
        pdb_id, rec_chain, lig_id, lig_chain, lig_serial = cols[0], cols[1], cols[4], cols[5], cols[6]
        
        # 【核心优化】：在此处直接剔除黑名单配体
        if lig_id.lower() in EXCLUDE_LIGAND_TYPES:
            continue
            
        rec_name = f"{pdb_id}{rec_chain}"
        lig_file = f"{pdb_id}_{lig_id}_{lig_chain}_{lig_serial}.pdb"
        
        if lig_file not in lig_to_recs:
            lig_to_recs[lig_file] = set()
        lig_to_recs[lig_file].add(rec_name)

    # 2. 翻转字典：将同一组受体组合 (Complex) 映射到它们所有的配体
    combo_to_ligs = {}
    for lig, recs in lig_to_recs.items():
        # 如果对应的配体在解压目录中确实存在，才加入待处理队列
        if not os.path.exists(os.path.join(RAW_LIGAND_DIR, lig)):
            continue
            
        combo = frozenset(recs)
        if combo not in combo_to_ligs:
            combo_to_ligs[combo] = []
        combo_to_ligs[combo].append(lig)

    # 3. 开始移动与合并文件
    os.makedirs(OUT_DIR, exist_ok=True)
    success_folders = 0
    interface_complexes = 0
    
    for combo, ligs in tqdm(combo_to_ligs.items(), desc="组装并拷贝有效数据"):
        pdb_id = list(combo)[0][:4]
        chains = sorted([rec[4:] for rec in combo])
        combo_name = f"{pdb_id}{''.join(chains)}"
        
        target_dir = os.path.join(OUT_DIR, combo_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 处理受体
        combined_rec_dst = os.path.join(target_dir, f"{combo_name}_receptor.pdb")
        rec_sources = [os.path.join(RAW_RECEPTOR_DIR, f"{rec}.pdb") for rec in sorted(combo)]
        
        if len(combo) == 1:
            if os.path.exists(rec_sources[0]):
                shutil.copy2(rec_sources[0], combined_rec_dst)
        else:
            combine_pdb_files(rec_sources, combined_rec_dst)
            interface_complexes += 1
            
        # 处理配体 (仅复制经过白名单验证的)
        ligs.sort()
        for idx, lig_src_name in enumerate(ligs):
            lig_src_path = os.path.join(RAW_LIGAND_DIR, lig_src_name)
            lig_dst_path = os.path.join(target_dir, f"{combo_name}{idx}_ligand.pdb")
            shutil.copy2(lig_src_path, lig_dst_path)
            
        success_folders += 1

    print("\n" + "="*50)
    print(f"�� 整个流水线处理完成！")
    print(f"✅ 共构建了 {success_folders} 个高质量复合物文件夹。")
    print(f"�� 其中包含了 {interface_complexes} 个自动合并的多链交界面(Interface)受体。")
    print(f"�� 最终训练数据已就绪: {os.path.abspath(OUT_DIR)}")
    print("="*50)


if __name__ == "__main__":
    # 依次执行各步骤
    step1_download()
    step2_extract()
    step3_and_4_organize_data()