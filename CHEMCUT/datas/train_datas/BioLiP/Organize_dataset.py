import os
import shutil
from tqdm import tqdm

# ==================== 配置区域 ====================
BIOLIP_TXT_PATH = "metadate/BioLiP.txt"
RAW_LIGAND_DIR = "raw_ligand/ligand"
RAW_RECEPTOR_DIR = "raw_receptor/receptor"
OUT_DIR = "data"  # 使用新目录存放包含多链合并的数据
# ==================================================

def combine_pdb_files(input_files, output_file):
    """将多个 PDB 文件合并为一个，自动剔除中间多余的 END 标签"""
    with open(output_file, 'w') as outfile:
        for f in input_files:
            if not os.path.exists(f):
                continue
            with open(f, 'r') as infile:
                for line in infile:
                    # 过滤掉结束符，防止大部分解析工具（如 RDKit/BioPython）提前终止读取
                    if not line.startswith("END"):
                        outfile.write(line)
        # 在合并文件的最后补上一个 END
        outfile.write("END\n")

def organize_complexes_with_interfaces():
    if not os.path.exists(BIOLIP_TXT_PATH):
        print(f"❌ 找不到注释文件: {BIOLIP_TXT_PATH}")
        return

    print("1. 解析 BioLiP.txt 获取以配体为中心的映射关系...")
    # 字典结构: { '10gs_VWW_A_1.pdb': {'10gsA', '10gsB'} }
    lig_to_recs = {}
    
    with open(BIOLIP_TXT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 7:
                continue
            
            pdb_id = cols[0]
            rec_chain = cols[1]
            lig_id = cols[4]
            lig_chain = cols[5]
            lig_serial = cols[6]
            
            rec_name = f"{pdb_id}{rec_chain}"
            lig_file = f"{pdb_id}_{lig_id}_{lig_chain}_{lig_serial}.pdb"
            
            if lig_file not in lig_to_recs:
                lig_to_recs[lig_file] = set()
            lig_to_recs[lig_file].add(rec_name)

    print(f"✅ 从注释文件中找到了 {len(lig_to_recs)} 个小分子配体的结合记录。")
    
    # 翻转字典，按“受体组合(Complex)”进行归类
    # 比如: frozenset({'10gsA', '10gsB'}) 对应的所有配体
    combo_to_ligs = {}
    for lig, recs in lig_to_recs.items():
        # 安全检查：只处理那些被我们保留下来的有效小分子
        if not os.path.exists(os.path.join(RAW_LIGAND_DIR, lig)):
            continue
            
        combo = frozenset(recs)
        if combo not in combo_to_ligs:
            combo_to_ligs[combo] = []
        combo_to_ligs[combo].append(lig)

    print(f"\n2. 开始处理多链交界面，合并 PDB 并复制至 {OUT_DIR}/ 目录...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    success_folders = 0
    interface_complexes = 0
    
    for combo, ligs in tqdm(combo_to_ligs.items(), desc="组装复合物"):
        # combo 是一个集合，如 {'10gsA', '10gsB'}
        # 提取公共的 PDB ID (前4位) 和所有的链 ID
        pdb_id = list(combo)[0][:4]
        chains = sorted([rec[4:] for rec in combo]) # 提取并排序链名
        
        # 组合出新的复合物名称，例如 "10gsAB"
        combo_name = f"{pdb_id}{''.join(chains)}"
        
        target_dir = os.path.join(OUT_DIR, combo_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # --- 处理受体 (合并逻辑) ---
        combined_rec_dst = os.path.join(target_dir, f"{combo_name}_receptor.pdb")
        rec_sources = [os.path.join(RAW_RECEPTOR_DIR, f"{rec}.pdb") for rec in sorted(combo)]
        
        # 如果只结合在单链上，直接复制；如果结合在多链上，进行 PDB 文本拼接合并
        if len(combo) == 1:
            shutil.copy2(rec_sources[0], combined_rec_dst)
        else:
            combine_pdb_files(rec_sources, combined_rec_dst)
            interface_complexes += 1
            
        # --- 处理配体 ---
        ligs.sort()
        for idx, lig_src_name in enumerate(ligs):
            lig_src_path = os.path.join(RAW_LIGAND_DIR, lig_src_name)
            lig_dst_path = os.path.join(target_dir, f"{combo_name}{idx}_ligand.pdb")
            shutil.copy2(lig_src_path, lig_dst_path)
            
        success_folders += 1

    print("\n" + "="*50)
    print(f"�� 高级组装完成！")
    print(f"✅ 共构建了 {success_folders} 个复合物文件夹。")
    print(f"�� 其中包含了 {interface_complexes} 个自动合并的多链交界面(Interface)受体！")
    print(f"�� 优化后的训练数据已保存在: {os.path.abspath(OUT_DIR)}")
    print("="*50)

if __name__ == "__main__":
    organize_complexes_with_interfaces()