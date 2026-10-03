import os
import glob
import json
import torch
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from scipy.spatial import cKDTree
import multiprocessing
from tqdm import tqdm
import warnings

# --- 屏蔽警告 ---
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 路径与配置区
# =============================================================================
PROJECT_ROOT = os.getcwd()

# [修改点 1] 增加 processed_data 子目录变量
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed_data')

# --- 词汇表路径 ---
# 假设词汇表也放在根目录或 processed_data 里，请核对。这里假设在根目录。
# 如果词汇表也在 processed_data 里，请将 PROJECT_ROOT 改为 PROCESSED_DIR
VOCAB_768_FILE = os.path.join(PROJECT_ROOT, 'fragment_vocabulary_768.json')
VOCAB_64_FILE = os.path.join(PROJECT_ROOT, 'fragment_vocabulary_64_dynamic_margin.json')

# --- 待处理的输入数据集目录 ---
# [修改点 2] 将 PROJECT_ROOT 改为 PROCESSED_DIR
INPUT_DIRS = [
    os.path.join(PROCESSED_DIR, 'SE3TD_Pre_datas_768'),
    os.path.join(PROCESSED_DIR, 'SE3TD_fine_datas_768')
]

# --- 转换后保存的输出数据集目录 ---
# [修改点 3] 同样输出到 processed_data 目录下保持整洁
OUTPUT_DIRS = [
    os.path.join(PROCESSED_DIR, 'SE3TD_Pre_datas_64_dynamic'),
    os.path.join(PROCESSED_DIR, 'SE3TD_fine_datas_64_dynamic')
]

# 兜底机制：对于无法解析的特例，赋予一个较大的 HAC 值
FALLBACK_HAC = 100.0 

# 获取 CPU 核心数用于高并发
NUM_WORKERS = min(multiprocessing.cpu_count(), 32)

# =============================================================================
# 2. 全局词汇表加载与映射体系构建
# =============================================================================
print(">>> [1/3] 正在加载双维度词汇表并计算重原子数 (HAC)...")

# 加载旧 768 维词汇表 (用于寻址)
if not os.path.exists(VOCAB_768_FILE):
    raise FileNotFoundError(f"找不到高维词汇表: {VOCAB_768_FILE}")
with open(VOCAB_768_FILE, 'r', encoding='utf-8') as f:
    vocab_768_data = json.load(f)

# 加载新 64 维词汇表 (用于替换)
if not os.path.exists(VOCAB_64_FILE):
    raise FileNotFoundError(f"找不到降维词汇表: {VOCAB_64_FILE}")
with open(VOCAB_64_FILE, 'r', encoding='utf-8') as f:
    vocab_64_data = json.load(f)

# 确保我们不在处理 <UNK> 的污染数据
if "<UNK>" in vocab_768_data: vocab_768_data.pop("<UNK>")
if "<UNK>" in vocab_64_data: vocab_64_data.pop("<UNK>")

smiles_list = []
vectors_768_list = []
vectors_64_list = []
hac_list = []

# 对齐两个词汇表
for smi in tqdm(vocab_768_data.keys(), desc="构建空间映射表"):
    if smi not in vocab_64_data:
        print(f"警告：SMILES {smi} 在 64 维词汇表中丢失，已跳过。")
        continue
        
    smiles_list.append(smi)
    vectors_768_list.append(vocab_768_data[smi])
    vectors_64_list.append(vocab_64_data[smi])
    
    # 计算 HAC
    mol = Chem.MolFromSmiles(smi)
    if mol:
        # 强制兜底至少为 1.0
        hac_list.append(max(1.0, float(mol.GetNumHeavyAtoms())))
    else:
        hac_list.append(FALLBACK_HAC)

# 转为高效的 numpy 数组
vectors_768_np = np.array(vectors_768_list, dtype=np.float32)
vectors_64_np = np.array(vectors_64_list, dtype=np.float32)
hac_np = np.array(hac_list, dtype=np.float32)

print(">>> [2/3] 正在构建高维 KDTree 空间索引 (用于精准反向检索)...")
# 我们在旧的高维空间上建树，因为 .pt 文件里的数据是高维的
vector_tree = cKDTree(vectors_768_np)
print(f"映射体系构建完成！共成功对齐 {len(smiles_list)} 个片段。")

# =============================================================================
# 3. 多进程 Worker 任务函数
# =============================================================================
def process_single_file(args):
    """
    args = (input_path, output_path)
    处理单个 .pt 文件：
    1. 读取旧特征
    2. 通过 KDTree 在 768 维空间锁定身份索引
    3. 获取对应的 64 维特征和 HAC
    4. 替换 data.frag_embeds, 新增 data.hac
    5. 保存到全新目录
    """
    input_path, output_path = args
    try:
        # 如果目标文件已存在，说明之前跑过，直接跳过 (断点续传)
        if os.path.exists(output_path):
            return True
            
        data = torch.load(input_path, weights_only=False)
        
        if not hasattr(data, 'frag_embeds') or data.frag_embeds is None:
            return f"Error: 文件 {os.path.basename(input_path)} 缺少 frag_embeds 属性。"
            
        # 提取高维特征并转为 numpy
        embeds_old_np = data.frag_embeds.cpu().numpy().astype(np.float32)
        
        # 使用 KDTree 在 768/728 维空间查询最近的词汇表身份
        distances, indices = vector_tree.query(embeds_old_np, k=1)
        
        # 使用查到的身份索引，直接拉取新的 64 维特征和 HAC
        new_embeds_64 = vectors_64_np[indices]
        new_hacs = hac_np[indices]
        
        # 将新特征覆盖原属性，并写入 hac
        data.frag_embeds = torch.tensor(new_embeds_64, dtype=torch.float32)
        data.hac = torch.tensor(new_hacs, dtype=torch.float32)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 安全保存：先存为临时文件再重命名，防止写入中断导致损坏
        temp_path = output_path + ".tmp"
        torch.save(data, temp_path)
        os.replace(temp_path, output_path)
        
        return True
    except Exception as e:
        return f"Error on {os.path.basename(input_path)}: {str(e)}"

# =============================================================================
# 4. 主执行逻辑
# =============================================================================
if __name__ == '__main__':
    print(f"\n>>> [3/3] 开始扫描并生成新的降维数据集...")
    
    tasks = []
    for in_dir, out_dir in zip(INPUT_DIRS, OUTPUT_DIRS):
        if not os.path.exists(in_dir):
            print(f"⚠️ 警告: 输入目录不存在，已跳过 -> {in_dir}")
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        
        # 扫描所有 .pt 文件
        search_pattern = os.path.join(in_dir, '**', '*.pt')
        pt_files = glob.glob(search_pattern, recursive=True)
        
        for f_path in pt_files:
            # 保持子目录结构不变
            rel_path = os.path.relpath(f_path, in_dir)
            out_path = os.path.join(out_dir, rel_path)
            tasks.append((f_path, out_path))
            
    if not tasks:
        print("未找到任何需要处理的 .pt 文件。")
        exit()
        
    print(f"总计找到 {len(tasks)} 个复合物文件。准备分发至 {NUM_WORKERS} 个核心并行处理...")
    
    success_count = 0
    error_list = []
    
    # 启用多进程池极速替换
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(tasks), desc="维度替换 & HAC 注入") as pbar:
            for result in pool.imap_unordered(process_single_file, tasks):
                if result is True:
                    success_count += 1
                else:
                    error_list.append(result)
                pbar.update(1)
                
    print(f"\n{'='*60}")
    print(f"🎉 全部转化完成！")
    print(f"成功创建新数据集文件: {success_count} 个。")
    print(f"新数据集根目录 (请在 config 中更新它们):")
    for out_dir in OUTPUT_DIRS:
        print(f"  -> {out_dir}")
        
    if error_list:
        print(f"\n⚠️ 发生错误的文件数: {len(error_list)}")
        print("前 5 个错误示例:")
        for err in error_list[:5]:
            print("  ", err)
    print(f"{'='*60}")