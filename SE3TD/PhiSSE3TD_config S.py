import os
import torch
import torch.nn.functional as F

# =============================================================================
# 1. 系统与环境设置 (System & Environment)
# =============================================================================
# 获取当前脚本所在的绝对路径 (即 GNN/SE3TD/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 定义 GNN 根目录 (即 GNN/)
GNN_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))

# 随机种子
SEED = 42

# 计算设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#  分布式后端配置 (通常使用 nccl)
DIST_BACKEND = 'nccl'

# =============================================================================
# 2. 数据与维度定义 (Data & Dimensions)
# =============================================================================
# 原始数据中的维度定义
EMBEDDING_DIM_IN = 64      # 片段嵌入向量维度 (来自预训练的词汇表向量)
CHEM_PROPS_DIM_IN = 14      # 化学属性维度 (如原子质量、价键数等物理属性)
TYPE_ENCODING_DIM_IN = 3    # 节点类型编码维度 (One-hot: [1,0,0]=配体, [0,1,0]=蛋白, [0,0,1]=虚拟锚点)

# 核心特征维度 (用于扩散和生成)
# 作用: 扩散模型预测的主要特征部分，包含嵌入和化学属性，不含类型编码
LIGAND_FEATURE_DIM = EMBEDDING_DIM_IN + CHEM_PROPS_DIM_IN 

# =============================================================================
# 3. E(3)NN 模型超参数 (Model Hyperparameters)
# =============================================================================
# 边特征相关
EDGE_ATTR_DIM = 6           # 边属性的 One-hot 编码维度 (区分 LL-Local, LL-Global, LP, PP 等边类型，新增Dummy-Ligand 虚拟拉扯边)
NUM_BASIS = 32              # 径向基函数 (RBF) 的数量，用于将连续的距离值离散化编码
MAX_EDGE_LENGTH = 12        # Transformer 编码边特征时的最大截断距离 (Angstrom)，超过此距离的相互作用在 Transformer 层被忽略

# 网络架构相关
NUM_TRANSFORMER_LAYERS = 4           # 等变 Transformer 的层数 (深度)，层数越深感受野越大
FC_NEURONS = [96, 64]               # 全连接层 (MLP) 的隐藏层神经元数量，用于处理边权和标量特征
EDGE_PREDICTOR_NEURONS = [192, 96]  # 第一层用 256 承接 544 维的高密度输入，第二层用 128 提炼特征，最后输出 6 分类
TIME_EMB_DIM = 64             # 扩散时间步 (t) 的正弦波嵌入维度
ACTIVATION_FUNCTION = F.silu  # 激活函数 (SiLU / Swish)，在深度学习中表现优于 ReLU

# Irreps (不可约表示) 配置 - 核心等变性参数
# 定义了网络隐藏层中包含多少个标量通道(0e)和向量通道(1o)
TOTAL_HIDDEN_VEC_CHANNELS = 160  # 隐藏层总共分配的向量通道数 (Vector Channels)
REF_COORDS_HIDDEN_CHANNELS = 64  # 其中专门用于编码参考坐标系 (Ref Coords) 的向量通道数
# 剩余 (128 - 32 = 96) 个通道作为"自由向量"，用于学习不依赖特定几何约束的抽象方向特征

HIDDEN_SCALAR_CHANNELS = 192     # 隐藏层标量通道数 
SH_LMAX = 2                      # 球谐函数最大阶数

# 正则化
SCALAR_DROPOUT = 0.1        # 标量特征的 Dropout 率，防止过拟合

# =============================================================================
# 4. 扩散过程与动态图参数 (Diffusion & Graph)
# =============================================================================
# 扩散参数
NUM_TIMESTEPS = 1000        # 总扩散步数 T。训练时将数据加噪 T 步，生成时去噪 T 步
BETA_SCHEDULE = 'cosine'    # 噪声调度策略。'cosine' 比 'linear' 能在中间步数保留更多信息，生成质量通常更高

# 裁剪/截断参数 (Clamping)
# 作用: 在采样过程中防止预测值数值爆炸 (Exploding)，保证数值稳定性
FEAT_CLAMP_BUFFER = 20.0         # 特征值裁剪：限制预测的特征值范围
REF_COORDS_CLAMP_BUFFER = 20.0   # 相对参考系裁剪：限制局部坐标系的轴长不超过此值
POS_ABS_SAFETY_CLAMP = 50.0     # 坐标绝对安全截断：仅在没有动态半径边界时防止 x0_pred_pos 数值爆炸
EMBEDDING_SAFETY_CLAMP = 20.0   # Embedding 绝对安全截断：仅防止 x0_pred_feat embedding 维度数值爆炸
POS_CLAMP_BUFFER = 20.0          # 坐标裁剪：限制生成的配体原子不能跑出 (蛋白最大半径 + Buffer) 的范围

# 动态构图参数 (Dynamic Graph Construction)
# 作用: 扩散过程中原子位置在变，需要在每一步重新构建图连接
DYNAMIC_GRAPH_MAX_RADIUS = 12.0 # 构建半径图的最大搜索半径 (Angstrom)。注意：此处比Transformer编码半径小，是为了节省显存

# =============================================================================
# 5. 训练超参数 (Training Hyperparameters)
# =============================================================================
GRAD_ACCUMULATION_STEPS = 3   # 假设实际 BatchSize=8，累积4步，等效 BatchSize=32
BATCH_SIZE = 14                # 批处理大小 (受限于显存，E3NN 计算量较大，通常设较小值)
PIN_MEMORY = True             # 开启锁页内存，加速 CPU 到 GPU 的传输
SMOOTH_L1_BETA = 1.0          # SmoothL1Loss 的 Beta 参数，控制 L1 和 L2 损失的过渡点

# Dataset
TRAIN_VAL_SPLIT = 0.85           # 训练集占比

# --- 数据集加载策略 (Dataset Loading Strategy) ---
# True:  懒加载 (Lazy Loading) - 节省内存，每次从硬盘实时读取，但受限于磁盘 I/O 速度。
# False: 预加载 (Preloaded)    - 消耗大量内存，一次性读入物理内存，训练极快，消除 I/O 瓶颈。
USE_LAZY_DATASET = False
# 智能调整 Worker 数量
NUM_WORKERS = 32  # 满血 32 线程并发，彻底消除 I/O 和预处理瓶颈

# 阶段一：预训练 (Pre-training) - 几何重建
# ---  Warmup 与梯度裁剪参数 ---
PRETRAIN_EPOCHS = 200            # 预训练轮数
PRETRAIN_WARMUP_EPOCHS = 6      # 阶段 1: 预热期 (LR 从小爬升到 100%)
PRETRAIN_HOLD_EPOCHS = 2        # 阶段 2: 稳定期 (LR 保持 100% 满血输出)
GRAD_CLIP_WARMUP_NORM = 10      # Warmup 期的最大梯度范数（容忍原子散开的剧烈变化）
GRAD_CLIP_STABLE_NORM = 10     # 稳定期的最大梯度范数（保护脆弱的化学几何结构）
WARMUP_START_LR_RATIO = 0.4      # Warmup 起始时的学习率比例 (即从 20% 的 LR 开始爬升)

PRETRAIN_LR = 8e-5              # 初始学习率       
PRETRAIN_WEIGHT_DECAY = 4e-6    # 权重衰减 (L2 正则化)

# 阶段二：微调 (Fine-tuning) - 活性引导
FINETUNE_EPOCHS = 100          # 微调轮数
FINETUNE_WARMUP_EPOCHS = 4     # warmup 轮数
FINETUNE_HOLD_EPOCHS = 0      # 稳定期 轮数
FINETUNE_GRAD_CLIP_WARMUP_NORM = 10.0  # 微调初期的宽容截断 
FINETUNE_GRAD_CLIP_STABLE_NORM = 10.0   # 微调稳定期的严格截断
FINETUNE_WARMUP_START_LR_RATIO = 0.6    # 微调 Warmup 起始时的学习率比例

ACTIVITY_GUIDANCE_RATIO = 0.2  # 在扩散过程的最后 40% 步数中启用活性分类器指导
FINETUNE_LR = 1e-5             # 微调通常使用更小的学习率
FINETUNE_WEIGHT_DECAY = 1e-6   # 权重衰减 

# 损失权重 (Phase 1 & 2 通用)
# 总 Loss = w1*Feat + w2*Pos + w3*Frame
W_FEAT_LOSS = 1.0     # 特征噪声预测的权重
W_POS_LOSS = 8.0      # 原子坐标噪声预测的权重
W_FRAME_LOSS = 6.0    # 局部参考系噪声预测的权重
W_EDGE_LOSS = 6.0     # 图拓扑边预测的 Focal Loss 权重
# --- 稀疏相对距离损失参数 (Sparse Distance Loss) ---
W_DIST_LL_LOSS = 0.5  # 配体内部刚性约束权重 (保持 1.0，提供强力的内部骨架支撑)
W_DIST_LP_LOSS = 0.5  # 配体-蛋白互作约束权重 (设为 1.0，引导准确落位但不剥夺生成多样性)
DIST_LL_MAX_RADIUS = 12.0  # LL 距离损失的雷达扫描半径 (12埃足以囊括整个配体)
DIST_LP_MAX_RADIUS = 6.0  # LP 距离损失的截断半径 (仅关注核心药效团与范德华接触区的真实物理互作)

# ---  特征损失 HAC Focal Loss 参数 ---
# 作用: 基于重原子数 (HAC) 施加不平等的特征重构惩罚，强迫模型精准预测单原子/双原子
USE_HAC_FOCAL_LOSS = True       # 是否开启 HAC 焦点惩罚机制
HAC_PENALTY_MAX = 12.0          # HAC=1 时(单原子)的最大惩罚倍率 (建议 3.0 ~ 5.0)
HAC_DECAY_BETA = 1.0            # 衰减速率指数 (反比例衰减)。1.0 为平滑衰减，值越大衰减越陡峭

# --- 拓扑边预测 Focal Loss 参数 ---
# 类别顺序: [LLCov, PPCov, LPGlob, LLGlob, PPGlob, Null]
# 抑制占比95%的Null(假边)，放大LLCov和LPGlob等稀有真实边的学习权重
EDGE_CLASS_WEIGHTS = [8.0, 0.0, 6.0, 6.0, 0.0, 2.0] 
FOCAL_LOSS_GAMMA = 3.0 # FocalLoss难易样本聚焦参数值越大越关注预测困难的边

# --- 物理约束损失超参数 (Physical Constraint Loss) ---
PHYSICS_GUIDANCE_RATIO = 0.6   
W_PHYSICS_CLASH = 2.0          # 二次方碰撞的主权重保持 2.0 左右
# HAC 自适应碰撞参数
CLASH_DIST_MIN = 0.8            # 单原子(HAC=1)的最小安全距离 (埃)
CLASH_DIST_MAX = 2.4            # 大片段(HAC>=12)的最小安全距离 (埃)
CLASH_WEIGHT_MIN = 1.0          # 单原子(HAC=1)的惩罚系数衰减 
CLASH_WEIGHT_MAX = 1.2          # 大片段(HAC>=12)的惩罚系数
CLASH_HAC_CAP = 12.0            # 达到该 HAC 值后(如苯环)，阈值和权重不再增加

# 微调损失分量权重
# 总 Loss = w_diff*DiffLoss + w_act*ActivityLoss + w_var*VarianceLoss
W_DIFFUSION = 0.6     # 保持几何合理性的基础扩散损失权重
W_ACTIVITY = 0.2      # 外部模型 (PhiSGATv2) 给出的活性评分损失权重 (希望活性越高越好)
W_VARIANCE = 0.2      # 多样性方差损失权重 (防止生成的分子坍缩成同一种模式)

# =============================================================================
# 6. 生成与采样参数 (Generation & Sampling)
# =============================================================================
# 任务控制
NUM_POCKETS_TO_PROCESS = -1  # 处理多少个口袋任务，-1 表示处理文件夹下所有 .pt 文件
LIGANDS_PER_POCKET = 10      # 每个口袋生成多少个候选配体

# 扩散采样控制
DDIM_STEPS = 600          # 推理时的去噪步数 (DDIM加速采样)，步数越多质量越高但越慢
DDIM_ETA = 0.0               # 控制采样随机性，0.0 为确定性，1.0 为 DDPM
MAX_GENERATION_ROUNDS = 1000    # 如果解码/聚类失败，最大重试次数
# 智能节点数目预测控制
USE_SIZE_PREDICTOR = False    # True: 使用训练好的容量预测器动态决定; False: 使用固定值 INPUT_NOISE_NODES
INPUT_NOISE_NODES = 3         # 如果关闭预测器(或预测器加载失败)时的默认兜底值
MIN_LCC_RATIO = 0.6          # 最大连通子图 (LCC) 包含的节点数占解码成功节点总数的最小比例。
GENERATION_LIGAND_CENTER_MAX_RADIUS = 4.6  # Inference reject threshold for ||mean(ligand pos)||; <=0 disables.
LLCOV_MAX_BOND_DIST = 2.8    # 拓扑边成键物理阈值 (LLCov Bond Threshold)
EDGE_CONFIDENCE_THRESHOLD = 0.28  # 判定 LLCov 真实建键的最低概率要求 (0.0~1.0)，防过拟合“乱拉红线”
POS_SOFT_CLAMP_RATIO = 1.0      # DDIM 采样中 xt_prev_pos 的软截断比例 (相对于 pos_clamp_max)超出此比例后 tanh 压缩
# Langevin 能量引导参数 (Langevin Dynamics) ---
ENABLE_LANGEVIN_GUIDANCE = True  # 是否开启推理期排斥引导 (强力防止聚集坍缩)
LANGEVIN_SCALE = 2e-5             # 基础推力系数 (推荐 0.1 ~ 0.5。太大把分子炸飞，太小推不开)
# Generation  
DECODING_TOP_K = 2              # 词汇表匹配候选数量
LINEARITY_THRESHOLD = 0.2       # 判定分子为线性的 PCA 阈值
# --- 刚体滑移 (Translation Gradient Descent, TGD) 参数 ---
TGD_MAX_STEPS = 40        # 最大迭代步数
TGD_LEARNING_RATE = 1e-7  # 滑移步长系数
TGD_MAX_STEP_DIST = 2e-2  # 单步最大允许位移 (埃)，防止被弹飞
TGD_FORCE_TOL = 0.2       # 收敛阈值，受力小于此值则提前停止
ENABLE_POCKET_AWARE_POST_FF = True
POCKET_AWARE_LOCAL_CUTOFF = 8.0 # Pocket-aware 优化时纳入配体周围该距离(A)内的局部蛋白原子
# --- MMFF94 串行优化步数控制 ---
POCKET_AWARE_MMFF_STEPS = 0            # 阶段一：固定局部蛋白，进行 pocket-aware MMFF94 联合组装的步数 (防漂移，强行按入口袋)
POCKET_AWARE_LIGAND_ONLY_MMFF_STEPS = 60 # 阶段二：仅对配体单独进行 MMFF94 松弛的步数 (防畸变，释放第一步造成的局部张力)

# 生存模式：动态特征索引
# 作用: 在 Survival 模式中，指定哪些特征维度允许随模型更新 (不被 x0 强制覆盖)
DYNAMIC_CHEM_INDICES = [EMBEDDING_DIM_IN]

# =============================================================================
# 7. 路径配置 (Path Configurations)
# =============================================================================

# --- A. 输入数据路径 (位于 GNN/CHEMCUT/processed_data) ---
# CHEMCUT 项目根目录
CHEMCUT_ROOT = os.path.join(GNN_ROOT, 'CHEMCUT')
# CHEMCUT 数据处理目录
CHEMCUT_DATA_DIR = os.path.join(CHEMCUT_ROOT, 'processed_data')

# 训练用的全量数据 (包含配体和蛋白图)
PYG_DATA_DIR = os.path.join(CHEMCUT_DATA_DIR, 'SE3TD_fine_datas_64_dynamic')
# 测试用的独立数据
PYG_TEST_DATA_DIR = os.path.join(CHEMCUT_DATA_DIR, 'SE3TD_test_datas')

# [生成任务] 各类输入数据路径
# Survival 模式: 从 TASK_A 目录读取 complex.pt + pocket.pdb
# Creative 模式: 从 TASK_B 目录读取 {id}.pt + pocket.pdb
TASK_A_DATA_DIR = os.path.join(CHEMCUT_DATA_DIR, 'TASK_A')
TASK_B_DATA_DIR = os.path.join(CHEMCUT_DATA_DIR, 'TASK_B')

# 定义各个词汇表的路径
VOCAB_BASE_DIR = os.path.join(GNN_ROOT, 'CHEMCUT', 'processed_data')
VOCAB_FILES_MAP = {
    'puppy':  os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_puppy.json'),
    'linker': os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_linker.json'),
    'frame':  os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_frame.json'),
    'filter': os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_filter.json'),
    'valid': os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_valid.json'),
    'ol': os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_64_pca.json')
}

# [核心超参数] 定义不同模式下需要加载的词汇表组合
# 你可以在这里自由添加或删除列表中的 key ('puppy', 'linker', 'frame', 'filter'，'valid'）
MODE_VOCAB_SETTINGS = {
    'SURVIVAL': ['puppy', 'frame', 'linker'],
    'CREATIVE': ['valid']
}

# --- B. 输出/保存路径 (位于 GNN/SE3TD-DDP/processed_data) ---
# 本项目 (SE3TD-DDP) 的数据输出根目录
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'processed_data')

# -----------------------------------------------------------------------------
# 阶段一：预训练模型与日志保存路径 (Pre-training Configurations)
# (对应脚本: PhiSSE3TD_train_pretrain.py)
# -----------------------------------------------------------------------------
CHECKPOINT_DIR_PRETRAIN = os.path.join(SAVE_ROOT, 'checkpoints_pretrained_64')
LOG_DIR_PRETRAIN = os.path.join(SAVE_ROOT, 'logs_pretrained_64')
BEST_MODEL_PATH_PRETRAIN = os.path.join(CHECKPOINT_DIR_PRETRAIN, 'best_pretrained_model_64.pt')

# -----------------------------------------------------------------------------
# 阶段二：活性联合微调路径 (Fine-tuning Configurations)
# (对应脚本: PhiSSE3TD_train_Finetune_Activity.py)
# -----------------------------------------------------------------------------
CHECKPOINT_DIR_FINETUNE = os.path.join(SAVE_ROOT, 'checkpoints_finetuned_64')
LOG_DIR_FINETUNE_SA  = os.path.join(SAVE_ROOT, 'logs_finetuned_64_SA')
BEST_MODEL_PATH_FINETUNE_SA  = os.path.join(CHECKPOINT_DIR_FINETUNE, 'best_finetuned_SA_model_64.pt')

# =============================================================================
# 🚀 核心路由字典 (Model Router)
# =============================================================================
# 推理脚本 (generate.py) 将根据此字典动态加载对应权重
MODEL_WEIGHT_ROUTES = {
    'PRETRAIN':    BEST_MODEL_PATH_PRETRAIN,     # 基础结构预训练版 (双模式混合)
    'FINETUNE_SA': BEST_MODEL_PATH_FINETUNE_SA   # 活性与结构联合微调版 
}

# 默认使用的模型路由
DEFAULT_MODEL_CHOICE = 'PRETRAIN'

# 全局模式选择
# 'CREATIVE': 从零开始生成，读取 TASK_B 目录
# 'SURVIVAL': 固定锚点的位置和构象，仅生长，读取 TASK_A 目录
DEFAULT_GENERATION_MODE = 'SURVIVAL'

# 节点数量预测权重文件
SIZE_PREDICTOR_MODEL_PATH = os.path.join(SAVE_ROOT, 'size_predictor_64', 'pocket_size_predictor_64.pt')

# 生成结果 (SDF/PDB) 输出路径
OUTPUT_DIR_GENERATION = os.path.join(SAVE_ROOT, 'generated_molecules')
OUTPUT_DIR_GENERATION2 = os.path.join(SAVE_ROOT, 'generated_molecules')

# --- C. 外部指导模型路径 (PhiSGATv2) ---
# 位于 GNN/GATv2
PHISGAT_ROOT = os.path.join(GNN_ROOT, 'GATv2')
# 如果模型权重在 GNN/GATv2/processed_data 下，微调脚本中需引用此路径
PHISGAT_DATA_DIR = os.path.join(PHISGAT_ROOT, 'processed_data')