# PhiSGATv2_config.py

import torch
import os

# -----------------------------------------------------------------------------
# 1. 动态路径计算 (Path Calculation) - 核心修改部分
# -----------------------------------------------------------------------------

# 获取当前脚本 (PhiSGATv2_config.py) 所在的绝对目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 计算 CHEMCUT 目录的路径
# 逻辑：从 BASE_DIR (GATv2) 向上一级 (..) 走，然后进入 CHEMCUT
# 结果指向: .../project/GNN/CHEMCUT
CHEMCUT_ROOT = os.path.normpath(
    os.environ.get('CHEMCUT_ROOT', os.path.join(BASE_DIR, '..', 'CHEMCUT'))
)


# -----------------------------------------------------------------------------
# 2. 数据相关配置 (Data Settings)
# -----------------------------------------------------------------------------

# 训练集和验证集的预处理数据路径
# 拼接后: .../GNN/CHEMCUT/processed_data/GATv2_datas
TRAIN_VAL_PROCESSED_DIR = os.path.join(CHEMCUT_ROOT, 'processed_data', 'GATv2_datas')

# 独立测试集的预处理数据路径
# 拼接后: .../GNN/CHEMCUT/processed_data/GATv2_test_datas
TEST_PROCESSED_DIR = os.path.join(CHEMCUT_ROOT, 'processed_data', 'GATv2_test_datas')

# 未知活性预测数据路径 (用于 predict.py)
# 拼接后: .../GNN/CHEMCUT/processed_data/unknown_active_datas
UNKNOWN_DATA_DIR = os.path.join(CHEMCUT_ROOT, 'processed_data', 'unknown_active_datas')

# 数据集划分比例
TRAIN_RATIO = 0.9

# DataLoader 配置
# 排除不需要转换为Tensor的键
EXCLUDE_KEYS_IN_LOADER = ['mol_id', 'smiles']


# -----------------------------------------------------------------------------
# 3. 通用设置 (General Settings)
# -----------------------------------------------------------------------------

RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------------------------------------------------------
# 4. 模型超参数 (Model Hyperparameters)
# -----------------------------------------------------------------------------

# 节点原始数值特征的维度
INPUT_DIM_NUMERIC = 11

#  边特征的维度 
EDGE_DIM = 3 

# 分子片段嵌入向量的维度
EMBEDDING_DIM = 24

# GATv2 隐藏层的通道数
HIDDEN_CHANNELS = 128

# GATv2 层的数量
NUM_LAYERS = 4

# GATv2 中的多头注意力头数
HEADS = 8

# Dropout比率
DROPOUT = 0.1

# 多任务学习中的任务数量
NUM_TASKS = 4


# -----------------------------------------------------------------------------
# 5. 训练超参数 (Training Hyperparameters)
# -----------------------------------------------------------------------------

LEARNING_RATE = 1e-4
BATCH_SIZE = 24
EPOCHS = 100
WEIGHT_DECAY = 1e-5

# 最大梯度 L2 范数；超过该值时执行梯度裁剪，降低数值爆炸风险。
GRAD_CLIP_MAX_NORM = 5.0

# -----------------------------------------------------------------------------
# 6. 输出与保存配置 (Output & Saving Settings)
# -----------------------------------------------------------------------------

# 模型保存目录
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'processed_data')

# 最佳模型的文件名
MODEL_FILENAME = "best_model.pth"

# 完整的模型保存路径
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
