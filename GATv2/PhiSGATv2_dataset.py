# PhiSGATv2_dataset.py 

import os
import glob
import torch
from torch_geometric.data import Dataset
# [必须导入 config]
import PhiSGATv2_config as config

class MoleculeDataset(Dataset):
    """
    加载预处理好的分子图数据。
    """
    def __init__(self, root=None, data_type='train_val', data_dir=None, transform=None, pre_transform=None):
        """
        Args:
            root (str): 在新逻辑中，这个参数仅用于占位，实际路径由 config 决定。
        """
        self.data_type = data_type
        self.data_dir = data_dir
        self.data_files = []
        super(MoleculeDataset, self).__init__(root or config.CHEMCUT_ROOT, transform, pre_transform)
        
        # 查找所有预处理好的 .pt 文件
        self.data_files = sorted(glob.glob(os.path.join(self.processed_dir, '*.pt')))
        
        if not self.data_files:
            print(f"警告: 在目录 '{self.processed_dir}' 中没有找到任何 .pt 文件。")
        else:
            print(f"在 '{self.processed_dir}' 中找到了 {len(self.data_files)} 个数据文件。")

    @property
    def processed_dir(self):
        """返回预处理数据的具体路径。"""
        # [核心修改] 直接读取 config 中的绝对路径，忽略 self.root
        if self.data_dir is not None:
            return self.data_dir
        if self.data_type == 'train_val':
            return config.TRAIN_VAL_PROCESSED_DIR
        if self.data_type == 'test':
            return config.TEST_PROCESSED_DIR
        if self.data_type in ('unknown', 'predict'):
            return config.UNKNOWN_DATA_DIR
        raise ValueError(f"Unknown data_type: {self.data_type!r}")

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            return []
        return [
            os.path.basename(f)
            for f in sorted(glob.glob(os.path.join(self.processed_dir, '*.pt')))
        ]

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        data = torch.load(self.data_files[idx], weights_only=False)
        return data
