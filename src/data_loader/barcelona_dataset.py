"""
巴塞罗那基站能耗数据集
功能：
1. 加载预处理后的节点数据
2. 构建LSTM序列（28个时段输入，4个时段输出）
3. One-Hot编码部门特征
4. PyTorch Dataset类
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class BarcelonaDataset(Dataset):
    """
    巴塞罗那基站能耗数据集
    输入: 过去28个时段的能耗值 + 部门特征
    输出: 未来4个时段的能耗值
    """
    
    def __init__(self, data_path, window_size=28, predict_size=4, sector_feature=True):
        """
        Args:
            data_path: 节点数据路径（如 node_8001/train.pkl）
            window_size: 输入窗口大小（默认28个时段=7天）
            predict_size: 输出窗口大小（默认4个时段=1天）
            sector_feature: 是否使用部门特征
        """
        # 加载数据
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.sector_feature = sector_feature
        
        # 获取能耗序列
        self.energy = self.df['Valor_norm'].values
        self.energy_raw = self.df['Valor'].values  # 用于反归一化验证
        
        # 获取部门特征（One-Hot编码）
        if sector_feature:
            sector_codes = self.df['sector_code'].values
            self.sector_onehot = self._one_hot_sector(sector_codes)
        
        # 构建有效样本索引
        self.indices = self._build_indices()
        
    def _one_hot_sector(self, sector_codes):
        """将部门代码转换为One-Hot编码（4类）"""
        n_sectors = self.df['sector_code'].nunique()
        onehot = np.zeros((len(sector_codes), n_sectors))
        for i, code in enumerate(sector_codes):
            if 0 <= code < n_sectors:
                onehot[i, code] = 1
        return onehot
    
    def _build_indices(self):
        """
        构建有效样本索引
        需要满足: 有连续 window_size + predict_size 个数据点
        """
        indices = []
        total_len = len(self.energy)
        
        for i in range(total_len - self.window_size - self.predict_size + 1):
            indices.append(i)
        
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        # 输入序列: 过去 window_size 个时段的能耗
        x_energy = self.energy[start_idx:start_idx + self.window_size]
        x_energy = torch.FloatTensor(x_energy).unsqueeze(-1)  # [window_size, 1]
        
        # 部门特征: 取输入序列最后一个时段的部门
        if self.sector_feature:
            sector_idx = start_idx + self.window_size - 1
            x_sector = self.sector_onehot[sector_idx]
            x_sector = torch.FloatTensor(x_sector)
            x = torch.cat([x_energy, x_sector.unsqueeze(0).repeat(self.window_size, 1)], dim=1)
        else:
            x = x_energy
        
        # 输出序列: 未来 predict_size 个时段的能耗
        y = self.energy[start_idx + self.window_size:start_idx + self.window_size + self.predict_size]
        y = torch.FloatTensor(y)
        
        return x, y


def get_node_data_loader(node_id, split='train', batch_size=64, shuffle=True,
                         window_size=28, predict_size=4, sector_feature=True):
    """
    获取指定节点的DataLoader
    
    Args:
        node_id: 节点编号（如 8001）
        split: 'train', 'val', 'test'
        batch_size: 批次大小
        shuffle: 是否打乱
        window_size: 输入窗口
        predict_size: 输出窗口
        sector_feature: 是否使用部门特征
    
    Returns:
        DataLoader, scaler_path, dataset
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "processed", "barcelona_ready", f"node_{node_id}")
    
    file_path = os.path.join(data_dir, f"{split}.pkl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    dataset = BarcelonaDataset(
        data_path=file_path,
        window_size=window_size,
        predict_size=predict_size,
        sector_feature=sector_feature
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    
    # 返回scaler路径用于反归一化
    scaler_path = os.path.join(data_dir, "scaler.pkl")
    
    return dataloader, scaler_path, dataset


def test_dataset():
    """测试数据集"""
    print("=" * 60)
    print("测试 BarcelonaDataset")
    print("=" * 60)
    
    # 获取节点8001的训练数据
    dataloader, scaler_path, dataset = get_node_data_loader(
        node_id=8001,
        split='train',
        batch_size=64,
        shuffle=False,
        window_size=28,
        predict_size=4,
        sector_feature=False
    )
    
    print(f"\n节点8001训练集:")
    print(f"  样本数: {len(dataset)}")
    print(f"  批次: {len(dataloader)}")
    
    # 查看一个batch
    for x, y in dataloader:
        print(f"\n输入形状: {x.shape}")  # [batch, window_size, features]
        print(f"输出形状: {y.shape}")    # [batch, predict_size]
        print(f"输入范围: {x[:, :, 0].min():.4f} ~ {x[:, :, 0].max():.4f}")
        print(f"输出范围: {y.min():.4f} ~ {y.max():.4f}")
        break
    
    print("\n✅ Dataset测试通过")
    
    return dataloader, scaler_path


if __name__ == "__main__":
    test_dataset()