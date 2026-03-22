"""
巴塞罗那基站能耗数据集 v1
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class BarcelonaDataset(Dataset):
    def __init__(self, data_path, window_size=28, predict_size=4,
                 sector_feature=True, holiday_feature=True, weekend_feature=True):
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.sector_feature = sector_feature
        self.holiday_feature = holiday_feature
        self.weekend_feature = weekend_feature
        
        self.energy = self.df['Valor_norm'].values
        
        # 部门特征
        if sector_feature:
            sector_codes = self.df['sector_code'].values
            self.sector_onehot = self._one_hot_sector(sector_codes)
        
        # 节假日特征
        if holiday_feature:
            self.holiday = self.df['is_holiday'].values
        
        # 周末特征
        if weekend_feature:
            self.weekend = self.df['is_weekend'].values
        
        self.indices = self._build_indices()
    
    def _one_hot_sector(self, sector_codes):
        n_sectors = 4
        onehot = np.zeros((len(sector_codes), n_sectors))
        for i, code in enumerate(sector_codes):
            if 0 <= code < n_sectors:
                onehot[i, code] = 1
        return onehot
    
    def _build_indices(self):
        indices = []
        total_len = len(self.energy)
        for i in range(total_len - self.window_size - self.predict_size + 1):
            indices.append(i)
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        # 能耗
        x_energy = self.energy[start_idx:start_idx + self.window_size]
        x_energy = torch.FloatTensor(x_energy).unsqueeze(-1)
        
        all_features = [x_energy]
        
        # 部门特征
        if self.sector_feature:
            sector_idx = start_idx + self.window_size - 1
            x_sector = self.sector_onehot[sector_idx]
            x_sector = torch.FloatTensor(x_sector)
            all_features.append(x_sector.unsqueeze(0).repeat(self.window_size, 1))
        
        # 节假日特征
        if self.holiday_feature:
            x_holiday = self.holiday[start_idx:start_idx + self.window_size]
            x_holiday = torch.FloatTensor(x_holiday).unsqueeze(-1)
            all_features.append(x_holiday)
        
        # 周末特征
        if self.weekend_feature:
            x_weekend = self.weekend[start_idx:start_idx + self.window_size]
            x_weekend = torch.FloatTensor(x_weekend).unsqueeze(-1)
            all_features.append(x_weekend)
        
        x = torch.cat(all_features, dim=1)
        
        y = self.energy[start_idx + self.window_size:start_idx + self.window_size + self.predict_size]
        y = torch.FloatTensor(y)
        
        return x, y


def get_node_data_loader(node_id, split='train', batch_size=64, shuffle=True,
                         window_size=28, predict_size=4,
                         sector_feature=True, holiday_feature=True, weekend_feature=True):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "processed", "barcelona_ready_v1", f"node_{node_id}")
    
    file_path = os.path.join(data_dir, f"{split}.pkl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    dataset = BarcelonaDataset(
        data_path=file_path,
        window_size=window_size,
        predict_size=predict_size,
        sector_feature=sector_feature,
        holiday_feature=holiday_feature,
        weekend_feature=weekend_feature
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    scaler_path = os.path.join(data_dir, "scaler.pkl")
    
    return dataloader, scaler_path, dataset


if __name__ == "__main__":
    print("v1 Dataset 测试")
    loader, _, _ = get_node_data_loader(8001, 'train', 1)
    for x, y in loader:
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {y.shape}")
        break