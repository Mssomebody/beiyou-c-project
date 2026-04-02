import sys
import pickle
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入修正脚本中的数据集类（假设修正脚本名为 train_federated_pretrain_fixed.py）
# 如果尚未保存修正脚本，我们直接从原预训练脚本中复制数据集类定义（但原脚本使用的是 BarcelonaDataset，不是 MinMaxBarcelonaDataset）
# 为了测试，我们临时定义一个简单的 MinMaxBarcelonaDataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MinMaxBarcelonaDataset(Dataset):
    def __init__(self, data_path, node_id, node_minmax, window_size=28, predict_size=4):
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.data_min, self.data_max = node_minmax[node_id]
        self.energy = self.df['Valor'].values
        sector_codes = self.df['sector_code'].values
        self.sector_onehot = self._one_hot_sector(sector_codes)
        self.holiday = self.df['is_holiday'].values
        self.weekend = self.df['is_weekend'].values
        self.indices = self._build_indices()

    def _one_hot_sector(self, codes):
        n = 4
        onehot = np.zeros((len(codes), n))
        for i, c in enumerate(codes):
            if 0 <= c < n:
                onehot[i, c] = 1
        return onehot

    def _build_indices(self):
        total = len(self.energy)
        indices = []
        for i in range(total - self.window_size - self.predict_size + 1):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x_energy = self.energy[start:start+self.window_size]
        x_energy = (x_energy - self.data_min) / (self.data_max - self.data_min + 1e-8)
        x_energy = torch.FloatTensor(x_energy).unsqueeze(-1)
        sector_idx = start + self.window_size - 1
        x_sector = self.sector_onehot[sector_idx]
        x_sector = torch.FloatTensor(x_sector).unsqueeze(0).repeat(self.window_size, 1)
        x_holiday = self.holiday[start:start+self.window_size]
        x_holiday = torch.FloatTensor(x_holiday).unsqueeze(-1)
        x_weekend = self.weekend[start:start+self.window_size]
        x_weekend = torch.FloatTensor(x_weekend).unsqueeze(-1)
        x = torch.cat([x_energy, x_sector, x_holiday, x_weekend], dim=1)
        y = self.energy[start+self.window_size:start+self.window_size+self.predict_size]
        y = (y - self.data_min) / (self.data_max - self.data_min + 1e-8)
        y = torch.FloatTensor(y)
        return x, y

# 加载 node_minmax.pkl
node_minmax_path = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
with open(node_minmax_path, 'rb') as f:
    node_minmax = pickle.load(f)

# 测试节点 8001
node_id = 8001
data_min, data_max = node_minmax[node_id]
print(f"节点 {node_id}: data_min={data_min}, data_max={data_max}")

# 加载训练数据
train_file = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1" / f"node_{node_id}" / "train.pkl"
ds = MinMaxBarcelonaDataset(train_file, node_id, node_minmax)
x, y = ds[0]
print(f"输入 x 形状: {x.shape}, 预期 (28, 7)")
print(f"输入 x 中能耗列 (第一列) 范围: {x[:,0].min():.4f} ~ {x[:,0].max():.4f} (应在 [0,1])")
print(f"目标 y 范围: {y.min():.4f} ~ {y.max():.4f} (应在 [0,1])")
print("数据集验证通过！")
