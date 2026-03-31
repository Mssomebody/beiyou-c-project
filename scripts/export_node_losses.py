#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出所有节点的每轮验证损失值到 CSV 文件（每轮记录）
基于与 screen_nodes.py 完全一致的数据集和模型（滑动窗口，28步输入，4步输出）
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# 项目根目录（脚本所在目录的父目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_2019_2022"
OUTPUT_CSV = PROJECT_ROOT / "data" / "node_epoch_losses.csv"

# ============================================================
# 数据集类（与 screen_nodes.py 一致，基于滑动窗口）
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, window_size=28, predict_size=4):
        df = pd.read_pickle(data_path)
        self.energy = df['Valor_norm'].values.astype(np.float32)
        self.window_size = window_size
        self.predict_size = predict_size
        self.indices = self._build_indices()

    def _build_indices(self):
        indices = []
        total_len = len(self.energy)
        for i in range(total_len - self.window_size - self.predict_size + 1):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.energy[start:start + self.window_size]
        y = self.energy[start + self.window_size:start + self.window_size + self.predict_size]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y)


# ============================================================
# 模型定义（LSTM，与联邦学习一致）
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 训练单个节点并返回每轮的损失记录
# ============================================================
def train_node(node_id, epochs=10, batch_size=64, lr=0.001):
    train_path = DATA_DIR / f"node_{node_id}" / "train.pkl"
    if not train_path.exists():
        print(f"节点 {node_id} 数据不存在，跳过")
        return None

    dataset = TimeSeriesDataset(train_path, window_size=28, predict_size=4)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_records = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        epoch_records.append({
            'node': node_id,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        print(f"节点 {node_id} - Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return epoch_records


def main():
    nodes = list(range(8001, 8043))  # 8001-8042
    all_records = []
    for node in nodes:
        print(f"\n处理节点 {node}...")
        records = train_node(node)
        if records:
            all_records.extend(records)
        else:
            print(f"节点 {node} 处理失败")

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n每轮损失已保存至: {OUTPUT_CSV}")
    print(df.head(10))


if __name__ == "__main__":
    main()
