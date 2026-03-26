#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级节点筛选脚本
- 使用与 train_federated_pro_valor.py 完全一致的数据加载和模型
- 对每个节点独立训练10轮，记录验证集最佳损失
- 输出正常/异常节点列表，保存结果到 results/node_quality.json
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录（脚本所在目录的父目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed/barcelona_ready_2019_2022"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 模型配置（与 train_federated_pro_valor.py 一致）
class Config:
    window_size = 28
    predict_size = 4
    batch_size = 64
    input_dim = 7
    hidden_dim = 64
    num_layers = 2
    dropout = 0.2
    learning_rate = 0.001
    epochs = 10
    device = 'cpu'
    seed = 42

# 复制自 train_federated_pro_valor.py 的 Dataset 类（确保一致）
class BarcelonaDatasetMinMax(Dataset):
    def __init__(self, data_path, window_size=28, predict_size=4,
                 sector_feature=True, holiday_feature=True, weekend_feature=True,
                 norm_params=None):
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.sector_feature = sector_feature
        self.holiday_feature = holiday_feature
        self.weekend_feature = weekend_feature

        # 原始能耗 Valor
        self.energy = self.df['Valor'].values.astype(np.float32)

        if norm_params is None:
            # 训练集：计算归一化参数
            self.energy_min = self.energy.min()
            self.energy_max = self.energy.max()
            self.norm_params = {
                'x_min': self.energy_min,
                'x_max': self.energy_max,
                'y_min': self.energy_min,
                'y_max': self.energy_max
            }
        else:
            self.energy_min = norm_params['x_min']
            self.energy_max = norm_params['x_max']
            self.norm_params = norm_params

        self.energy_norm = (self.energy - self.energy_min) / (self.energy_max - self.energy_min + 1e-8)

        if sector_feature:
            sector_codes = self.df['sector_code'].values
            self.sector_onehot = self._one_hot_sector(sector_codes)

        if holiday_feature:
            self.holiday = self.df['is_holiday'].values.astype(np.float32)

        if weekend_feature:
            self.weekend = self.df['is_weekend'].values.astype(np.float32)

        self.indices = self._build_indices()

    def _one_hot_sector(self, sector_codes):
        n_sectors = 4
        onehot = np.zeros((len(sector_codes), n_sectors), dtype=np.float32)
        for i, code in enumerate(sector_codes):
            if 0 <= code < n_sectors:
                onehot[i, code] = 1
        return onehot

    def _build_indices(self):
        indices = []
        total_len = len(self.energy_norm)
        for i in range(total_len - self.window_size - self.predict_size + 1):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        x_energy = self.energy_norm[start_idx:start_idx + self.window_size]
        x_energy = torch.FloatTensor(x_energy).unsqueeze(-1)

        all_features = [x_energy]

        if self.sector_feature:
            sector_idx = start_idx + self.window_size - 1
            x_sector = self.sector_onehot[sector_idx]
            x_sector = torch.FloatTensor(x_sector)
            all_features.append(x_sector.unsqueeze(0).repeat(self.window_size, 1))

        if self.holiday_feature:
            x_holiday = self.holiday[start_idx:start_idx + self.window_size]
            x_holiday = torch.FloatTensor(x_holiday).unsqueeze(-1)
            all_features.append(x_holiday)

        if self.weekend_feature:
            x_weekend = self.weekend[start_idx:start_idx + self.window_size]
            x_weekend = torch.FloatTensor(x_weekend).unsqueeze(-1)
            all_features.append(x_weekend)

        x = torch.cat(all_features, dim=1)
        y = self.energy_norm[start_idx + self.window_size:start_idx + self.window_size + self.predict_size]
        y = torch.FloatTensor(y)
        return x, y


# 模型定义（与联邦脚本一致）
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


def train_single_node(node, config):
    """单节点训练，返回最佳验证损失"""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_path = DATA_DIR / f"node_{node}" / "train.pkl"
    val_path = DATA_DIR / f"node_{node}" / "val.pkl"

    if not train_path.exists() or not val_path.exists():
        logger.warning(f"节点 {node} 数据文件缺失")
        return None

    # 训练集（计算归一化参数）
    train_dataset = BarcelonaDatasetMinMax(
        str(train_path),
        window_size=config.window_size,
        predict_size=config.predict_size,
        sector_feature=True,
        holiday_feature=True,
        weekend_feature=True,
        norm_params=None
    )
    # 验证集（使用训练集的归一化参数）
    val_dataset = BarcelonaDatasetMinMax(
        str(val_path),
        window_size=config.window_size,
        predict_size=config.predict_size,
        sector_feature=True,
        holiday_feature=True,
        weekend_feature=True,
        norm_params=train_dataset.norm_params
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    model = LSTMPredictor(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_dim=config.predict_size,
        dropout=config.dropout
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
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
                x, y = x.to(config.device), y.to(config.device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # 每2轮打印一次进度（可选）
        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info(f"节点 {node} epoch {epoch+1}/{config.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return best_val_loss


def main():
    config = Config()
    nodes = list(range(8001, 8043))
    results = []

    logger.info(f"开始节点质量评估（共{len(nodes)}个节点，每个节点{config.epochs}轮，预计耗时约{len(nodes)*3}分钟）")

    for node in nodes:
        loss = train_single_node(node, config)
        if loss is not None:
            results.append({'node': node, 'val_loss': loss})
            logger.info(f"节点 {node}: 最佳验证损失 = {loss:.6f}")
        else:
            logger.warning(f"节点 {node} 跳过")

    # 设定阈值：参考2节点正常值（0.0067）放大10倍
    threshold = 0.05  # 可调整
    normal_nodes = [r['node'] for r in results if r['val_loss'] < threshold]
    abnormal_nodes = [r['node'] for r in results if r['val_loss'] >= threshold]

    # 保存结果
    output = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'threshold': threshold,
        'normal_nodes': normal_nodes,
        'abnormal_nodes': abnormal_nodes,
        'details': results
    }
    with open(OUTPUT_DIR / 'node_quality.json', 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n结果已保存至 {OUTPUT_DIR / 'node_quality.json'}")
    logger.info(f"正常节点数: {len(normal_nodes)}")
    logger.info(f"异常节点数: {len(abnormal_nodes)}")
    logger.info(f"正常节点列表: {normal_nodes}")
    logger.info(f"异常节点列表: {abnormal_nodes}")

if __name__ == "__main__":
    main()
