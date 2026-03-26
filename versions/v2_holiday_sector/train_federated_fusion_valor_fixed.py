#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粒度融合：巴塞罗那滑动窗口 + 清华聚类组（6小时粒度）
修复：DataLoader 的 drop_last 改为 False，避免空 loader
"""

import sys
import os
import json
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# 配置
# ============================================================
@dataclass
class Config:
    barcelona_data_dir: Path = PROJECT_ROOT / "data/processed/barcelona_ready_2019_2022"
    tsinghua_data_dir: Path = PROJECT_ROOT / "data/processed/tsinghua_6h"
    barcelona_nodes: List[int] = None
    tsinghua_clusters_file: Path = None
    batch_size: int = 64
    input_dim_barcelona: int = 7
    input_dim_tsinghua: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    rounds: int = 10
    local_epochs: int = 5
    mu: float = 0.05
    device: str = 'cpu'
    seed: int = 42

    def __post_init__(self):
        if self.barcelona_nodes is None:
            self.barcelona_nodes = [8001, 8002]  # 默认2个
        if self.tsinghua_clusters_file is None:
            self.tsinghua_clusters_file = self.tsinghua_data_dir / "5g_clusters.json"
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


# ============================================================
# 巴塞罗那滑动窗口数据集（4个时段输入/输出）
# ============================================================
class BarcelonaCoarseDataset(Dataset):
    def __init__(self, data_path, norm_params=None):
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
        df = df.sort_values('Data')
        dates = pd.to_datetime(df['Data']).unique()
        self.samples = []

        for i in range(len(dates)-1):
            prev_date = dates[i]
            last_date = dates[i+1]
            # 检查是否都有完整4个时段
            def has_full(date):
                day_df = df[pd.to_datetime(df['Data']) == date]
                return set(day_df['hour_code'].unique()) == {0,1,2,3}
            if has_full(prev_date) and has_full(last_date):
                input_df = df[pd.to_datetime(df['Data']) == prev_date]
                target_df = df[pd.to_datetime(df['Data']) == last_date]

                def aggregate_group(group):
                    total_energy = group['Valor'].sum()
                    weekend = group['is_weekend'].iloc[0]
                    holiday = group['is_holiday'].iloc[0]
                    sector_means = group[['sector_0','sector_1','sector_2','sector_3']].mean().values
                    return pd.Series({
                        'total_energy': total_energy,
                        'is_weekend': weekend,
                        'is_holiday': holiday,
                        'sector_0': sector_means[0],
                        'sector_1': sector_means[1],
                        'sector_2': sector_means[2],
                        'sector_3': sector_means[3]
                    })

                input_agg = input_df.groupby('hour_code').apply(aggregate_group).reset_index()
                target_agg = target_df.groupby('hour_code').apply(aggregate_group).reset_index()
                # 确保4个时段
                input_agg = input_agg.set_index('hour_code').reindex([0,1,2,3], fill_value=0).reset_index()
                target_agg = target_agg.set_index('hour_code').reindex([0,1,2,3], fill_value=0).reset_index()
                input_agg = input_agg.sort_values('hour_code')
                target_agg = target_agg.sort_values('hour_code')

                x = input_agg[['total_energy','sector_0','sector_1','sector_2','sector_3','is_weekend','is_holiday']].values.astype(np.float32)
                y = target_agg['total_energy'].values.astype(np.float32)
                self.samples.append((x, y))

        if not self.samples:
            raise ValueError(f"没有找到连续两天完整数据: {data_path}")

        print(f"Loaded {len(self.samples)} samples from {data_path}")

        # 归一化
        if norm_params is None:
            # 训练集：计算全局 min/max
            all_x = np.stack([s[0] for s in self.samples])  # (n,4,7)
            all_y = np.stack([s[1] for s in self.samples])  # (n,4)
            self.x_min = all_x.min(axis=(0,1), keepdims=True)  # (1,1,7)
            self.x_max = all_x.max(axis=(0,1), keepdims=True)
            self.y_min = all_y.min()
            self.y_max = all_y.max()
            self.norm_params = {
                'x_min': self.x_min.squeeze(),
                'x_max': self.x_max.squeeze(),
                'y_min': self.y_min,
                'y_max': self.y_max
            }
            self.x_norm = (all_x - self.x_min) / (self.x_max - self.x_min + 1e-8)
            self.y_norm = (all_y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        else:
            self.x_min = norm_params['x_min'][None, None, :]
            self.x_max = norm_params['x_max'][None, None, :]
            self.y_min = norm_params['y_min']
            self.y_max = norm_params['y_max']
            all_x = np.stack([s[0] for s in self.samples])
            all_y = np.stack([s[1] for s in self.samples])
            self.x_norm = (all_x - self.x_min) / (self.x_max - self.x_min + 1e-8)
            self.y_norm = (all_y - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x_norm[idx]), torch.FloatTensor(self.y_norm[idx])


# ============================================================
# 清华聚类组数据集（每个组一个样本）
# ============================================================
class TsinghuaClusterDataset(Dataset):
    def __init__(self, cluster_stations, data_dir):
        self.features = []
        self.targets = []
        for sid in cluster_stations:
            file_path = data_dir / f"station_{sid}.pkl"
            if not file_path.exists():
                continue
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            self.features.append(data['features'])   # (4,5)
            self.targets.append(data['target'])     # (4,1)
        if not self.features:
            raise ValueError(f"No valid station data for cluster")
        self.features = np.mean(self.features, axis=0)   # (4,5)
        self.targets = np.mean(self.targets, axis=0)     # (4,1)
        # 归一化（清华数据已经归一化到[0,1]左右，不再额外处理）

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features), torch.FloatTensor(self.targets.squeeze())


# ============================================================
# 模型：LSTM + 投影层
# ============================================================
class FusionLSTM(nn.Module):
    def __init__(self, input_dim_barcelona, input_dim_tsinghua, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim_barcelona, input_dim_tsinghua)
        self.lstm = nn.LSTM(input_dim_tsinghua, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, is_barcelona):
        if is_barcelona:
            x = self.proj(x)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 联邦训练器
# ============================================================
class FederatedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

    def _create_model(self):
        return FusionLSTM(
            self.config.input_dim_barcelona,
            self.config.input_dim_tsinghua,
            self.config.hidden_dim,
            self.config.num_layers,
            4,  # output_dim
            self.config.dropout
        )

    def train_round(self, model, client_loaders, mu):
        client_weights = []
        client_sizes = []
        client_ids = []
        total_loss = 0.0
        num_clients = 0

        for client_id, loader in client_loaders.items():
            client_ids.append(client_id)
            local_model = self._create_model().to(self.device)
            local_model.load_state_dict(model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()

            is_barcelona = client_id.startswith('barcelona')
            client_loss = 0.0

            for _ in range(self.config.local_epochs):
                epoch_loss = 0.0
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(x, is_barcelona)
                    loss = criterion(output, y)
                    if mu > 0:
                        prox_loss = 0.0
                        for param, global_param in zip(local_model.parameters(), model.parameters()):
                            prox_loss += torch.norm(param - global_param) ** 2
                        loss += (mu / 2) * prox_loss
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if len(loader) > 0:
                    avg_epoch_loss = epoch_loss / len(loader)
                else:
                    avg_epoch_loss = 0.0
                client_loss += avg_epoch_loss
            if self.config.local_epochs > 0:
                client_loss /= self.config.local_epochs
            total_loss += client_loss
            num_clients += 1

            client_weights.append(local_model.state_dict())
            client_sizes.append(len(loader.dataset))

        # 聚合（按数据量加权）
        total = sum(client_sizes)
        if total == 0:
            return model, 0.0
        global_weights = {}
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key])
            for w, size in zip(client_weights, client_sizes):
                global_weights[key] += w[key] * (size / total)
        model.load_state_dict(global_weights)

        avg_loss = total_loss / num_clients if num_clients > 0 else 0.0
        return model, avg_loss

    def evaluate_loss(self, model, loaders):
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for loader in loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x, is_barcelona=True)
                    loss = criterion(output, y)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)
        return total_loss / total_samples if total_samples > 0 else 0.0

    def evaluate_smape(self, model, test_loaders):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for loader in test_loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x, is_barcelona=True)
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        print(f"预测值范围: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
        print(f"目标值范围: [{all_targets.min():.4f}, {all_targets.max():.4f}]")

        denominator = (np.abs(all_targets) + np.abs(all_preds)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(np.abs(all_targets - all_preds) / denominator) * 100
        return smape

    def train_full(self, mu, client_loaders, test_loaders):
        model = self._create_model().to(self.device)
        for round_num in range(1, self.config.rounds + 1):
            model, avg_loss = self.train_round(model, client_loaders, mu)
            print(f"Round {round_num}: avg_train_loss = {avg_loss:.6f}")
        smape = self.evaluate_smape(model, test_loaders)
        return smape


# ============================================================
# 数据加载
# ============================================================
def load_barcelona_clients(nodes, data_dir):
    clients = {}
    norm_params_dict = {}
    for node in nodes:
        train_path = data_dir / f"node_{node}" / "train.pkl"
        dataset = BarcelonaCoarseDataset(train_path, norm_params=None)
        clients[f"barcelona_{node}"] = dataset
        norm_params_dict[node] = dataset.norm_params
    # 测试集也使用训练集归一化参数
    test_clients = {}
    for node in nodes:
        test_path = data_dir / f"node_{node}" / "test.pkl"
        dataset = BarcelonaCoarseDataset(test_path, norm_params=norm_params_dict[node])
        test_clients[f"barcelona_{node}"] = dataset
    return clients, test_clients

def load_tsinghua_clients(clusters_file, data_dir):
    with open(clusters_file, 'r') as f:
        clusters = json.load(f)
    tech = list(clusters.keys())[0]  # 假设是 '5g'
    clients = {}
    for cluster_id, station_ids in clusters[tech].items():
        dataset = TsinghuaClusterDataset(station_ids, data_dir / tech)
        clients[f"tsinghua_{cluster_id}"] = dataset
    return clients


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--barcelona_nodes', type=str, default='8001,8002')
    parser.add_argument('--tsinghua_clusters', type=str, default='data/processed/tsinghua_6h/5g_clusters.json')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--mu', type=float, default=0.05)
    args = parser.parse_args()

    config = Config()
    config.barcelona_nodes = [int(n) for n in args.barcelona_nodes.split(',')]
    config.tsinghua_clusters_file = PROJECT_ROOT / args.tsinghua_clusters
    config.rounds = args.rounds
    config.mu = args.mu

    print("加载巴塞罗那节点...")
    barcelona_clients, test_clients = load_barcelona_clients(config.barcelona_nodes, config.barcelona_data_dir)
    print(f"巴塞罗那节点数: {len(barcelona_clients)}")

    print("加载清华聚类组...")
    tsinghua_clients = load_tsinghua_clients(config.tsinghua_clusters_file, config.tsinghua_data_dir)
    print(f"清华聚类组数: {len(tsinghua_clients)}")

    all_clients = {**barcelona_clients, **tsinghua_clients}
    print(f"总客户端数: {len(all_clients)}")

    # 转换为 DataLoader（统一使用 drop_last=False）
    client_loaders = {}
    for cid, dataset in all_clients.items():
        client_loaders[cid] = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_loaders = {}
    for cid, dataset in test_clients.items():
        test_loaders[cid] = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    # 训练
    trainer = FederatedTrainer(config)
    smape = trainer.train_full(config.mu, client_loaders, test_loaders)
    print(f"\n最终 sMAPE: {smape:.2f}%")


if __name__ == "__main__":
    main()
