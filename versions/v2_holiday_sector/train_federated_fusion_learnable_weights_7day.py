#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粒度融合 + 可学习时段权重 (E5) - 7天窗口版
- 时段权重作为模型的可训练参数，在联邦训练中自动学习
- 所有巴塞罗那节点共享同一套时段权重（全局模式）
- 输入窗口长度支持多天（window_days），每天内4个时段权重共享
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
import warnings
warnings.filterwarnings('ignore')

# 强制无缓冲输出
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# 日志系统
# ============================================================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
        self.log.write(f"\n{'='*60}\n")
        self.log.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*60}\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"fusion_learnable_7day_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
sys.stdout = Logger(LOG_FILE)

# ============================================================
# 配置
# ============================================================
@dataclass
class Config:
    barcelona_data_dir: Path = PROJECT_ROOT / "data/processed/barcelona_ready_2019_2022"
    tsinghua_data_dir: Path = PROJECT_ROOT / "data/processed/tsinghua_6h"
    barcelona_nodes: List[int] = None
    tsinghua_clusters_file: Path = None
    node_type_file: Path = PROJECT_ROOT / "results/barcelona_clustering/node_classification_advanced.csv"
    window_days: int = 7                     # 输入天数
    timezone_offset: int = 0
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
            self.barcelona_nodes = [8001, 8002]
        if self.tsinghua_clusters_file is None:
            self.tsinghua_clusters_file = self.tsinghua_data_dir / "5g_clusters.json"
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # 加载节点类型映射（仅用于标识，不用于权重）
        self.node_types = self._load_node_types()

    def _load_node_types(self):
        if not self.node_type_file.exists():
            print(f"警告: 节点类型文件不存在 {self.node_type_file}，所有节点使用混合权重")
            return {}
        df = pd.read_csv(self.node_type_file)
        node_types = {}
        for _, row in df.iterrows():
            node = row['node']
            type_label = row['type']
            node_types[node] = type_label
        return node_types


# ============================================================
# 巴塞罗那数据集（支持多天输入）
# ============================================================
class BarcelonaCoarseDataset(Dataset):
    def __init__(self, data_path, window_days=7, norm_params=None):
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
        df = df.sort_values('Data')
        dates = pd.to_datetime(df['Data']).unique()
        self.window_days = window_days
        self.samples = []

        # 检查某一天是否有完整4个时段
        def has_full(date):
            day_df = df[pd.to_datetime(df['Data']) == date]
            return set(day_df['hour_code'].unique()) == {0,1,2,3}

        # 按天聚合函数
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

        # 构建样本：连续 window_days 天输入，下一天输出
        for i in range(len(dates) - window_days):
            input_dates = dates[i:i+window_days]
            target_date = dates[i+window_days]
            if all(has_full(d) for d in input_dates) and has_full(target_date):
                input_seq = []
                for d in input_dates:
                    day_df = df[pd.to_datetime(df['Data']) == d]
                    day_agg = day_df.groupby('hour_code').apply(aggregate_group).reset_index()
                    day_agg = day_agg.set_index('hour_code').reindex([0,1,2,3], fill_value=0).reset_index()
                    day_agg = day_agg.sort_values('hour_code')
                    x_day = day_agg[['total_energy','sector_0','sector_1','sector_2','sector_3','is_weekend','is_holiday']].values.astype(np.float32)
                    input_seq.append(x_day)
                x = np.stack(input_seq, axis=0)          # (window_days, 4, 7)
                # 输出目标：target_date 的4个时段
                target_df = df[pd.to_datetime(df['Data']) == target_date]
                target_agg = target_df.groupby('hour_code').apply(aggregate_group).reset_index()
                target_agg = target_agg.set_index('hour_code').reindex([0,1,2,3], fill_value=0).reset_index()
                target_agg = target_agg.sort_values('hour_code')
                y = target_agg['total_energy'].values.astype(np.float32)
                self.samples.append((x, y))

        if not self.samples:
            raise ValueError(f"没有找到连续 {window_days+1} 天完整数据: {data_path}")

        print(f"Loaded {len(self.samples)} samples from {data_path}")

        # 归一化
        if norm_params is None:
            all_x = np.stack([s[0] for s in self.samples])   # (n, window_days, 4, 7)
            all_y = np.stack([s[1] for s in self.samples])   # (n, 4)
            self.x_min = all_x.min(axis=(0,1,2), keepdims=True)
            self.x_max = all_x.max(axis=(0,1,2), keepdims=True)
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
            self.x_min = norm_params['x_min'][None, None, None, :]
            self.x_max = norm_params['x_max'][None, None, None, :]
            self.y_min = norm_params['y_min']
            self.y_max = norm_params['y_max']
            all_x = np.stack([s[0] for s in self.samples])
            all_y = np.stack([s[1] for s in self.samples])
            self.x_norm = (all_x - self.x_min) / (self.x_max - self.x_min + 1e-8)
            self.y_norm = (all_y - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 将输入展平为 (window_days*4, 7)
        x = self.x_norm[idx].reshape(-1, self.x_norm.shape[-1])
        y = self.y_norm[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ============================================================
# 清华聚类组数据集（不变）
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
            self.features.append(data['features'])
            self.targets.append(data['target'])
        if not self.features:
            raise ValueError(f"No valid station data for cluster")
        self.features = np.mean(self.features, axis=0)
        self.targets = np.mean(self.targets, axis=0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features), torch.FloatTensor(self.targets.squeeze())


# ============================================================
# 模型：LSTM + 投影层 + 可学习时段权重（支持多天重复）
# ============================================================
class FusionLSTM(nn.Module):
    def __init__(self, input_dim_barcelona, input_dim_tsinghua, hidden_dim, num_layers, output_dim, dropout, window_days):
        super().__init__()
        self.proj = nn.Linear(input_dim_barcelona, input_dim_tsinghua)
        self.lstm = nn.LSTM(input_dim_tsinghua, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # 可学习时段权重（4个时段）
        self.segment_weights = nn.Parameter(torch.ones(4))
        self.window_days = window_days

    def forward(self, x, is_barcelona):
        if is_barcelona:
            x = self.proj(x)                     # (batch, seq_len, 5)  seq_len = window_days*4
            # 将4个时段的权重重复 window_days 次，得到与序列长度匹配的权重向量
            w = self.segment_weights.repeat(self.window_days)  # (window_days*4)
            w = w.view(1, -1, 1)                 # (1, seq_len, 1)
            x = x * w
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 联邦训练器
# ============================================================
class FederatedTrainer:
    def __init__(self, config, node_types):
        self.config = config
        self.device = torch.device(config.device)
        self.node_types = node_types

    def _create_model(self):
        return FusionLSTM(
            self.config.input_dim_barcelona,
            self.config.input_dim_tsinghua,
            self.config.hidden_dim,
            self.config.num_layers,
            4,
            self.config.dropout,
            self.config.window_days
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

    def evaluate_smape(self, model, test_loaders):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for client_id, loader in test_loaders.items():
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
        torch.save(model.state_dict(), "model_7day_e5_5nodes.pth")
        print("模型已保存为 model_7day_e5_5nodes.pth")
        return smape


# ============================================================
# 数据加载函数
# ============================================================
def load_barcelona_clients(nodes, data_dir, window_days=7, node_types=None):
    clients = {}
    norm_params_dict = {}
    for node in nodes:
        train_path = data_dir / f"node_{node}" / "train.pkl"
        dataset = BarcelonaCoarseDataset(train_path, window_days=window_days, norm_params=None)
        clients[f"barcelona_{node}"] = dataset
        norm_params_dict[node] = dataset.norm_params
    test_clients = {}
    for node in nodes:
        test_path = data_dir / f"node_{node}" / "test.pkl"
        dataset = BarcelonaCoarseDataset(test_path, window_days=window_days, norm_params=norm_params_dict[node])
        test_clients[f"barcelona_{node}"] = dataset
    return clients, test_clients, node_types

def load_tsinghua_clients(clusters_file, data_dir):
    with open(clusters_file, 'r') as f:
        clusters = json.load(f)
    tech = list(clusters.keys())[0]
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
    parser.add_argument('--window_days', type=int, default=7, help='输入天数')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--mu', type=float, default=0.05)
    parser.add_argument('--timezone_offset', type=int, default=0)
    args = parser.parse_args()

    config = Config()
    config.barcelona_nodes = [int(n) for n in args.barcelona_nodes.split(',')]
    config.tsinghua_clusters_file = PROJECT_ROOT / args.tsinghua_clusters
    config.window_days = args.window_days
    config.rounds = args.rounds
    config.mu = args.mu
    config.timezone_offset = args.timezone_offset

    print(f"配置: 窗口天数={config.window_days}")

    print("加载巴塞罗那节点...")
    barcelona_clients, test_clients, node_types = load_barcelona_clients(
        config.barcelona_nodes, config.barcelona_data_dir, window_days=config.window_days, node_types=config.node_types
    )
    print(f"巴塞罗那节点数: {len(barcelona_clients)}")

    print("加载清华聚类组...")
    tsinghua_clients = load_tsinghua_clients(config.tsinghua_clusters_file, config.tsinghua_data_dir)
    print(f"清华聚类组数: {len(tsinghua_clients)}")

    all_clients = {**barcelona_clients, **tsinghua_clients}
    print(f"总客户端数: {len(all_clients)}")

    # 构建 DataLoader
    client_loaders = {}
    for cid, dataset in all_clients.items():
        client_loaders[cid] = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_loaders = {}
    for cid, dataset in test_clients.items():
        test_loaders[cid] = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    # 训练
    trainer = FederatedTrainer(config, node_types)
    smape = trainer.train_full(config.mu, client_loaders, test_loaders)
    print(f"\n最终 sMAPE: {smape:.2f}%")

if __name__ == "__main__":
    main()
