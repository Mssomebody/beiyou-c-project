#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粒度融合 + 知识迁移加权 (E4) - 时差偏移版
- 在原始脚本基础上，确保时差偏移参数生效
- 添加权重打印，便于验证
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
LOG_FILE = LOG_DIR / f"fusion_offset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    weights_file: Path = PROJECT_ROOT / "results/barcelona_clustering/barcelona_weights_for_federated.json"
    node_type_file: Path = PROJECT_ROOT / "results/barcelona_clustering/node_classification_advanced.csv"
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
            if '工业' in type_label:
                node_types[node] = '4g_like'
            elif '商业' in type_label:
                node_types[node] = '5g_like'
            else:
                node_types[node] = 'mixed'
        return node_types


# ============================================================
# 加载时段权重（支持时差偏移，带打印）
# ============================================================
def load_all_segment_weights(weights_file, timezone_offset=0):
    with open(weights_file, 'r') as f:
        data = json.load(f)
    segments_order = ['00-06', '06-12', '12-18', '18-24']
    weights = {}
    for wtype in ['mixed', '4g_like', '5g_like']:
        w = [data[seg][wtype] for seg in segments_order]
        print(f"原始权重 ({wtype}): {w}")
        if timezone_offset != 0:
            shift = int(timezone_offset / 6)   # 6小时一个时段
            w = w[-shift:] + w[:-shift] if shift != 0 else w
            print(f"移位后权重 ({wtype}): {w}")
        weights[wtype] = torch.tensor(w, dtype=torch.float32)
    return weights


# ============================================================
# 巴塞罗那滑动窗口数据集
# ============================================================
class BarcelonaCoarseDataset(Dataset):
    def __init__(self, data_path, norm_params=None):
        import pandas as pd
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
        df = df.sort_values('Data')
        dates = pd.to_datetime(df['Data']).unique()
        self.samples = []

        for i in range(len(dates)-1):
            prev_date = dates[i]
            last_date = dates[i+1]
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
            all_x = np.stack([s[0] for s in self.samples])
            all_y = np.stack([s[1] for s in self.samples])
            self.x_min = all_x.min(axis=(0,1), keepdims=True)
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
# 清华聚类组数据集
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
# 联邦训练器（输入加权版）
# ============================================================
class FederatedTrainer:
    def __init__(self, config, all_weights, node_types):
        self.config = config
        self.device = torch.device(config.device)
        self.all_weights = all_weights
        self.node_types = node_types

    def _create_model(self):
        return FusionLSTM(
            self.config.input_dim_barcelona,
            self.config.input_dim_tsinghua,
            self.config.hidden_dim,
            self.config.num_layers,
            4,
            self.config.dropout
        )

    def get_segment_weights(self, client_id):
        if client_id.startswith('barcelona'):
            node = int(client_id.split('_')[1])
            wtype = self.node_types.get(node, 'mixed')
            return self.all_weights[wtype].to(self.device)
        else:
            return self.all_weights['mixed'].to(self.device)

    def apply_time_weights(self, x, client_id):
        w = self.get_segment_weights(client_id)
        return x * w.view(1, -1, 1)

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
                    if is_barcelona:
                        x = self.apply_time_weights(x, client_id)
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
                    x = self.apply_time_weights(x, client_id)
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
# 数据加载函数
# ============================================================
def load_barcelona_clients(nodes, data_dir, node_types):
    clients = {}
    norm_params_dict = {}
    for node in nodes:
        train_path = data_dir / f"node_{node}" / "train.pkl"
        dataset = BarcelonaCoarseDataset(train_path, norm_params=None)
        clients[f"barcelona_{node}"] = dataset
        norm_params_dict[node] = dataset.norm_params
    test_clients = {}
    for node in nodes:
        test_path = data_dir / f"node_{node}" / "test.pkl"
        dataset = BarcelonaCoarseDataset(test_path, norm_params=norm_params_dict[node])
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
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--mu', type=float, default=0.05)
    parser.add_argument('--timezone_offset', type=int, default=0)
    args = parser.parse_args()

    config = Config()
    config.barcelona_nodes = [int(n) for n in args.barcelona_nodes.split(',')]
    config.tsinghua_clusters_file = PROJECT_ROOT / args.tsinghua_clusters
    config.rounds = args.rounds
    config.mu = args.mu
    config.timezone_offset = args.timezone_offset

    print(f"时差偏移参数: {config.timezone_offset}")

    all_weights = load_all_segment_weights(config.weights_file, config.timezone_offset)
    print("时段权重已加载（混合/4G-like/5G-like）")

    print("加载巴塞罗那节点...")
    barcelona_clients, test_clients, node_types = load_barcelona_clients(
        config.barcelona_nodes, config.barcelona_data_dir, config.node_types
    )
    print(f"巴塞罗那节点数: {len(barcelona_clients)}")

    print("加载清华聚类组...")
    tsinghua_clients = load_tsinghua_clients(config.tsinghua_clusters_file, config.tsinghua_data_dir)
    print(f"清华聚类组数: {len(tsinghua_clients)}")

    all_clients = {**barcelona_clients, **tsinghua_clients}
    print(f"总客户端数: {len(all_clients)}")

    client_loaders = {}
    for cid, dataset in all_clients.items():
        client_loaders[cid] = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_loaders = {}
    for cid, dataset in test_clients.items():
        test_loaders[cid] = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    trainer = FederatedTrainer(config, all_weights, node_types)
    smape = trainer.train_full(config.mu, client_loaders, test_loaders)
    print(f"\n最终 sMAPE: {smape:.2f}%")


if __name__ == "__main__":
    main()
