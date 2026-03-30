#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级粒度融合 + 时段加权 (7天窗口版)
- 巴塞罗那数据：7天历史 (28个时间步)
- 清华数据：6小时粒度 (4个时间步)
- 自动日志保存至 logs/ 目录
- 支持动态节点权重 (可选)
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
LOG_FILE = LOG_DIR / f"fusion_7day_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
sys.stdout = Logger(LOG_FILE)

# ============================================================
# 巴塞罗那 7天窗口数据集（直接复制自不加权基线，避免导入冲突）
# ============================================================
class BarcelonaDatasetValorNorm(Dataset):
    def __init__(self, data_path, window_size=28, predict_size=4,
                 sector_feature=True, holiday_feature=True, weekend_feature=True,
                 norm_params=None):
        """
        norm_params: dict with keys 'x_min', 'x_max', 'y_min', 'y_max' (for training set)
        If None, compute from this data (only for training set)
        """
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.sector_feature = sector_feature
        self.holiday_feature = holiday_feature
        self.weekend_feature = weekend_feature

        # 直接使用 Valor_norm 列
        self.energy = self.df['Valor_norm'].values.astype(np.float32)

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
    window_size: int = 28                     # 7天
    predict_size: int = 4
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
    use_node_weights: bool = True             # 是否使用动态节点权重（时段加权）

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
# 加载时段权重（用于时段加权）
# ============================================================
def load_all_segment_weights(weights_file):
    with open(weights_file, 'r') as f:
        data = json.load(f)
    segments_order = ['00-06', '06-12', '12-18', '18-24']
    weights = {}
    for wtype in ['mixed', '4g_like', '5g_like']:
        w = [data[seg][wtype] for seg in segments_order]
        weights[wtype] = torch.tensor(w, dtype=torch.float32)
    return weights


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
# 联邦训练器（支持时段加权）
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
                    if is_barcelona and self.config.use_node_weights:
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
                    if self.config.use_node_weights:
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
def load_barcelona_clients(nodes, data_dir, config):
    clients = {}
    norm_params_dict = {}
    for node in nodes:
        train_path = data_dir / f"node_{node}" / "train.pkl"
        dataset = BarcelonaDatasetValorNorm(
            data_path=str(train_path),
            window_size=config.window_size,
            predict_size=config.predict_size,
            sector_feature=True,
            holiday_feature=True,
            weekend_feature=True,
            norm_params=None
        )
        clients[f"barcelona_{node}"] = dataset
        norm_params_dict[node] = dataset.norm_params
    test_clients = {}
    for node in nodes:
        test_path = data_dir / f"node_{node}" / "test.pkl"
        dataset = BarcelonaDatasetValorNorm(
            data_path=str(test_path),
            window_size=config.window_size,
            predict_size=config.predict_size,
            sector_feature=True,
            holiday_feature=True,
            weekend_feature=True,
            norm_params=norm_params_dict[node]
        )
        test_clients[f"barcelona_{node}"] = dataset
    return clients, test_clients, config.node_types

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
    parser.add_argument('--window_size', type=int, default=28, help='输入窗口大小（时间步）')
    parser.add_argument('--use_node_weights', action='store_true', default=True, help='是否使用时段加权')
    args = parser.parse_args()

    config = Config()
    config.barcelona_nodes = [int(n) for n in args.barcelona_nodes.split(',')]
    config.tsinghua_clusters_file = PROJECT_ROOT / args.tsinghua_clusters
    config.rounds = args.rounds
    config.mu = args.mu
    config.window_size = args.window_size
    config.use_node_weights = args.use_node_weights

    print(f"配置: 窗口大小={config.window_size}, 时段加权={config.use_node_weights}")

    # 加载时段权重（如果需要）
    all_weights = None
    if config.use_node_weights:
        all_weights = load_all_segment_weights(config.weights_file)
        print("时段权重已加载（混合/4G-like/5G-like）")

    # 加载巴塞罗那节点
    print("加载巴塞罗那节点...")
    barcelona_clients, test_clients, node_types = load_barcelona_clients(
        config.barcelona_nodes, config.barcelona_data_dir, config
    )
    print(f"巴塞罗那节点数: {len(barcelona_clients)}")

    # 加载清华聚类组
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
    trainer = FederatedTrainer(config, all_weights, node_types)
    smape = trainer.train_full(config.mu, client_loaders, test_loaders)
    print(f"\n最终 sMAPE: {smape:.2f}%")


if __name__ == "__main__":
    main()
