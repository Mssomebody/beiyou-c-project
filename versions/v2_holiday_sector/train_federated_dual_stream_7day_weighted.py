#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双流粒度融合 + 时段加权（7天窗口）
- 巴塞罗那7天窗口（28步）输入进行时段加权
- 清华4步输入不变
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
LOG_FILE = LOG_DIR / f"dual_stream_7day_weighted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    window_size_barcelona: int = 28
    window_size_tsinghua: int = 4
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
    use_time_weights: bool = True

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
# 加载时段权重
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
# 巴塞罗那数据集（7天窗口）
# ============================================================
class Barcelona7DayDataset(Dataset):
    def __init__(self, data_path, norm_params=None):
        self.df = pd.read_pickle(data_path)
        self.energy = self.df['Valor_norm'].values.astype(np.float32)
        self.sector_onehot = self._one_hot_sector(self.df['sector_code'].values)
        self.holiday = self.df['is_holiday'].values.astype(np.float32)
        self.weekend = self.df['is_weekend'].values.astype(np.float32)
        self.indices = self._build_indices()
        self.norm_params = norm_params

    def _one_hot_sector(self, sector_codes):
        n_sectors = 4
        onehot = np.zeros((len(sector_codes), n_sectors), dtype=np.float32)
        for i, code in enumerate(sector_codes):
            if 0 <= code < n_sectors:
                onehot[i, code] = 1
        return onehot

    def _build_indices(self):
        indices = []
        total_len = len(self.energy)
        for i in range(total_len - 28 - 4 + 1):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x_energy = self.energy[start:start+28]
        x_energy = torch.FloatTensor(x_energy).unsqueeze(-1)

        x_sector = self.sector_onehot[start+27]  # 取最后一个时间步的部门
        x_sector = torch.FloatTensor(x_sector).repeat(28, 1)

        x_holiday = self.holiday[start:start+28]
        x_holiday = torch.FloatTensor(x_holiday).unsqueeze(-1)

        x_weekend = self.weekend[start:start+28]
        x_weekend = torch.FloatTensor(x_weekend).unsqueeze(-1)

        x = torch.cat([x_energy, x_sector, x_holiday, x_weekend], dim=1)
        y = self.energy[start+28:start+28+4]
        y = torch.FloatTensor(y)
        return x, y


# ============================================================
# 清华数据集（4步）
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
        self.features = np.mean(self.features, axis=0)
        self.targets = np.mean(self.targets, axis=0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features), torch.FloatTensor(self.targets.squeeze())


# ============================================================
# 双流模型（支持时段加权）
# ============================================================
class DualStreamLSTM(nn.Module):
    def __init__(self, input_dim_barcelona, input_dim_tsinghua, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.lstm_b = nn.LSTM(input_dim_barcelona, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm_t = nn.LSTM(input_dim_tsinghua, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x_b, x_t):
        _, (h_b, _) = self.lstm_b(x_b)
        _, (h_t, _) = self.lstm_t(x_t)
        last_h_b = h_b[-1]
        last_h_t = h_t[-1]
        concat = torch.cat([last_h_b, last_h_t], dim=1)
        return self.fc(concat)


# ============================================================
# 联邦训练器（带时段加权）
# ============================================================
class FederatedTrainer:
    def __init__(self, config, all_weights):
        self.config = config
        self.device = torch.device(config.device)
        self.all_weights = all_weights

    def _create_model(self):
        return DualStreamLSTM(
            input_dim_barcelona=self.config.input_dim_barcelona,
            input_dim_tsinghua=self.config.input_dim_tsinghua,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            output_dim=self.config.predict_size,
            dropout=self.config.dropout
        )

    def get_segment_weights(self, client_id):
        if client_id.startswith('barcelona'):
            node = int(client_id.split('_')[1])
            wtype = self.config.node_types.get(node, 'mixed')
            return self.all_weights[wtype].to(self.device)
        else:
            return self.all_weights['mixed'].to(self.device)

    def apply_time_weights(self, x_b, client_id):
        w = self.get_segment_weights(client_id)  # (4,)
        # 扩展到28个时间步
        w_seq = w.repeat(28 // 4)  # 7次重复
        return x_b * w_seq.view(1, -1, 1)

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
                    x = x.to(self.device)
                    y = y.to(self.device)
                    if is_barcelona:
                        x_b = x
                        if self.config.use_time_weights:
                            x_b = self.apply_time_weights(x_b, client_id)
                        x_t = torch.zeros(x_b.size(0), 4, self.config.input_dim_tsinghua).to(self.device)
                    else:
                        x_t = x
                        x_b = torch.zeros(x_t.size(0), 28, self.config.input_dim_barcelona).to(self.device)
                    optimizer.zero_grad()
                    output = local_model(x_b, x_t)
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
                    x = x.to(self.device)
                    y = y.to(self.device)
                    x_b = x
                    if self.config.use_time_weights:
                        x_b = self.apply_time_weights(x_b, client_id)
                    x_t = torch.zeros(x_b.size(0), 4, self.config.input_dim_tsinghua).to(self.device)
                    pred = model(x_b, x_t)
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
    for node in nodes:
        train_path = data_dir / f"node_{node}" / "train.pkl"
        dataset = Barcelona7DayDataset(train_path)
        clients[f"barcelona_{node}"] = dataset
    test_clients = {}
    for node in nodes:
        test_path = data_dir / f"node_{node}" / "test.pkl"
        dataset = Barcelona7DayDataset(test_path)
        test_clients[f"barcelona_{node}"] = dataset
    return clients, test_clients

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
    parser.add_argument('--use_time_weights', action='store_true', default=True, help='是否使用时段加权')
    args = parser.parse_args()

    config = Config()
    config.barcelona_nodes = [int(n) for n in args.barcelona_nodes.split(',')]
    config.tsinghua_clusters_file = PROJECT_ROOT / args.tsinghua_clusters
    config.rounds = args.rounds
    config.mu = args.mu
    config.use_time_weights = args.use_time_weights

    all_weights = load_all_segment_weights(config.weights_file) if config.use_time_weights else None

    print("加载巴塞罗那节点...")
    barcelona_clients, test_clients = load_barcelona_clients(config.barcelona_nodes, config.barcelona_data_dir)
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

    trainer = FederatedTrainer(config, all_weights)
    smape = trainer.train_full(config.mu, client_loaders, test_loaders)
    print(f"\n最终 sMAPE: {smape:.2f}%")


if __name__ == "__main__":
    main()
