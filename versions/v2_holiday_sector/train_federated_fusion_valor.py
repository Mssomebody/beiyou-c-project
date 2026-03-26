#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粒度融合：巴塞罗那节点 + 清华5G聚类组（6小时粒度）
使用原始能耗 Valor，MinMax归一化
"""

import sys
import os
import json
import logging
import argparse
import optuna
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# 加载全局配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_global_config():
    config_path = PROJECT_ROOT / "versions" / \
        "v2_holiday_sector" / "configs" / "paths.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


GLOBAL_CONFIG = load_global_config()
DATA_ROOT = Path(GLOBAL_CONFIG['data_root'])
BARCE_DATA_VERSION = GLOBAL_CONFIG['current']['barcelona']
BARCE_DATA_PATH = DATA_ROOT / GLOBAL_CONFIG['barcelona'][BARCE_DATA_VERSION]

# ============================================================
# 联邦学习配置
# ============================================================


@dataclass
class FedConfig:
    data_path: Path = BARCE_DATA_PATH
    barcelona_nodes: List[int] = field(
        default_factory=lambda: [
            8001,
            8002,
            8003,
            8004,
            8005,
            8006,
            8007,
            8008,
            8009,
            8010,
            8011,
            8012,
            8013,
            8014,
            8015,
            8016,
            8017,
            8018,
            8019,
            8020,
            8021,
            8022,
            8023,
            8024,
            8026,
            8027,
            8028,
            8029,
            8030,
            8031,
            8032,
            8033,
            8034,
            8035,
            8036,
            8037,
            8038,
            8039,
            8040,
            8041,
            8042])
    tsinghua_clusters_file: Path = PROJECT_ROOT / \
        "data/processed/tsinghua_6h/5g_clusters.json"
    window_size: int = 28
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
    mu_min: float = 0.0
    mu_max: float = 0.1
    n_trials: int = 1
    seed: int = 42
    device: str = 'cpu'
    output_dir: Path = None
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "results" / "fed_optuna"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


# ============================================================
# 日志系统（复用）
# ============================================================
class Logger:
    def __init__(self, config: FedConfig, name: str = "fed_fusion"):
        self.config = config
        self.name = name
        self.logger = self._setup()

    def _setup(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        if logger.handlers:
            return logger
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console)
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        return logger

    def info(self, msg: str):
        self.logger.info(msg)
        sys.stdout.flush()


# ============================================================
# 巴塞罗那数据加载（与原脚本一致）
# ============================================================
class BarcelonaCoarseDataset(Dataset):
    def __init__(self, data_path, norm_params=None):
        """
        norm_params: 训练集计算得到的归一化参数，用于验证/测试集。若为 None，则从当前数据计算。
        """
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
        df = df.sort_values('Data')
        dates = pd.to_datetime(df['Data']).unique()
        self.samples = []   # 每个元素为 (x, y)，x: (4,7), y: (4,)
        print(f"Loaded {len(self.samples)} samples from {data_path}")
        # 遍历所有连续两天，提取完整时段样本
        for i in range(len(dates) - 1):
            prev_date = dates[i]
            last_date = dates[i + 1]
            # 检查两天是否都有完整4个时段

            def has_full_hours(date):
                day_df = df[pd.to_datetime(df['Data']) == date]
                return set(day_df['hour_code'].unique()) == {0, 1, 2, 3}
            if has_full_hours(prev_date) and has_full_hours(last_date):
                input_df = df[pd.to_datetime(df['Data']) == prev_date]
                target_df = df[pd.to_datetime(df['Data']) == last_date]

                # 聚合函数：将同一时段各部门数据合并
                def aggregate_group(group):
                    total_energy = group['Valor'].sum()
                    weekend = group['is_weekend'].iloc[0]
                    holiday = group['is_holiday'].iloc[0]
                    sector_means = group[[
                        'sector_0', 'sector_1', 'sector_2', 'sector_3']].mean().values
                    return pd.Series({
                        'total_energy': total_energy,
                        'is_weekend': weekend,
                        'is_holiday': holiday,
                        'sector_0': sector_means[0],
                        'sector_1': sector_means[1],
                        'sector_2': sector_means[2],
                        'sector_3': sector_means[3]
                    })

                input_agg = input_df.groupby('hour_code').apply(
                    aggregate_group).reset_index()
                target_agg = target_df.groupby('hour_code').apply(
                    aggregate_group).reset_index()
                # 确保四个时段都存在（用0填充缺失）
                input_agg = input_agg.set_index('hour_code').reindex(
                    [0, 1, 2, 3], fill_value=0).reset_index()
                target_agg = target_agg.set_index('hour_code').reindex(
                    [0, 1, 2, 3], fill_value=0).reset_index()
                input_agg = input_agg.sort_values('hour_code')
                target_agg = target_agg.sort_values('hour_code')

                # 特征：总能耗 + 部门占比 + 周末 + 节假日 (7维)
                x = input_agg[['total_energy',
                               'sector_0',
                               'sector_1',
                               'sector_2',
                               'sector_3',
                               'is_weekend',
                               'is_holiday']].values.astype(np.float32)
                y = target_agg['total_energy'].values.astype(np.float32)
                self.samples.append((x, y))

        if not self.samples:
            raise ValueError(f"数据 {data_path} 中没有足够连续两天完整时段")

        # 归一化：对训练集计算 min/max，对验证/测试集使用传入参数
        if norm_params is None:
            # 训练集：计算所有样本的 min/max
            all_x = np.stack([s[0] for s in self.samples])  # (n_samples, 4, 7)
            all_y = np.stack([s[1] for s in self.samples])  # (n_samples, 4)
            self.x_min = all_x.min(axis=(0, 1), keepdims=True)  # (1,1,7)
            self.x_max = all_x.max(axis=(0, 1), keepdims=True)
            self.y_min = all_y.min()
            self.y_max = all_y.max()
            self.norm_params = {
                'x_min': self.x_min.squeeze(),
                'x_max': self.x_max.squeeze(),
                'y_min': self.y_min,
                'y_max': self.y_max
            }
            # 归一化样本
            self.x_norm = (all_x - self.x_min) / \
                (self.x_max - self.x_min + 1e-8)
            self.y_norm = (all_y - self.y_min) / \
                (self.y_max - self.y_min + 1e-8)
        else:
            # 验证/测试集：使用训练集参数
            self.x_min = norm_params['x_min'][None, None, :]
            self.x_max = norm_params['x_max'][None, None, :]
            self.y_min = norm_params['y_min']
            self.y_max = norm_params['y_max']
            all_x = np.stack([s[0] for s in self.samples])
            all_y = np.stack([s[1] for s in self.samples])
            self.x_norm = (all_x - self.x_min) / \
                (self.x_max - self.x_min + 1e-8)
            self.y_norm = (all_y - self.y_min) / \
                (self.y_max - self.y_min + 1e-8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(
            self.x_norm[idx]), torch.FloatTensor(
            self.y_norm[idx])


# ============================================================
# 清华聚类组数据加载（6小时粒度，生成多个样本）
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
            # data['features'] 形状 (4,5)，data['target'] 形状 (4,1)
            self.features.append(data['features'])
            self.targets.append(data['target'].squeeze())
        if not self.features:
            raise ValueError(f"No valid station data for cluster")
        # 取组内平均
        self.features = np.mean(self.features, axis=0)   # (4,5)
        self.targets = np.mean(self.targets, axis=0)     # (4,)

    def __len__(self):
        return 1   # 每个聚类组只有一个样本（平均后的4个时段）

    def __getitem__(self, idx):
        return torch.FloatTensor(
            self.features), torch.FloatTensor(
            self.targets)


# ============================================================
# 模型：LSTM + 投影层
# ============================================================
class FusionLSTM(nn.Module):
    def __init__(
            self,
            input_dim_barcelona,
            input_dim_tsinghua,
            hidden_dim,
            num_layers,
            output_dim,
            dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim_barcelona, input_dim_tsinghua)
        self.lstm = nn.LSTM(
            input_dim_tsinghua,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, is_barcelona):
        if is_barcelona:
            x = self.proj(x)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 数据加载工厂
# ============================================================
class DataLoaderFactory:
    @staticmethod
    def get_barcelona_loader(
            node_id,
            split,
            batch_size,
            shuffle,
            config,
            norm_params=None):
        data_dir = config.data_path / f"node_{node_id}"
        file_path = data_dir / f"{split}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if split == 'train':
            dataset = BarcelonaDatasetMinMax(
                data_path=str(file_path),
                window_size=config.window_size,
                predict_size=config.predict_size,
                sector_feature=True,
                holiday_feature=True,
                weekend_feature=True,
                norm_params=None
            )
            norm_params = dataset.norm_params
        else:
            dataset = BarcelonaDatasetMinMax(
                data_path=str(file_path),
                window_size=config.window_size,
                predict_size=config.predict_size,
                sector_feature=True,
                holiday_feature=True,
                weekend_feature=True,
                norm_params=norm_params
            )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True)
        return loader, norm_params

    @staticmethod
    def get_tsinghua_loader(cluster_stations, data_dir, batch_size, shuffle):
        dataset = TsinghuaClusterDataset(cluster_stations, data_dir)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True)
        return loader

    @classmethod
    def load_all_loaders(cls, config: FedConfig):
        # 巴塞罗那节点
        barcelona_train = {}
        barcelona_val = {}
        barcelona_test = {}
        norm_params_dict = {}
        for node in config.barcelona_nodes:
            train_loader, norm_params = cls.get_barcelona_loader(
                node, 'train', config.batch_size, True, config)
            barcelona_train[node] = train_loader
            norm_params_dict[node] = norm_params

        for node in config.barcelona_nodes:
            val_loader, _ = cls.get_barcelona_loader(
                node, 'val', config.batch_size, False, config, norm_params_dict[node])
            test_loader, _ = cls.get_barcelona_loader(
                node, 'test', config.batch_size, False, config, norm_params_dict[node])
            barcelona_val[node] = val_loader
            barcelona_test[node] = test_loader

        # 清华聚类组
        with open(config.tsinghua_clusters_file, 'r') as f:
            clusters = json.load(f)
        tech = list(clusters.keys())[0]  # 假设是 '5g'
        tsinghua_clients = {}
        for cluster_id, station_ids in clusters[tech].items():
            loader = cls.get_tsinghua_loader(
                station_ids, config.data_path / tech, config.batch_size, False)
            tsinghua_clients[f"tsinghua_{cluster_id}"] = loader

        return barcelona_train, barcelona_val, barcelona_test, tsinghua_clients


# ============================================================
# 联邦训练器（支持两种客户端）
# ============================================================
class FederatedTrainer:
    def __init__(self, config: FedConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.logger.info(f"设备: {self.device}")

    def _create_model(self):
        return FusionLSTM(
            input_dim_barcelona=self.config.input_dim_barcelona,
            input_dim_tsinghua=self.config.input_dim_tsinghua,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            output_dim=self.config.predict_size,
            dropout=self.config.dropout
        )

    def _create_optimizer_and_scheduler(self, model):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            min_lr=self.config.lr_scheduler_min_lr,
        )
        return optimizer, scheduler

    def train_round(self,
                    model: nn.Module,
                    client_loaders: Dict[str,
                                         DataLoader],
                    mu: float) -> nn.Module:
        client_weights = []
        client_sizes = []
        client_types = []  # 'barcelona' or 'tsinghua'

        for client_id, loader in client_loaders.items():
            local_model = self._create_model().to(self.device)
            local_model.load_state_dict(model.state_dict())
            optimizer, scheduler = self._create_optimizer_and_scheduler(
                local_model)
            criterion = nn.MSELoss()

            best_local_loss = float('inf')
            patience_counter = 0
            is_barcelona = client_id.startswith('barcelona')

            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(x, is_barcelona)
                    loss = criterion(output, y)
                    if mu > 0:
                        prox_loss = 0.0
                        for param, global_param in zip(
                                local_model.parameters(), model.parameters()):
                            prox_loss += torch.norm(param - global_param) ** 2
                        loss += (mu / 2) * prox_loss
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(loader)
                scheduler.step(avg_loss)

                if avg_loss < best_local_loss - self.config.early_stop_min_delta:
                    best_local_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stop_patience:
                    break

            client_weights.append(local_model.state_dict())
            client_sizes.append(len(loader.dataset))
            client_types.append(is_barcelona)

        # 聚合（均匀加权，按数据量）
        total = sum(client_sizes)
        global_weights = {}
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key])
            for w, size in zip(client_weights, client_sizes):
                global_weights[key] += w[key] * (size / total)

        model.load_state_dict(global_weights)
        return model

    def evaluate_loss(self,
                      model: nn.Module,
                      loaders: Dict[str,
                                    DataLoader]) -> float:
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for loader in loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    is_barcelona = True  # 评估只用巴塞罗那节点
                    output = model(x, is_barcelona)
                    loss = criterion(output, y)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)

        return total_loss / total_samples

    def evaluate_smape(self,
                       model: nn.Module,
                       test_loaders: Dict[str,
                                          DataLoader]) -> float:
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for loader in test_loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    is_barcelona = True
                    output = model(x, is_barcelona)
                    all_preds.append(output.cpu().numpy())
                    all_targets.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        denominator = (np.abs(all_targets) + np.abs(all_preds)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(np.abs(all_targets - all_preds) / denominator) * 100

        return smape

    def train_full(self, mu: float, barcelona_train, barcelona_val,
                   barcelona_test, tsinghua_clients) -> Tuple[float, float]:
        # 合并所有客户端
        all_clients = {**{f"barcelona_{node}": loader for node,
                          loader in barcelona_train.items()}, **tsinghua_clients}
        val_clients = {
            f"barcelona_{node}": loader for node,
            loader in barcelona_val.items()}
        test_clients = {
            f"barcelona_{node}": loader for node,
            loader in barcelona_test.items()}

        model = self._create_model().to(self.device)
        optimizer, scheduler = self._create_optimizer_and_scheduler(model)

        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        best_round = 0

        for round_num in range(1, self.config.rounds + 1):
            model = self.train_round(model, all_clients, mu)
            val_loss = self.evaluate_loss(model, val_clients)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - self.config.early_stop_min_delta:
                best_val_loss = val_loss
                best_model = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                best_round = round_num
                patience_counter = 0
                self.logger.info(
                    f"  Round {round_num}: val_loss={
                        val_loss:.6f} (new best)")
            else:
                patience_counter += 1
                self.logger.info(
                    f"  Round {round_num}: val_loss={
                        val_loss:.6f} (best: {
                        best_val_loss:.6f} @ round {best_round})")

            if patience_counter >= self.config.early_stop_patience:
                self.logger.info(f"  早停触发！连续 {patience_counter} 轮无改善")
                break

        if best_model:
            model.load_state_dict(best_model)
            self.logger.info(f"使用最佳模型 (round {best_round})")

        test_smape = self.evaluate_smape(model, test_clients)
        return best_val_loss, test_smape


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--barcelona_nodes', type=str, default=None)
    parser.add_argument(
        '--tsinghua_clusters',
        type=str,
        default='data/processed/tsinghua_6h/5g_clusters.json')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--mu', type=float, default=0.05)
    args = parser.parse_args()

    config = FedConfig()
    if args.barcelona_nodes:
        config.barcelona_nodes = [int(n)
                                  for n in args.barcelona_nodes.split(',')]
    config.rounds = args.rounds
    config.mu = args.mu
    config.tsinghua_clusters_file = PROJECT_ROOT / args.tsinghua_clusters

    logger = Logger(config)

    logger.info("=" * 70)
    logger.info("粒度融合：巴塞罗那 + 清华5G聚类组（6小时粒度）")
    logger.info("=" * 70)
    logger.info(f"巴塞罗那节点数: {len(config.barcelona_nodes)}")
    logger.info(f"清华聚类组文件: {config.tsinghua_clusters_file}")
    logger.info(f"μ: {config.mu}, 轮数: {config.rounds}")
    logger.info("=" * 70)

    # 加载数据
    barcelona_train, barcelona_val, barcelona_test, tsinghua_clients = DataLoaderFactory.load_all_loaders(
        config)

    # 训练
    trainer = FederatedTrainer(config, logger)
    best_val_loss, test_smape = trainer.train_full(
        config.mu, barcelona_train, barcelona_val, barcelona_test, tsinghua_clients)

    logger.info("\n" + "=" * 70)
    logger.info("训练完成")
    logger.info("=" * 70)
    logger.info(f"最终 sMAPE: {test_smape:.2f}%")

    # 保存结果
    result_file = config.output_dir / \
        f"fusion_mu_{config.mu:.6f}_smape_{test_smape:.2f}.txt"
    with open(result_file, 'w') as f:
        f.write(
            f"mu={
                config.mu}\nsmape={
                test_smape:.2f}\nrounds={
                config.rounds}\nbarcelona_nodes={
                    config.barcelona_nodes}\ntsinghua_file={
                        config.tsinghua_clusters_file}")
    logger.info(f"结果保存: {result_file}")

    return test_smape


if __name__ == "__main__":
    main()
