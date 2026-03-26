import yaml
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加权联邦学习（基于原始能耗 Valor，MinMax归一化）
在聚合时使用节点权重（全局平均权重）
"""

import sys
import os
import json
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# 加载全局配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_global_config():
    config_path = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "configs" / "paths.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

GLOBAL_CONFIG = load_global_config()
DATA_ROOT = Path(GLOBAL_CONFIG['data_root'])
BARCE_DATA_VERSION = GLOBAL_CONFIG['current']['barcelona']
BARCE_DATA_PATH = DATA_ROOT / GLOBAL_CONFIG['barcelona'][BARCE_DATA_VERSION]

# ============================================================
# 加载节点权重（全局平均）
# ============================================================
WEIGHTS_PATH = PROJECT_ROOT / "results" / "barcelona_clustering" / "node_weights_for_federated.json"
NODE_WEIGHTS = None
if WEIGHTS_PATH.exists():
    try:
        with open(WEIGHTS_PATH, 'r') as f:
            node_weights_raw = json.load(f)
        NODE_WEIGHTS = {}
        for node, seg_weights in node_weights_raw.items():
            avg_weight = sum(seg_weights.values()) / len(seg_weights)
            NODE_WEIGHTS[node] = avg_weight
        print(f"✅ 已加载节点权重，共 {len(NODE_WEIGHTS)} 个节点")
    except Exception as e:
        print(f"⚠️ 加载节点权重失败: {e}，将使用均匀权重")
        NODE_WEIGHTS = None
else:
    print(f"⚠️ 节点权重文件不存在: {WEIGHTS_PATH}，将使用均匀权重")

# ============================================================
# 联邦学习配置（同原脚本）
# ============================================================
@dataclass
class FedConfig:
    data_path: Path = BARCE_DATA_PATH
    nodes: List[int] = field(default_factory=lambda: list(range(8001, 8025)))
    window_size: int = 28
    predict_size: int = 4
    batch_size: int = 64
    input_dim: int = 7
    hidden_dim: int = 160
    num_layers: int = 4
    dropout: float = 0.1565
    learning_rate: float = 0.00572
    rounds: int = 50
    local_epochs: int = 5
    mu_min: float = 0.0
    mu_max: float = 0.1
    n_trials: int = 15
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
# 日志系统（同原脚本）
# ============================================================
class Logger:
    def __init__(self, config: FedConfig, name: str = "fed_optuna"):
        self.config = config
        self.name = name
        self.logger = self._setup()

    def _setup(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        if logger.handlers:
            return logger
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console)
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        return logger

    def info(self, msg: str):
        self.logger.info(msg)
        sys.stdout.flush()


# ============================================================
# 数据集（与原脚本完全相同）
# ============================================================
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

        self.energy = self.df['Valor'].values.astype(np.float32)

        if norm_params is None:
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
# 数据加载器
# ============================================================
class DataLoaderFactory:
    @staticmethod
    def get_node_loader(node_id: int, split: str, batch_size: int, shuffle: bool, config: FedConfig,
                        norm_params_dict: Dict = None) -> Tuple[DataLoader, Dict]:
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
            if norm_params_dict is None:
                raise ValueError("验证/测试集需要提供 norm_params_dict")
            dataset = BarcelonaDatasetMinMax(
                data_path=str(file_path),
                window_size=config.window_size,
                predict_size=config.predict_size,
                sector_feature=True,
                holiday_feature=True,
                weekend_feature=True,
                norm_params=norm_params_dict
            )
            norm_params = norm_params_dict

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        return loader, norm_params

    @classmethod
    def load_all_loaders(cls, config: FedConfig):
        train_loaders = {}
        val_loaders = {}
        test_loaders = {}
        norm_params_dict = {}

        for node in config.nodes:
            train_loader, norm_params = cls.get_node_loader(node, 'train', config.batch_size, True, config)
            train_loaders[node] = train_loader
            norm_params_dict[node] = norm_params

        for node in config.nodes:
            val_loader, _ = cls.get_node_loader(node, 'val', config.batch_size, False, config, norm_params_dict[node])
            test_loader, _ = cls.get_node_loader(node, 'test', config.batch_size, False, config, norm_params_dict[node])
            val_loaders[node] = val_loader
            test_loaders[node] = test_loader

        return train_loaders, val_loaders, test_loaders


# ============================================================
# 模型定义
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 联邦训练器（带加权聚合）
# ============================================================
class FederatedTrainer:
    def __init__(self, config: FedConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.node_weights = NODE_WEIGHTS
        if self.node_weights:
            self.logger.info(f"使用节点权重，共 {len(self.node_weights)} 个节点")
        else:
            self.logger.info("未加载节点权重，使用均匀聚合")
        self.logger.info(f"设备: {self.device}")

    def _create_model(self):
        return LSTMPredictor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            output_dim=self.config.predict_size,
            dropout=self.config.dropout
        )

    def _create_optimizer_and_scheduler(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            min_lr=self.config.lr_scheduler_min_lr,
        )
        return optimizer, scheduler

    def train_round(self, model: nn.Module, train_loaders: Dict[int, DataLoader], mu: float) -> nn.Module:
        client_weights = []
        client_sizes = []
        client_ids = []

        for client_id, loader in train_loaders.items():
            client_ids.append(client_id)
            local_model = self._create_model().to(self.device)
            local_model.load_state_dict(model.state_dict())

            optimizer, scheduler = self._create_optimizer_and_scheduler(local_model)
            criterion = nn.MSELoss()

            best_local_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(x)
                    loss = criterion(output, y)

                    if mu > 0:
                        prox_loss = 0.0
                        for param, global_param in zip(local_model.parameters(), model.parameters()):
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

        # 加权聚合
        total = sum(client_sizes)
        if self.node_weights is not None:
            total_weighted = sum(self.node_weights.get(str(cid), 1.0) * size for cid, size in zip(client_ids, client_sizes))
            global_weights = {}
            for key in client_weights[0].keys():
                global_weights[key] = torch.zeros_like(client_weights[0][key])
                for w, cid, size in zip(client_weights, client_ids, client_sizes):
                    weight = self.node_weights.get(str(cid), 1.0) * size / total_weighted
                    global_weights[key] += w[key] * weight
        else:
            global_weights = {}
            for key in client_weights[0].keys():
                global_weights[key] = torch.zeros_like(client_weights[0][key])
                for w, size in zip(client_weights, client_sizes):
                    global_weights[key] += w[key] * (size / total)

        model.load_state_dict(global_weights)
        return model

    def evaluate_loss(self, model: nn.Module, loaders: Dict[int, DataLoader]) -> float:
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for loader in loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    loss = criterion(output, y)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)

        return total_loss / total_samples

    def evaluate_smape(self, model: nn.Module, test_loaders: Dict[int, DataLoader]) -> float:
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for loader in test_loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    all_preds.append(output.cpu().numpy())
                    all_targets.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        denominator = (np.abs(all_targets) + np.abs(all_preds)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(np.abs(all_targets - all_preds) / denominator) * 100

        return smape

    def train_full(self, mu: float, train_loaders: Dict, val_loaders: Dict, test_loaders: Dict) -> Tuple[float, float]:
        model = self._create_model().to(self.device)
        optimizer, scheduler = self._create_optimizer_and_scheduler(model)

        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        best_round = 0

        for round_num in range(1, self.config.rounds + 1):
            model = self.train_round(model, train_loaders, mu)
            val_loss = self.evaluate_loss(model, val_loaders)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - self.config.early_stop_min_delta:
                best_val_loss = val_loss
                best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_round = round_num
                patience_counter = 0
                self.logger.info(f"  Round {round_num}: val_loss={val_loss:.6f} (new best)")
            else:
                patience_counter += 1
                self.logger.info(f"  Round {round_num}: val_loss={val_loss:.6f} (best: {best_val_loss:.6f} @ round {best_round})")

            if patience_counter >= self.config.early_stop_patience:
                self.logger.info(f"  早停触发！连续 {patience_counter} 轮无改善")
                break

        if best_model:
            model.load_state_dict(best_model)
            self.logger.info(f"使用最佳模型 (round {best_round})")

        test_smape = self.evaluate_smape(model, test_loaders)
        return best_val_loss, test_smape


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=str, default=None, help='节点列表，如: 8001,8005,8013')
    parser.add_argument('--nodes_start', type=int, default=8001)
    parser.add_argument('--nodes_end', type=int, default=8025)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--lr_scheduler_patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mu', type=float, default=None, help='固定 μ 值，若不提供则贝叶斯搜索')
    args = parser.parse_args()

    config = FedConfig()
    if args.nodes:
        config.nodes = [int(n.strip()) for n in args.nodes.split(',')]
    else:
        config.nodes = list(range(args.nodes_start, args.nodes_end))
    config.n_trials = args.n_trials
    config.rounds = args.rounds
    config.early_stop_patience = args.early_stop_patience
    config.lr_scheduler_patience = args.lr_scheduler_patience
    config.seed = args.seed

    logger = Logger(config)

    logger.info("="*70)
    logger.info("联邦学习 + 贝叶斯自动找最优 μ（加权版，MinMax归一化）")
    logger.info("="*70)
    logger.info(f"数据版本: {BARCE_DATA_VERSION}")
    logger.info(f"节点: {config.nodes[0]} - {config.nodes[-1]} ({len(config.nodes)} 个)")
    logger.info(f"μ 范围: [{config.mu_min}, {config.mu_max}]")
    logger.info(f"试验次数: {config.n_trials}")
    logger.info(f"最大轮数: {config.rounds}")
    logger.info(f"早停耐心值: {config.early_stop_patience}")
    logger.info(f"学习率衰减: factor={config.lr_scheduler_factor}, patience={config.lr_scheduler_patience}")
    logger.info(f"使用节点权重: {NODE_WEIGHTS is not None}")
    logger.info("="*70)
    sys.stdout.flush()

    if args.mu is not None:
        logger.info(f"固定 μ = {args.mu}，跳过贝叶斯优化")
        train_loaders, val_loaders, test_loaders = DataLoaderFactory.load_all_loaders(config)
        trainer = FederatedTrainer(config, logger)
        best_val_loss, best_smape = trainer.train_full(args.mu, train_loaders, val_loaders, test_loaders)
        logger.info("\n" + "="*70)
        logger.info("训练完成（固定 μ）")
        logger.info("="*70)
        logger.info(f"最终 sMAPE: {best_smape:.2f}%")
        result_file = config.output_dir / f"weighted_fixed_mu_{args.mu:.6f}_smape_{best_smape:.2f}.txt"
        with open(result_file, 'w') as f:
            f.write(f"mu={args.mu}\nsmape={best_smape:.2f}\nrounds={config.rounds}\nnodes={config.nodes}")
        logger.info(f"结果保存: {result_file}")
        return args.mu

    # 贝叶斯优化
    import optuna
    study_name = f"fed_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{config.output_dir / study_name}.db"

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    def objective(trial):
        mu = trial.suggest_float('mu', config.mu_min, config.mu_max)
        logger.info(f"Trial {trial.number}: μ={mu:.6f}")
        train_loaders, val_loaders, test_loaders = DataLoaderFactory.load_all_loaders(config)
        trainer = FederatedTrainer(config, logger)
        val_loss, test_smape = trainer.train_full(mu, train_loaders, val_loaders, test_loaders)
        logger.info(f"  Test sMAPE: {test_smape:.2f}%")
        return test_smape

    for i in range(config.n_trials):
        logger.info(f"开始 Trial {i+1}/{config.n_trials}")
        sys.stdout.flush()

        def wrapped_objective(trial):
            trial._number = i
            return objective(trial)

        study.optimize(wrapped_objective, n_trials=1)

    saver = ResultSaver(config, logger)
    saver.save(study.best_params['mu'], study.best_value, study)

    logger.info("\n" + "="*70)
    logger.info("优化完成")
    logger.info("="*70)
    logger.info(f"最优 μ: {study.best_params['mu']:.6f}")
    logger.info(f"最优 sMAPE: {study.best_value:.2f}%")
    return study.best_params['mu']


if __name__ == "__main__":
    best_mu = main()
    print(f"\n✅ 最优 μ = {best_mu:.6f}")
