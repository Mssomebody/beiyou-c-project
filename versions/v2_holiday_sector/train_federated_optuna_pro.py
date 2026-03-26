#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级联邦学习 + 贝叶斯自动找最优 μ
- 支持多数据版本
- 配置管理 (@dataclass)
- 完整日志
- 结果保存
"""

import sys
import os
import json
import logging
import argparse
import optuna
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor, train_epoch, evaluate
from torch.utils.data import DataLoader


# ============================================================
# 配置管理
# ============================================================

@dataclass
class Config:
    """集中配置管理"""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_version: str = "barcelona_ready_2019_2022"
    output_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    
    nodes: List[int] = field(default_factory=lambda: list(range(8001, 8025)))
    window_size: int = 28
    predict_size: int = 4
    input_dim: int = 7
    hidden_dim: int = 160
    num_layers: int = 4
    dropout: float = 0.1565
    lr: float = 0.00572
    batch_size: int = 64
    
    rounds: int = 20
    local_epochs: int = 5
    mu_min: float = 0.0
    mu_max: float = 0.1
    n_trials: int = 15
    
    seed: int = 42
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.project_root / "results" / "fed_optuna"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


# ============================================================
# 日志系统
# ============================================================

def setup_logger(config: Config) -> logging.Logger:
    logger = logging.getLogger("fed_optuna")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.logs_dir / f"fed_optuna_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


# ============================================================
# 数据加载器（支持多数据版本）
# ============================================================

def get_node_data_loader_v2(node_id, split, batch_size, shuffle,
                             window_size, predict_size,
                             sector_feature, holiday_feature, weekend_feature,
                             data_version):
    """支持多数据版本的数据加载器"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "processed", data_version, f"node_{node_id}")
    
    file_path = os.path.join(data_dir, f"{split}.pkl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    dataset = BarcelonaDataset(
        data_path=file_path,
        window_size=window_size,
        predict_size=predict_size,
        sector_feature=sector_feature,
        holiday_feature=holiday_feature,
        weekend_feature=weekend_feature
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader


def load_all_loaders(config: Config):
    """加载所有节点的数据加载器"""
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    
    for node in config.nodes:
        train_loaders[node] = get_node_data_loader_v2(
            node, 'train', config.batch_size, True,
            config.window_size, config.predict_size,
            True, True, True, config.data_version
        )
        val_loaders[node] = get_node_data_loader_v2(
            node, 'val', config.batch_size, False,
            config.window_size, config.predict_size,
            True, True, True, config.data_version
        )
        test_loaders[node] = get_node_data_loader_v2(
            node, 'test', config.batch_size, False,
            config.window_size, config.predict_size,
            True, True, True, config.data_version
        )
    
    return train_loaders, val_loaders, test_loaders


# ============================================================
# 联邦训练器
# ============================================================

class FederatedTrainer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"设备: {self.device}")
    
    def train_round(self, model, train_loaders, mu, val_loaders=None):
        client_weights = []
        client_sizes = []
        
        for client_id, loader in train_loaders.items():
            local_model = LSTMPredictor(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                output_dim=self.config.predict_size,
                dropout=self.config.dropout
            ).to(self.device)
            local_model.load_state_dict(model.state_dict())
            
            optimizer = torch.optim.Adam(local_model.parameters(), lr=self.config.lr)
            criterion = nn.MSELoss()
            
            for _ in range(self.config.local_epochs):
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
            
            client_weights.append(local_model.state_dict())
            client_sizes.append(len(loader.dataset))
        
        # 聚合
        total_samples = sum(client_sizes)
        global_weights = {}
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key])
            for w, size in zip(client_weights, client_sizes):
                global_weights[key] += w[key] * (size / total_samples)
        
        model.load_state_dict(global_weights)
        return model
    
    def evaluate(self, model, test_loaders):
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for loader in test_loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    loss = criterion(output, y)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)
        
        return total_loss / total_samples


# ============================================================
# 贝叶斯目标函数
# ============================================================

def objective(trial, config: Config, logger: logging.Logger) -> float:
    mu = trial.suggest_float('mu', config.mu_min, config.mu_max, log=True)
    logger.info(f"Trial {trial.number}: μ={mu:.6f}")
    
    train_loaders, val_loaders, test_loaders = load_all_loaders(config)
    
    model = LSTMPredictor(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_dim=config.predict_size,
        dropout=config.dropout
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    trainer = FederatedTrainer(config, logger)
    
    best_val_loss = float('inf')
    best_model = None
    
    for round_num in range(1, config.rounds + 1):
        model = trainer.train_round(model, train_loaders, mu)
        val_loss = trainer.evaluate(model, val_loaders)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if round_num % 5 == 0:
            logger.info(f"  Round {round_num}: val_loss={val_loss:.6f}")
    
    if best_model:
        model.load_state_dict(best_model)
    
    test_loss = trainer.evaluate(model, test_loaders)
    logger.info(f"  Test loss: {test_loss:.6f}")
    
    return test_loss


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', type=str, default='barcelona_ready_2019_2022')
    parser.add_argument('--nodes_start', type=int, default=8001)
    parser.add_argument('--nodes_end', type=int, default=8025)
    parser.add_argument('--n_trials', type=int, default=15)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    config = Config(
        data_version=args.data_version,
        nodes=list(range(args.nodes_start, args.nodes_end)),
        n_trials=args.n_trials,
        rounds=args.rounds,
        seed=args.seed
    )
    
    logger = setup_logger(config)
    
    logger.info("="*70)
    logger.info("联邦学习 + 贝叶斯自动找最优 μ")
    logger.info("="*70)
    logger.info(f"数据: {config.data_version}")
    logger.info(f"节点: {config.nodes[0]} - {config.nodes[-1]} ({len(config.nodes)} 个)")
    logger.info(f"μ 范围: [{config.mu_min}, {config.mu_max}]")
    logger.info(f"试验次数: {config.n_trials}")
    logger.info("="*70)
    
    study = optuna.create_study(direction='minimize')
    
    study.optimize(
        lambda trial: objective(trial, config, logger),
        n_trials=config.n_trials,
        show_progress_bar=True
    )
    
    logger.info("\n" + "="*70)
    logger.info("优化完成")
    logger.info("="*70)
    logger.info(f"最优 μ: {study.best_params['mu']:.6f}")
    logger.info(f"最优测试损失: {study.best_value:.6f}")
    
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'trials': [{'number': t.number, 'value': t.value, 'params': t.params}
                   for t in study.trials if t.value is not None]
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = config.output_dir / f"optuna_results_{timestamp}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"结果保存: {result_path}")
    
    return study.best_params['mu']


if __name__ == "__main__":
    best_mu = main()
    print(f"\n✅ 最优 μ = {best_mu:.6f}")
