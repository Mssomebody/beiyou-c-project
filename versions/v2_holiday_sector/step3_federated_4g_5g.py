#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4G+5G代际协同联邦学习 - 五星级专业版

问题：4G和5G基站数据不能共享（不同代际、不同厂商、不同运维团队）
方案：联邦学习让4G和5G在数据隔离的前提下协同训练
效果：5G数据少，从4G学到成熟模式；4G从5G学到节能特性

作者: FedGreen-C
版本: 1.0
"""

import os
import sys
import pickle
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# 配置管理
# ============================================================

@dataclass
class Config:
    """集中配置管理"""
    
    # 路径配置
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data" / "processed" / "tsinghua"
    results_dir: Path = project_root / "results" / "federated_4g_5g"
    logs_dir: Path = project_root / "logs"
    
    # 数据集配置
    datasets: Dict = None
    
    # 模型配置
    input_dim: int = 5          # 4G特征维度
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.2
    
    # 训练配置
    batch_size: int = 64
    local_epochs: int = 5
    num_rounds: int = 10
    learning_rate: float = 0.001
    seq_len: int = 24           # 输入序列长度（24个时间点）
    pred_len: int = 1           # 预测长度（1个点）
    
    # 联邦配置
    fed_algorithm: str = 'fedavg'  # 'fedavg' or 'fedprox'
    mu: float = 0.01            # FedProx 近端项系数
    
    # 其他
    seed: int = 42
    log_level: str = 'INFO'
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = {
                '4g': {'dir': '4g', 'type': '4g', 'n_stations': 12162, 'n_samples': 432},
                '5g': {'dir': '5g', 'type': '5g', 'n_stations': 5165, 'n_samples': 144},
            }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 日志系统
# ============================================================

def setup_logger(name: str = __name__, level: str = 'INFO') -> logging.Logger:
    """配置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console)
    
    return logger


# ============================================================
# 数据集
# ============================================================

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, data_path: Path, seq_len: int = 24, pred_len: int = 1):
        """
        Args:
            data_path: 数据文件路径
            seq_len: 输入序列长度
            pred_len: 预测长度
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.features = data['features']  # [n_samples, n_features]
        self.target = data['target']       # [n_samples, 1]
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx:idx + self.seq_len]
        y = self.target[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y.flatten())


class LSTMPredictor(nn.Module):
    """LSTM预测模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)
        return out


# ============================================================
# 联邦客户端
# ============================================================

class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: str, data_dir: Path, config: Config, logger: logging.Logger):
        self.client_id = client_id
        self.data_dir = data_dir
        self.config = config
        self.logger = logger
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.n_samples = 0
        
    def load_data(self):
        """加载数据"""
        # 获取所有基站
        station_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
        self.logger.info(f"  {self.client_id}: 加载 {len(station_dirs)} 个基站")
        
        # 取前100个基站作为代表（完整训练会太慢）
        sample_stations = station_dirs[:100]
        
        # 构建数据集
        all_features = []
        all_targets = []
        
        for station_dir in tqdm(sample_stations, desc=f"  {self.client_id} 加载", leave=False):
            data_path = station_dir / 'data.pkl'
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            features = data['features']
            target = data['target']
            all_features.append(features)
            all_targets.append(target)
        
        # 合并
        features = np.concatenate(all_features, axis=0)
        target = np.concatenate(all_targets, axis=0)
        self.n_samples = len(features)
        
        # 创建数据集
        dataset = TimeSeriesDatasetFromArrays(features, target, self.config.seq_len, self.config.pred_len)
        
        # 划分训练/测试
        n = len(dataset)
        train_size = int(n * 0.8)
        test_size = n - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.logger.info(f"  {self.client_id}: {self.n_samples} 样本, train={train_size}, test={test_size}")
        
    def set_model(self, model: nn.Module):
        """设置模型"""
        self.model = model
        
    def local_train(self, global_weights: Dict, epochs: int) -> Tuple[Dict, float]:
        """本地训练"""
        if self.model is None:
            raise ValueError("模型未设置")
        
        # 加载全局模型
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        total_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in self.train_loader:
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                
                # FedProx 近端项
                if self.config.fed_algorithm == 'fedprox':
                    prox_loss = 0.0
                    for param, global_param in zip(self.model.parameters(), global_weights.values()):
                        prox_loss += torch.norm(param - global_param) ** 2
                    loss += (self.config.mu / 2) * prox_loss
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(self.train_loader)
        
        avg_loss = total_loss / epochs
        return self.model.state_dict(), avg_loss
    
    def evaluate(self) -> float:
        """评估模型"""
        if self.model is None:
            raise ValueError("模型未设置")
        
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                output = self.model(x)
                loss = criterion(output, y)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)


class TimeSeriesDatasetFromArrays(Dataset):
    """从数组创建数据集"""
    
    def __init__(self, features: np.ndarray, target: np.ndarray, seq_len: int, pred_len: int):
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx:idx + self.seq_len]
        y = self.target[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y.flatten())


# ============================================================
# 联邦服务器
# ============================================================

class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.clients: List[FederatedClient] = []
        
    def initialize_model(self):
        """初始化模型"""
        self.model = LSTMPredictor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            output_dim=self.config.output_dim,
            dropout=self.config.dropout
        )
        self.logger.info(f"模型初始化: input_dim={self.config.input_dim}, hidden={self.config.hidden_dim}")
        
    def add_client(self, client: FederatedClient):
        """添加客户端"""
        client.set_model(self.model)
        client.load_data()
        self.clients.append(client)
        self.logger.info(f"添加客户端: {client.client_id}")
        
    def train(self) -> Dict:
        """联邦训练"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"联邦学习训练开始")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"算法: {self.config.fed_algorithm}")
        self.logger.info(f"轮数: {self.config.num_rounds}")
        self.logger.info(f"本地轮数: {self.config.local_epochs}")
        self.logger.info(f"客户端数: {len(self.clients)}")
        
        history = {
            'rounds': [],
            'client_losses': [],
            'avg_loss': []
        }
        
        for round_num in range(1, self.config.num_rounds + 1):
            self.logger.info(f"\n--- Round {round_num}/{self.config.num_rounds} ---")
            
            client_weights = []
            client_losses = []
            
            # 客户端本地训练
            for client in self.clients:
                weights, loss = client.local_train(self.model.state_dict(), self.config.local_epochs)
                client_weights.append(weights)
                client_losses.append(loss)
                self.logger.info(f"  {client.client_id}: loss={loss:.6f}")
            
            # 联邦聚合
            avg_loss = np.mean(client_losses)
            self.logger.info(f"  平均损失: {avg_loss:.6f}")
            
            # FedAvg 聚合
            new_weights = {}
            for key in client_weights[0].keys():
                new_weights[key] = torch.stack([w[key].float() for w in client_weights]).mean(dim=0)
            
            self.model.load_state_dict(new_weights)
            
            history['rounds'].append(round_num)
            history['client_losses'].append(client_losses)
            history['avg_loss'].append(avg_loss)
        
        return history
    
    def evaluate_all(self) -> Dict:
        """评估所有客户端"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"评估")
        self.logger.info(f"{'='*60}")
        
        results = {}
        for client in self.clients:
            loss = client.evaluate()
            results[client.client_id] = loss
            self.logger.info(f"  {client.client_id}: test_loss={loss:.6f}")
        
        return results


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='4G+5G代际协同联邦学习')
    parser.add_argument('--fed_algorithm', type=str, default='fedavg', choices=['fedavg', 'fedprox'])
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 创建配置
    config = Config(
        fed_algorithm=args.fed_algorithm,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # 设置日志
    logger = setup_logger(__name__, config.log_level)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    logger.info("="*60)
    logger.info("4G+5G代际协同联邦学习")
    logger.info("="*60)
    logger.info(f"配置: {json.dumps(asdict(config), indent=2, default=str)}")
    
    # 创建服务器
    server = FederatedServer(config, logger)
    server.initialize_model()
    
    # 添加客户端
    for name, info in config.datasets.items():
        data_dir = config.data_dir / info['dir']
        if data_dir.exists():
            client = FederatedClient(name, data_dir, config, logger)
            server.add_client(client)
    
    # 训练
    history = server.train()
    
    # 评估
    results = server.evaluate_all()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = config.results_dir / f"results_{timestamp}.json"
    
    output = {
        'config': asdict(config),
        'history': history,
        'results': results,
        'timestamp': timestamp
    }
    
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"\n✅ 结果保存: {result_path}")
    
    # 打印总结
    logger.info(f"\n{'='*60}")
    logger.info("训练完成")
    logger.info(f"{'='*60}")
    for client_id, loss in results.items():
        logger.info(f"  {client_id}: test_loss={loss:.6f}")
    
    return results


if __name__ == "__main__":
    main()
