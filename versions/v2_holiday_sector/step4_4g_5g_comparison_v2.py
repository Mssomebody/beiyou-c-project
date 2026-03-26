#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4G+5G代际协同对比实验 - 五星级专业版

实验设计:
    Exp1: 4G单独训练 (基线)
    Exp2: 5G单独训练 (基线)  
    Exp3: 4G+5G联邦学习 (代际协同)

假设: 5G数据少，联邦学习能让5G从4G学到知识，精度提升

作者: FedGreen-C
版本: 2.0
"""

import os
import sys
import pickle
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
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
    """集中配置管理 - 使用 dataclass"""
    
    # 路径配置
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    
    # 数据配置
    max_stations: int = 500          # 最大基站数（完整训练用全部）
    train_ratio: float = 0.8         # 训练集比例
    
    # 模型配置
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 24                # 输入序列长度
    pred_len: int = 1                # 预测长度
    
    # 训练配置
    batch_size: int = 64
    epochs: int = 50                 # 单独训练轮数
    learning_rate: float = 0.001
    
    # 联邦配置
    local_epochs: int = 5
    num_rounds: int = 10
    fed_algorithm: str = 'fedavg'    # 'fedavg' or 'fedprox'
    mu: float = 0.01                 # FedProx 系数
    
    # 其他
    seed: int = 42
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """初始化后处理"""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "processed" / "tsinghua"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results" / "4g_5g_comparison"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        # 创建目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 日志系统
# ============================================================

class Logger:
    """统一日志管理"""
    
    @staticmethod
    def setup(name: str = __name__, level: str = 'INFO', log_file: Optional[Path] = None) -> logging.Logger:
        """配置日志器"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        
        if logger.handlers:
            return logger
        
        # 控制台处理器
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console)
        
        # 文件处理器
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
        
        return logger


# ============================================================
# 数据集
# ============================================================

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, features: np.ndarray, target: np.ndarray, 
                 seq_len: int = 24, pred_len: int = 1):
        """
        初始化数据集
        
        Args:
            features: 特征矩阵 [n_samples, n_features]
            target: 目标向量 [n_samples, 1]
            seq_len: 输入序列长度
            pred_len: 预测长度
        """
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx:idx + self.seq_len]
        y = self.target[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y.flatten())


class LSTMPredictor(nn.Module):
    """LSTM预测模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)
        return out
    
    def get_params(self) -> Dict:
        """获取模型参数"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout.p
        }


# ============================================================
# 数据加载器
# ============================================================

class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def load_station_data(data_dir: Path, max_stations: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载基站数据
        
        Args:
            data_dir: 数据目录
            max_stations: 最大基站数，None表示全部
        
        Returns:
            (features, target) 合并后的特征和目标
        """
        # 获取所有基站
        station_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
        
        if max_stations:
            station_dirs = station_dirs[:max_stations]
        
        all_features = []
        all_targets = []
        
        for station_dir in tqdm(station_dirs, desc=f"加载 {data_dir.name}", leave=False):
            data_path = station_dir / 'data.pkl'
            if not data_path.exists():
                continue
            
            try:
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                all_features.append(data['features'])
                all_targets.append(data['target'])
            except Exception as e:
                print(f"警告: 读取失败 {data_path}: {e}")
                continue
        
        if not all_features:
            raise ValueError(f"没有成功加载任何数据: {data_dir}")
        
        features = np.concatenate(all_features, axis=0)
        target = np.concatenate(all_targets, axis=0)
        
        return features, target
    
    @staticmethod
    def create_dataloaders(features: np.ndarray, target: np.ndarray, 
                           batch_size: int, seq_len: int, pred_len: int,
                           train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        创建训练和测试数据加载器
        
        Returns:
            (train_loader, test_loader)
        """
        dataset = TimeSeriesDataset(features, target, seq_len, pred_len)
        
        n = len(dataset)
        train_size = int(n * train_ratio)
        test_size = n - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader


# ============================================================
# 训练器
# ============================================================

class Trainer:
    """训练器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"设备: {self.device}")
    
    def train_single(self, model: nn.Module, train_loader: DataLoader, 
                     test_loader: DataLoader) -> Tuple[nn.Module, float]:
        """单独训练"""
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0.0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 评估
            test_loss = self.evaluate(model, test_loader, criterion)
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"  Epoch {epoch+1}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
        
        model.load_state_dict(best_state)
        return model, best_loss
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader, 
                 criterion: nn.Module) -> float:
        """评估模型"""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                loss = criterion(output, y)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def federated_train(self, models: Dict[str, List[nn.Module]], 
                        loaders: Dict[str, DataLoader]) -> Dict[str, List[nn.Module]]:
        """联邦训练"""
        self.logger.info(f"联邦学习: {self.config.fed_algorithm}, {self.config.num_rounds}轮")
        
        for client_models in models.values():
            for m in client_models:
                m.to(self.device)
        
        criterion = nn.MSELoss()
        
        for round_num in range(1, self.config.num_rounds + 1):
            client_updates = []
            client_losses = []
            
            for client_name, client_models_list in models.items():
                for model in client_models_list:
                    optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
                    
                    for _ in range(self.config.local_epochs):
                        model.train()
                        for x, y in loaders[client_name]:
                            x, y = x.to(self.device), y.to(self.device)
                            optimizer.zero_grad()
                            output = model(x)
                            loss = criterion(output, y)
                            
                            # FedProx
                            if self.config.fed_algorithm == 'fedprox':
                                prox_loss = 0.0
                                for param in model.parameters():
                                    prox_loss += torch.norm(param) ** 2
                                loss += (self.config.mu / 2) * prox_loss
                            
                            loss.backward()
                            optimizer.step()
                    
                    client_updates.append(model.state_dict())
                    client_losses.append(loss.item())
            
            # 聚合
            avg_state = {}
            for key in client_updates[0].keys():
                avg_state[key] = torch.stack([u[key].float() for u in client_updates]).mean(dim=0)
            
            for client_models in models.values():
                for model in client_models:
                    model.load_state_dict(avg_state)
            
            if round_num % 5 == 0:
                self.logger.info(f"  Round {round_num}: avg_loss={np.mean(client_losses):.6f}")
        
        return models


# ============================================================
# 实验运行器
# ============================================================

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.trainer = Trainer(config, self.logger)
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.config.logs_dir / f"4g_5g_comparison_{timestamp}.log"
        return Logger.setup(__name__, self.config.log_level, log_file)
    
    def run(self) -> Dict[str, Any]:
        """运行所有实验"""
        self.logger.info("="*60)
        self.logger.info("4G+5G代际协同对比实验")
        self.logger.info("="*60)
        self.logger.info(f"配置: {asdict(self.config)}")
        
        results = {}
        
        # 加载数据
        self.logger.info("\n加载数据...")
        
        data_4g = self.config.data_dir / '4g'
        features_4g, target_4g = DataLoaderFactory.load_station_data(
            data_4g, self.config.max_stations
        )
        self.logger.info(f"4G: {len(features_4g)} 样本, {features_4g.shape[1]} 特征")
        
        data_5g = self.config.data_dir / '5g'
        features_5g, target_5g = DataLoaderFactory.load_station_data(
            data_5g, self.config.max_stations
        )
        self.logger.info(f"5G: {len(features_5g)} 样本, {features_5g.shape[1]} 特征")
        
        # 创建数据加载器
        train_4g, test_4g = DataLoaderFactory.create_dataloaders(
            features_4g, target_4g, self.config.batch_size,
            self.config.seq_len, self.config.pred_len, self.config.train_ratio
        )
        train_5g, test_5g = DataLoaderFactory.create_dataloaders(
            features_5g, target_5g, self.config.batch_size,
            self.config.seq_len, self.config.pred_len, self.config.train_ratio
        )
        
        # ========== 实验1: 4G单独训练 ==========
        self.logger.info("\n" + "="*60)
        self.logger.info("实验1: 4G单独训练")
        self.logger.info("="*60)
        
        model_4g = LSTMPredictor(
            input_dim=features_4g.shape[1],
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        _, loss_4g = self.trainer.train_single(model_4g, train_4g, test_4g)
        results['4g_alone'] = loss_4g
        self.logger.info(f"4G单独训练测试损失: {loss_4g:.6f}")
        
        # ========== 实验2: 5G单独训练 ==========
        self.logger.info("\n" + "="*60)
        self.logger.info("实验2: 5G单独训练")
        self.logger.info("="*60)
        
        model_5g = LSTMPredictor(
            input_dim=features_5g.shape[1],
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        _, loss_5g = self.trainer.train_single(model_5g, train_5g, test_5g)
        results['5g_alone'] = loss_5g
        self.logger.info(f"5G单独训练测试损失: {loss_5g:.6f}")
        
        # ========== 实验3: 联邦学习 ==========
        self.logger.info("\n" + "="*60)
        self.logger.info("实验3: 4G+5G联邦学习")
        self.logger.info("="*60)
        
        models = {
            '4g': [LSTMPredictor(
                input_dim=features_4g.shape[1],
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )],
            '5g': [LSTMPredictor(
                input_dim=features_5g.shape[1],
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )]
        }
        
        loaders = {'4g': train_4g, '5g': train_5g}
        models = self.trainer.federated_train(models, loaders)
        
        loss_4g_fed = self.trainer.evaluate(models['4g'][0], test_4g, nn.MSELoss())
        loss_5g_fed = self.trainer.evaluate(models['5g'][0], test_5g, nn.MSELoss())
        
        results['4g_federated'] = loss_4g_fed
        results['5g_federated'] = loss_5g_fed
        
        self.logger.info(f"4G联邦学习测试损失: {loss_4g_fed:.6f}")
        self.logger.info(f"5G联邦学习测试损失: {loss_5g_fed:.6f}")
        
        # ========== 结果汇总 ==========
        self.logger.info("\n" + "="*60)
        self.logger.info("结果汇总")
        self.logger.info("="*60)
        self.logger.info(f"4G单独训练:     {results['4g_alone']:.6f}")
        self.logger.info(f"4G联邦学习:     {results['4g_federated']:.6f}")
        self.logger.info(f"4G提升:         {results['4g_alone'] - results['4g_federated']:.6f}")
        self.logger.info(f"")
        self.logger.info(f"5G单独训练:     {results['5g_alone']:.6f}")
        self.logger.info(f"5G联邦学习:     {results['5g_federated']:.6f}")
        self.logger.info(f"5G提升:         {results['5g_alone'] - results['5g_federated']:.6f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.config.results_dir / f"results_{timestamp}.json"
        
        with open(result_path, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': results,
                'timestamp': timestamp
            }, f, indent=2, default=str)
        
        self.logger.info(f"\n✅ 结果保存: {result_path}")
        
        return results


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='4G+5G代际协同对比实验')
    
    parser.add_argument('--max_stations', type=int, default=500,
                        help='最大基站数 (默认500，设0表示全部)')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--fed_algorithm', type=str, default='fedavg',
                        choices=['fedavg', 'fedprox'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_level', type=str, default='INFO')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = Config(
        max_stations=args.max_stations if args.max_stations > 0 else None,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_rounds=args.num_rounds,
        fed_algorithm=args.fed_algorithm,
        seed=args.seed,
        log_level=args.log_level
    )
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    try:
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # 打印结论
        improvement_4g = results['4g_alone'] - results['4g_federated']
        improvement_5g = results['5g_alone'] - results['5g_federated']
        
        print("\n" + "="*60)
        print("结论")
        print("="*60)
        if improvement_5g > 0:
            print(f"✅ 联邦学习有效！5G精度提升 {improvement_5g:.6f}")
        else:
            print(f"⚠️ 5G精度未提升，需调整参数")
        
        return 0
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
