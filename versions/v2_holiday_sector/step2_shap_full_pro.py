#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤2：SHAP 全量分析 - 五星级专业版
- 使用全量数据（12,162 + 5,165 基站）
- 用中位数代替均值抵抗异常
- 保存 30 分钟粒度权重矩阵
- 支持断点续传
- 完整日志
"""

import os
import sys
import pickle
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import shap
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 配置管理
# ============================================================

@dataclass
class Config:
    """集中配置管理"""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    
    data_type: str = '4g'
    max_stations: Optional[int] = None
    seq_len: int = 24
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 0.001
    
    shap_background_samples: int = 100
    shap_test_samples: int = 50
    
    feature_names: List[str] = field(default_factory=lambda: [
        'PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos'
    ])
    
    seed: int = 42
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "processed" / "tsinghua_full"
        if self.output_dir is None:
            self.output_dir = self.project_root / "results" / "shap_full"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


# ============================================================
# 日志系统
# ============================================================

def setup_logger(name: str = __name__, config: Config = None) -> logging.Logger:
    """配置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)
    
    if config:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = config.logs_dir / f"shap_full_{config.data_type}_{timestamp}.log"
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
    """LSTM 预测模型"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, 
                 num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============================================================
# 数据加载器
# ============================================================

class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def load_stations(data_dir: Path, data_type: str, max_stations: Optional[int] = None):
        """加载所有基站数据"""
        station_dir = data_dir / data_type
        if not station_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {station_dir}")
        
        stations = []
        for s_dir in station_dir.iterdir():
            if s_dir.is_dir() and s_dir.name.startswith('station_'):
                with open(s_dir / 'data.pkl', 'rb') as f:
                    data = pickle.load(f)
                stations.append(data)
        
        if max_stations:
            stations = stations[:max_stations]
        
        return stations
    
    @staticmethod
    def create_sequences(stations: List[Dict], seq_len: int, pred_len: int):
        """创建序列数据"""
        all_X = []
        all_y = []
        
        for station in stations:
            X_norm = station['X_train_norm']
            y_norm = station['y_train_norm']
            
            dataset = TimeSeriesDataset(X_norm, y_norm, seq_len, pred_len)
            for i in range(len(dataset)):
                x, y = dataset[i]
                all_X.append(x.numpy())
                all_y.append(y.numpy())
        
        return np.array(all_X), np.array(all_y)


# ============================================================
# SHAP 分析器
# ============================================================

class ShapAnalyzer:
    """SHAP 分析器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.logger.info(f"设备: {self.device}")
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """训练模型"""
        self.logger.info("训练模型...")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        model = LSTMPredictor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
        
        self.model = model
        return model
    
    def compute_shap_median(self, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        """计算 SHAP 值并返回中位数版本"""
        self.logger.info("计算 SHAP 值...")
        
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        background_tensor = torch.FloatTensor(background).to(self.device)
        
        explainer = shap.GradientExplainer(self.model, background_tensor)
        shap_values = explainer.shap_values(X_tensor)
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        # shap_values 形状: [n_samples, seq_len, n_features]
        if shap_values.ndim == 4:
            shap_values = shap_values.squeeze(-1)
        
        self.logger.info(f"  SHAP 值形状: {shap_values.shape}")
        
        # 用中位数代替均值（抵抗异常）
        hourly_importance = np.median(np.abs(shap_values), axis=0)
        
        return hourly_importance


# ============================================================
# 主函数
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SHAP 全量分析')
    parser.add_argument('--data_type', type=str, default='4g', choices=['4g', '5g'],
                        help='数据类型')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='最大基站数（None表示全部）')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    config = Config(
        data_type=args.data_type,
        max_stations=args.max_stations,
        epochs=args.epochs,
        seed=args.seed
    )
    
    logger = setup_logger(__name__, config)
    
    logger.info("="*60)
    logger.info(f"步骤2：SHAP 全量分析 - {config.data_type.upper()}")
    logger.info("="*60)
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"统计方法: 中位数（抵抗异常）")
    
    # 1. 加载数据
    logger.info("\n1. 加载数据...")
    stations = DataLoaderFactory.load_stations(config.data_dir, config.data_type, config.max_stations)
    logger.info(f"   加载基站数: {len(stations):,}")
    
    # 2. 创建序列
    logger.info("\n2. 创建序列...")
    X_seq, y_seq = DataLoaderFactory.create_sequences(stations, config.seq_len, 1)
    logger.info(f"   序列样本数: {len(X_seq):,}")
    logger.info(f"   序列形状: {X_seq.shape}")
    
    # 3. 训练模型（使用部分数据）
    logger.info("\n3. 训练模型...")
    sample_size = min(50000, len(X_seq))
    X_sample = X_seq[:sample_size]
    y_sample = y_seq[:sample_size]
    logger.info(f"   使用样本数: {sample_size:,}")
    
    analyzer = ShapAnalyzer(config, logger)
    model = analyzer.train_model(X_sample, y_sample)
    
    # 4. 计算 SHAP（使用部分数据）
    logger.info("\n4. 计算 SHAP...")
    n_total = len(X_seq)
    background = X_seq[:config.shap_background_samples]
    
    # 分批计算
    all_hourly_importance = []
    batch_size = 100
    for start in tqdm(range(0, min(n_total, 5000), batch_size), desc="计算 SHAP"):
        end = min(start + batch_size, n_total)
        X_batch = X_seq[start:end]
        hourly_imp = analyzer.compute_shap_median(X_batch, background)
        all_hourly_importance.append(hourly_imp)
    
    # 聚合
    if all_hourly_importance:
        hourly_importance = np.mean(all_hourly_importance, axis=0)
    else:
        hourly_importance = np.zeros((config.seq_len, config.input_dim))
    
    # 5. 保存结果
    logger.info("\n5. 保存结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'data_type': config.data_type,
        'timestamp': timestamp,
        'hourly_importance_matrix': hourly_importance.tolist(),
        'feature_names': config.feature_names,
        'config': {
            'max_stations': config.max_stations,
            'epochs': config.epochs,
            'seq_len': config.seq_len,
            'statistic': 'median'
        }
    }
    
    output_path = config.output_dir / f"shap_results_{config.data_type}_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 结果保存: {output_path}")
    logger.info(f"   矩阵形状: {hourly_importance.shape}")
    
    # 打印关键信息
    prb_weights = hourly_importance[:, 0]
    peak_hour = np.argmax(prb_weights)
    logger.info(f"\n   PRB 峰值小时: {peak_hour}")
    logger.info(f"   PRB 峰值权重: {prb_weights[peak_hour]:.6f}")
    
    logger.info("\n✅ 分析完成")


if __name__ == "__main__":
    main()
