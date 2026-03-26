#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP 特征重要性分析 - GradientExplainer (PyTorch 原生)
五星级专业版，修复索引错误
"""

import os
import sys
import pickle
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 配置管理
# ============================================================

@dataclass
class Config:
    """集中配置管理"""
    
    # 路径配置
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    
    # 数据配置
    data_type: str = '4g'
    max_stations: int = 50
    samples_per_station: int = 100
    
    # 模型配置
    seq_len: int = 24
    pred_len: int = 1
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    
    # 训练配置
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    device: str = 'cpu'
    
    # SHAP 配置
    shap_background_samples: int = 50
    shap_test_samples: int = 30
    
    # 特征名称
    feature_names: List[str] = field(default_factory=lambda: [
        'PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos'
    ])
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "processed" / "tsinghua_v2"
        if self.output_dir is None:
            self.output_dir = self.project_root / "results" / "shap_analysis"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 日志系统
# ============================================================

def setup_logger(name: str = __name__, config: Config = None) -> logging.Logger:
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
        log_file = config.logs_dir / f"shap_analysis_{config.data_type}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# 数据加载器
# ============================================================

def load_data(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """加载数据"""
    data_dir = config.data_dir / config.data_type
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    station_dirs = [d for d in data_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('station_')]
    
    if config.max_stations:
        station_dirs = station_dirs[:config.max_stations]
    
    all_features = []
    
    for s_dir in tqdm(station_dirs, desc=f"加载 {config.data_type} 数据"):
        with open(s_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        features = data.get('X_train_norm', data.get('features_norm', None))
        if features is None:
            continue
        
        if config.samples_per_station:
            features = features[:config.samples_per_station]
        
        all_features.append(features)
    
    if not all_features:
        raise ValueError(f"没有成功加载任何数据: {data_dir}")
    
    X = np.concatenate(all_features, axis=0)
    
    # 创建序列
    X_seq = []
    for i in range(len(X) - config.seq_len):
        X_seq.append(X[i:i+config.seq_len])
    X_seq = np.array(X_seq, dtype=np.float32)
    
    y = X_seq[:, -1, 0:1]
    
    return X_seq, y


# ============================================================
# 模型定义
# ============================================================

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============================================================
# 训练函数
# ============================================================

def train_model(model: nn.Module, X: np.ndarray, y: np.ndarray, 
                config: Config, logger: logging.Logger) -> nn.Module:
    """训练模型"""
    device = torch.device(config.device)
    model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(config.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
    
    return model


# ============================================================
# SHAP 分析
# ============================================================

def compute_shap(model: nn.Module, X: np.ndarray, background: np.ndarray,
                 config: Config, logger: logging.Logger) -> np.ndarray:
    """计算 SHAP 值"""
    device = torch.device(config.device)
    model.eval()
    model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    background_tensor = torch.FloatTensor(background).to(device)
    
    explainer = shap.GradientExplainer(model, background_tensor)
    shap_values = explainer.shap_values(X_tensor)
    
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    logger.info(f"  SHAP 值形状: {shap_values.shape}")
    
    return shap_values


# ============================================================
# 可视化
# ============================================================

def plot_feature_importance(importance: np.ndarray, feature_names: List[str],
                            output_path: Path, data_type: str, logger: logging.Logger):
    """绘制特征重要性条形图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(feature_names, importance, color='#2E8B57', edgecolor='black')
    ax.set_xlabel('平均 SHAP 重要性')
    ax.set_title(f'{data_type.upper()} 基站特征重要性 (GradientExplainer)')
    
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"  保存: {output_path}")
    plt.close()


def plot_hourly_importance(shap_values: np.ndarray, feature_names: List[str],
                           output_path: Path, data_type: str, logger: logging.Logger):
    """绘制时间步特征重要性热力图"""
    # shap_values: [n_samples, seq_len, n_features] 或 [n_samples, seq_len, n_features, 1]
    if shap_values.ndim == 4:
        shap_values = shap_values.squeeze(-1)
    
    hourly_importance = np.abs(shap_values).mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(hourly_importance.T, aspect='auto', cmap='YlOrRd')
    
    ax.set_xlabel('时间步 (相对位置)')
    ax.set_ylabel('特征')
    ax.set_title(f'{data_type.upper()} 基站 - 每个时间步的特征重要性')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    
    plt.colorbar(im, ax=ax, label='SHAP 重要性')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"  保存: {output_path}")
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SHAP 特征重要性分析')
    parser.add_argument('--data_type', type=str, default='4g', choices=['4g', '5g'])
    parser.add_argument('--max_stations', type=int, default=50)
    parser.add_argument('--samples_per_station', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    config = Config(
        data_type=args.data_type,
        max_stations=args.max_stations,
        samples_per_station=args.samples_per_station,
        epochs=args.epochs
    )
    
    logger = setup_logger(__name__, config)
    
    logger.info("="*60)
    logger.info(f"SHAP 特征重要性分析 - {config.data_type.upper()} (GradientExplainer)")
    logger.info("="*60)
    logger.info(f"  设备: {config.device}")
    logger.info(f"  最大基站数: {config.max_stations}")
    logger.info(f"  每基站样本数: {config.samples_per_station}")
    
    # 1. 加载数据
    logger.info("\n1. 加载数据...")
    X_seq, y = load_data(config)
    logger.info(f"   序列样本数: {len(X_seq):,}")
    logger.info(f"   序列形状: {X_seq.shape}")
    
    # 2. 训练模型
    logger.info("\n2. 训练模型...")
    model = LSTMPredictor(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    model = train_model(model, X_seq, y, config, logger)
    
    # 3. 计算 SHAP
    logger.info("\n3. 计算 SHAP 值...")
    background = X_seq[:config.shap_background_samples]
    test_samples = X_seq[config.shap_background_samples:config.shap_background_samples + config.shap_test_samples]
    logger.info(f"   背景样本: {background.shape}")
    logger.info(f"   测试样本: {test_samples.shape}")
    
    shap_values = compute_shap(model, test_samples, background, config, logger)
    
    # 4. 计算特征重要性
    logger.info("\n4. 计算特征重要性...")
    # shap_values 形状: (n_samples, seq_len, n_features) 或 (n_samples, seq_len, n_features, 1)
    if shap_values.ndim == 4:
        shap_values = shap_values.squeeze(-1)
    
    importance = np.abs(shap_values).mean(axis=(0, 1))
    
    # 打印结果
    logger.info("\n" + "="*50)
    logger.info("特征重要性排序")
    logger.info("="*50)
    sorted_idx = np.argsort(importance)[::-1]
    for idx in sorted_idx:
        logger.info(f"  {config.feature_names[idx]}: {importance[idx]:.4f}")
    
    # 5. 可视化
    logger.info("\n5. 生成可视化...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    bar_path = config.output_dir / f"feature_importance_{config.data_type}_{timestamp}.png"
    plot_feature_importance(importance, config.feature_names, bar_path, config.data_type, logger)
    
    heat_path = config.output_dir / f"hourly_importance_{config.data_type}_{timestamp}.png"
    plot_hourly_importance(shap_values, config.feature_names, heat_path, config.data_type, logger)
    
    # 6. 保存结果
    logger.info("\n6. 保存结果...")
    results = {
        'data_type': config.data_type,
        'timestamp': timestamp,
        'feature_importance': {
            name: float(imp) for name, imp in zip(config.feature_names, importance)
        },
        'sorted_features': [
            {config.feature_names[i]: float(importance[i])} for i in sorted_idx
        ],
        'shap_values_shape': list(shap_values.shape)
    }
    
    result_path = config.output_dir / f"shap_results_{config.data_type}_{timestamp}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"   保存: {result_path}")
    
    logger.info("\n" + "="*50)
    logger.info("分析完成")
    logger.info("="*50)


if __name__ == "__main__":
    main()
