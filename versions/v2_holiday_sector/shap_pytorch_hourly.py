#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP 特征重要性分析 - 保存真实小时级权重
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
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Config:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data" / "processed" / "tsinghua_v2"
        self.output_dir = self.project_root / "results" / "shap_analysis"
        self.logs_dir = self.project_root / "logs"
        
        self.data_type = '4g'
        self.max_stations = 50
        self.samples_per_station = 100
        self.seq_len = 24
        self.input_dim = 5
        self.hidden_dim = 64
        self.num_layers = 2
        self.dropout = 0.0
        self.batch_size = 64
        self.epochs = 20
        self.learning_rate = 0.001
        
        self.shap_background_samples = 50
        self.shap_test_samples = 30
        
        self.feature_names = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(42)
        torch.manual_seed(42)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(config):
    """加载数据"""
    data_dir = config.data_dir / config.data_type
    station_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
    
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
    
    X = np.concatenate(all_features, axis=0)
    
    # 创建序列
    X_seq = []
    for i in range(len(X) - config.seq_len):
        X_seq.append(X[i:i+config.seq_len])
    X_seq = np.array(X_seq, dtype=np.float32)
    
    y = X_seq[:, -1, 0:1]
    
    return X_seq, y


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_model(model, X, y, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def compute_shap_with_hourly(model, X, background, config):
    """计算 SHAP 值并返回小时级权重"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    background_tensor = torch.FloatTensor(background).to(device)
    
    explainer = shap.GradientExplainer(model, background_tensor)
    shap_values = explainer.shap_values(X_tensor)
    
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    # shap_values 形状: [n_samples, seq_len, n_features] 或 [n_samples, seq_len, n_features, 1]
    if shap_values.ndim == 4:
        shap_values = shap_values.squeeze(-1)
    
    # 计算每小时平均重要性 [seq_len, n_features]
    hourly_importance = np.abs(shap_values).mean(axis=0)
    
    # 计算整体特征重要性
    total_importance = hourly_importance.mean(axis=0)
    
    return shap_values, hourly_importance, total_importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='4g', choices=['4g', '5g'])
    parser.add_argument('--max_stations', type=int, default=50)
    parser.add_argument('--samples_per_station', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    config = Config()
    config.data_type = args.data_type
    config.max_stations = args.max_stations
    config.samples_per_station = args.samples_per_station
    config.epochs = args.epochs
    
    logger.info("="*60)
    logger.info(f"SHAP 分析 - {config.data_type.upper()} (保存真实小时级权重)")
    logger.info("="*60)
    
    # 加载数据
    logger.info("\n1. 加载数据...")
    X_seq, y = load_data(config)
    logger.info(f"   序列样本数: {len(X_seq):,}")
    
    # 训练模型
    logger.info("\n2. 训练模型...")
    model = LSTMPredictor(config.input_dim, config.hidden_dim, config.num_layers)
    model = train_model(model, X_seq, y, config)
    
    # 计算 SHAP
    logger.info("\n3. 计算 SHAP 值...")
    background = X_seq[:config.shap_background_samples]
    test_samples = X_seq[config.shap_background_samples:config.shap_background_samples + config.shap_test_samples]
    
    shap_values, hourly_importance, total_importance = compute_shap_with_hourly(
        model, test_samples, background, config
    )
    
    logger.info(f"   SHAP 形状: {shap_values.shape}")
    logger.info(f"   小时级重要性形状: {hourly_importance.shape}")
    
    # 打印结果
    logger.info("\n" + "="*50)
    logger.info("特征重要性排序")
    logger.info("="*50)
    sorted_idx = np.argsort(total_importance)[::-1]
    for idx in sorted_idx:
        logger.info(f"  {config.feature_names[idx]}: {total_importance[idx]:.6f}")
    
    # 打印小时级关键时段
    logger.info("\n" + "="*50)
    logger.info("关键时段特征重要性")
    logger.info("="*50)
    for h in [8, 12, 18, 22]:
        logger.info(f"\n小时 {h}:")
        for i, name in enumerate(config.feature_names):
            logger.info(f"  {name}: {hourly_importance[h, i]:.6f}")
    
    # 保存结果（包含真实小时级权重）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存小时级权重热力图
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(hourly_importance.T, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('时间步 (相对位置)')
    ax.set_ylabel('特征')
    ax.set_title(f'{config.data_type.upper()} 基站 - 真实小时级特征重要性')
    ax.set_yticks(range(len(config.feature_names)))
    ax.set_yticklabels(config.feature_names)
    plt.colorbar(im, ax=ax, label='SHAP 重要性')
    plt.tight_layout()
    plt.savefig(config.output_dir / f'hourly_importance_{config.data_type}_{timestamp}.png', dpi=150)
    plt.close()
    
    # 保存条形图
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(config.feature_names, total_importance, color='#2E8B57')
    ax.set_xlabel('平均 SHAP 重要性')
    ax.set_title(f'{config.data_type.upper()} 基站特征重要性')
    for bar, val in zip(bars, total_importance):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.6f}', va='center')
    plt.tight_layout()
    plt.savefig(config.output_dir / f'feature_importance_{config.data_type}_{timestamp}.png', dpi=150)
    plt.close()
    
    # 保存 JSON（包含真实小时级权重）
    results = {
        'data_type': config.data_type,
        'timestamp': timestamp,
        'feature_importance': {name: float(imp) for name, imp in zip(config.feature_names, total_importance)},
        'hourly_importance': {f'h{h}': [float(val) for val in hourly_importance[h]] for h in range(hourly_importance.shape[0])},
        'hourly_importance_matrix': hourly_importance.tolist(),
        'feature_names': config.feature_names,
        'config': {
            'max_stations': config.max_stations,
            'samples_per_station': config.samples_per_station,
            'epochs': config.epochs
        }
    }
    
    result_path = config.output_dir / f'shap_results_{config.data_type}_{timestamp}.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✅ 结果保存: {result_path}")
    logger.info(f"   ✅ 包含真实小时级权重 (24时间步 × {len(config.feature_names)}特征)")


if __name__ == "__main__":
    main()
