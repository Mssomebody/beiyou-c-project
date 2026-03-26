#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP 小时级特征重要性分析 - KernelExplainer 版
更稳定，适用于 LSTM 模型
"""

import pickle
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 配置
# ============================================================

@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    
    data_type: str = '4g'
    max_stations: int = 50
    samples_per_station: int = 100
    seq_len: int = 24
    pred_len: int = 1
    
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    
    shap_background_samples: int = 50
    shap_test_samples: int = 30
    
    feature_names: List[str] = field(default_factory=lambda: ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos'])
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "processed" / "tsinghua_v2"
        if self.output_dir is None:
            self.output_dir = self.project_root / "results" / "shap_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(42)
        torch.manual_seed(42)


# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据集和模型
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, target: np.ndarray, 
                 seq_len: int = 24, pred_len: int = 1):
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.target[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y.flatten())


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # 支持 [seq_len, features] 和 [batch, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_out))


# ============================================================
# 数据加载
# ============================================================

def load_station_data(data_dir: Path, data_type: str, 
                      max_stations: int = 50,
                      samples_per_station: int = 100):
    """加载基站数据"""
    station_dir = data_dir / data_type
    if not station_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {station_dir}")
    
    station_dirs = [d for d in station_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
    station_dirs = station_dirs[:max_stations]
    
    all_features = []
    all_targets = []
    
    for s_dir in tqdm(station_dirs, desc=f"加载 {data_type} 数据"):
        with open(s_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        features = data.get('X_train_norm', data.get('features_norm', None))
        target = data.get('y_train_norm', data.get('target_norm', None))
        
        if features is None or target is None:
            continue
        
        if samples_per_station:
            features = features[:samples_per_station]
            target = target[:samples_per_station]
        
        all_features.append(features)
        all_targets.append(target)
    
    features = np.concatenate(all_features, axis=0)
    target = np.concatenate(all_targets, axis=0)
    
    return features, target


def create_sequences(features: np.ndarray, target: np.ndarray, 
                     seq_len: int = 24, pred_len: int = 1):
    """创建序列样本"""
    X, y = [], []
    for i in range(len(features) - seq_len - pred_len + 1):
        X.append(features[i:i+seq_len].reshape(-1))
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SHAP 分析 - KernelExplainer')
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
    
    logger.info("="*60)
    logger.info(f"SHAP 特征重要性分析 - {config.data_type.upper()} (KernelExplainer)")
    logger.info("="*60)
    
    # 加载数据
    logger.info("\n1. 加载数据...")
    X, y = load_station_data(config.data_dir, config.data_type,
                              config.max_stations, config.samples_per_station)
    logger.info(f"   样本数: {len(X):,}")
    
    # 创建序列
    logger.info("\n2. 创建序列样本...")
    X_seq, y_seq = create_sequences(X, y, config.seq_len, config.pred_len)
    logger.info(f"   序列样本数: {len(X_seq):,}")
    
    # 训练模型
    logger.info("\n3. 训练模型...")
    model = LSTMPredictor(config.input_dim, config.hidden_dim, config.num_layers, config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    dataset = TimeSeriesDataset(X, y, config.seq_len, config.pred_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    for epoch in range(config.epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
    
    # SHAP 分析
    logger.info("\n4. 计算 SHAP 值...")
    model.eval()
    
    # 选择背景样本和测试样本
    background = X_seq[:config.shap_background_samples]
    test_samples = X_seq[config.shap_background_samples:config.shap_background_samples + config.shap_test_samples]
    
    # 创建包装函数（KernelExplainer 需要接收展平的输入）
    def model_predict(x):
        x_tensor = torch.FloatTensor(x).reshape(-1, config.seq_len, config.input_dim)
        with torch.no_grad():
            return model(x_tensor).numpy().flatten()
    
    # 创建解释器
    explainer = shap.KernelExplainer(model_predict, background)
    
    # 计算 SHAP 值
    shap_values = explainer.shap_values(test_samples, nsamples=100)
    
    # 计算每个特征的平均重要性（24个时间步平均）
    n_features = config.input_dim
    importance_per_feature = np.zeros(n_features)
    for i in range(n_features):
        importance_per_feature[i] = np.abs(shap_values[:, i::n_features]).mean()
    
    # 打印结果
    logger.info("\n" + "="*50)
    logger.info("特征重要性排序")
    logger.info("="*50)
    sorted_idx = np.argsort(importance_per_feature)[::-1]
    for idx in sorted_idx:
        logger.info(f"  {config.feature_names[idx]}: {importance_per_feature[idx]:.4f}")
    
    # 可视化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(config.feature_names, importance_per_feature, color='#2E8B57')
    ax.set_xlabel('平均 SHAP 重要性')
    ax.set_title(f'{config.data_type.upper()} 基站特征重要性')
    for bar, val in zip(bars, importance_per_feature):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center')
    plt.tight_layout()
    
    output_path = config.output_dir / f"feature_importance_{config.data_type}_{timestamp}.png"
    plt.savefig(output_path, dpi=150)
    logger.info(f"\n✅ 图片保存: {output_path}")
    
    # 保存结果
    results = {
        'data_type': config.data_type,
        'timestamp': timestamp,
        'feature_importance': {name: float(imp) for name, imp in zip(config.feature_names, importance_per_feature)},
        'sorted': [{config.feature_names[i]: float(importance_per_feature[i])} for i in sorted_idx]
    }
    
    result_path = config.output_dir / f"shap_results_{config.data_type}_{timestamp}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ 结果保存: {result_path}")


if __name__ == "__main__":
    main()
