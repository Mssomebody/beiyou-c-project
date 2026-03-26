#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP 小时级特征重要性分析 - 修复版
"""

import pickle
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
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
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data" / "processed" / "tsinghua_v2"
    output_dir: Path = project_root / "results" / "shap_analysis"
    logs_dir: Path = project_root / "logs"
    
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
    
    feature_names: List[str] = None
    
    seed: int = 42
    
    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据集（修复版）
# ============================================================

class TimeSeriesDataset(Dataset):
    """时间序列数据集 - 修复维度问题"""
    
    def __init__(self, features: np.ndarray, target: np.ndarray, 
                 seq_len: int = 24, pred_len: int = 1):
        # features: [n_samples, n_features]
        # target: [n_samples, 1]
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入: [seq_len, n_features]
        x = self.features[idx:idx + self.seq_len]
        # 输出: [pred_len] -> 取第一个预测点
        y = self.target[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y.flatten())


class LSTMPredictor(nn.Module):
    """LSTM预测模型"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        last_out = lstm_out[:, -1, :]  # [batch, hidden_dim]
        out = self.dropout(last_out)
        return self.fc(out)  # [batch, 1]


# ============================================================
# 数据加载
# ============================================================

def load_station_data(data_dir: Path, data_type: str, 
                      max_stations: int = None,
                      samples_per_station: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载基站数据"""
    station_dir = data_dir / data_type
    if not station_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {station_dir}")
    
    station_dirs = [d for d in station_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
    if max_stations:
        station_dirs = station_dirs[:max_stations]
    
    all_features = []
    all_targets = []
    all_hours = []
    
    for s_dir in tqdm(station_dirs, desc=f"加载 {data_type} 数据"):
        with open(s_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # 获取数据
        features = data.get('X_train_norm', data.get('features_norm', None))
        target = data.get('y_train_norm', data.get('target_norm', None))
        
        if features is None or target is None:
            continue
        
        if samples_per_station:
            features = features[:samples_per_station]
            target = target[:samples_per_station]
        
        # 提取小时
        if features.shape[1] >= 5:
            hour_sin = features[:, 3]
            hour_cos = features[:, 4]
            hour = np.arctan2(hour_sin, hour_cos) / (2 * np.pi) * 24
            hour = (hour + 24) % 24
        else:
            hour = np.zeros(len(features))
        
        all_features.append(features)
        all_targets.append(target)
        all_hours.append(hour)
    
    features = np.concatenate(all_features, axis=0)
    target = np.concatenate(all_targets, axis=0)
    hour = np.concatenate(all_hours, axis=0)
    
    return features, target, hour


# ============================================================
# SHAP 分析器
# ============================================================

class ShapAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def train_model(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """训练模型"""
        logger.info("训练模型...")
        
        dataset = TimeSeriesDataset(X, y, self.config.seq_len, self.config.pred_len)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        model = LSTMPredictor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.config.epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
        
        self.model = model
        return model
    
    def compute_shap(self, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        """计算 SHAP 值"""
        logger.info("计算 SHAP 值...")
        
        self.model.eval()
        
        # 创建解释器
        explainer = shap.DeepExplainer(
            self.model, 
            torch.FloatTensor(background).to(self.device)
        )
        
        # 计算 SHAP 值
        shap_values = explainer.shap_values(torch.FloatTensor(X).to(self.device))
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        return shap_values
    
    def compute_hourly_importance(self, shap_values: np.ndarray, 
                                   hour: np.ndarray) -> Dict[int, np.ndarray]:
        """计算每小时特征重要性"""
        hourly_importance = {}
        
        for h in range(24):
            mask = (hour >= h) & (hour < h + 1)
            if mask.sum() > 0:
                if shap_values.ndim == 3:
                    shap_hour = shap_values[:, mask, :]
                    importance = np.abs(shap_hour).mean(axis=(0, 1))
                else:
                    shap_hour = shap_values[mask, :]
                    importance = np.abs(shap_hour).mean(axis=0)
                hourly_importance[h] = importance
            else:
                hourly_importance[h] = np.zeros(self.config.input_dim)
        
        return hourly_importance


# ============================================================
# 可视化
# ============================================================

def plot_hourly_importance(hourly_importance: Dict[int, np.ndarray],
                            feature_names: List[str], output_path: Path,
                            data_type: str):
    """绘制每小时特征重要性热力图"""
    hours = list(range(24))
    importance_matrix = np.array([hourly_importance[h] for h in hours])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(importance_matrix.T, aspect='auto', cmap='YlOrRd')
    
    ax.set_xlabel('小时')
    ax.set_ylabel('特征')
    ax.set_title(f'{data_type.upper()} 基站 - 每小时特征重要性')
    ax.set_xticks(range(0, 24, 2))
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
    parser = argparse.ArgumentParser(description='SHAP 小时级特征重要性分析')
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
    logger.info(f"SHAP 小时级特征重要性分析 - {config.data_type.upper()}")
    logger.info("="*60)
    logger.info(f"  最大基站数: {config.max_stations}")
    logger.info(f"  每基站样本数: {config.samples_per_station}")
    
    # 加载数据
    logger.info("\n1. 加载数据...")
    X, y, hour = load_station_data(
        config.data_dir, config.data_type,
        config.max_stations, config.samples_per_station
    )
    logger.info(f"   样本数: {len(X):,}")
    logger.info(f"   特征维度: {X.shape[1]}")
    
    # 训练模型
    logger.info("\n2. 训练模型...")
    analyzer = ShapAnalyzer(config)
    model = analyzer.train_model(X, y)
    
    # 计算 SHAP
    logger.info("\n3. 计算 SHAP 值...")
    background = X[:config.shap_background_samples]
    test_samples = X[config.shap_background_samples:config.shap_background_samples + config.shap_test_samples]
    test_hour = hour[config.shap_background_samples:config.shap_background_samples + config.shap_test_samples]
    
    shap_values = analyzer.compute_shap(test_samples, background)
    
    # 计算每小时重要性
    logger.info("\n4. 计算每小时特征重要性...")
    hourly_importance = analyzer.compute_hourly_importance(shap_values, test_hour)
    
    # 可视化
    logger.info("\n5. 生成可视化...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    hourly_path = config.output_dir / f"hourly_importance_{config.data_type}_{timestamp}.png"
    plot_hourly_importance(hourly_importance, config.feature_names, hourly_path, config.data_type)
    
    # 保存结果
    results = {
        'config': asdict(config),
        'timestamp': timestamp,
        'hourly_importance': {str(k): v.tolist() for k, v in hourly_importance.items()},
        'feature_names': config.feature_names
    }
    
    result_path = config.output_dir / f"shap_results_{config.data_type}_{timestamp}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✅ 结果保存: {result_path}")
    
    # 打印关键时段
    logger.info("\n" + "="*50)
    logger.info("关键时段特征重要性")
    logger.info("="*50)
    for h in [8, 12, 18, 22]:
        if h in hourly_importance:
            imp = hourly_importance[h]
            logger.info(f"\n小时 {h}:")
            for name, val in zip(config.feature_names, imp):
                logger.info(f"  {name}: {val:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()
