#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP 小时级特征重要性分析 - 五星级专业版
分析 4G 和 5G 基站数据的特征重要性，为后续文化迁移提供依据
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
from dataclasses import dataclass, asdict, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
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
    results_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    
    # 数据配置
    data_type: str = '4g'  # '4g' or '5g'
    max_stations: int = None  # 最大基站数（None表示全部）
    samples_per_station: int = None  # 每个基站取多少样本（None表示全部）
    
    # 模型配置
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.2
    
    # 训练配置
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    
    # SHAP配置
    shap_background_samples: int = 100  # 背景样本数
    shap_test_samples: int = 50         # 测试样本数
    
    # 特征名称
    feature_names: List[str] = field(default_factory=lambda: [
        'PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos'
    ])
    
    # 时段定义（小时）
    hour_segments: Dict[str, List[int]] = field(default_factory=lambda: {
        '凌晨(0-6)': list(range(0, 6)),
        '上午(6-12)': list(range(6, 12)),
        '下午(12-18)': list(range(12, 18)),
        '夜晚(18-24)': list(range(18, 24))
    })
    
    # 其他
    seed: int = 42
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """初始化后处理"""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "processed" / "tsinghua_v2"
        if self.output_dir is None:
            self.output_dir = self.project_root / "results" / "shap_analysis"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results" / "shap_analysis"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 固定随机种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


# ============================================================
# 日志系统
# ============================================================

def setup_logger(name: str = __name__, config: Config = None) -> logging.Logger:
    """配置日志器"""
    logger = logging.getLogger(name)
    level = getattr(logging, config.log_level if config else 'INFO')
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)
    
    # 文件处理器
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
    """LSTM预测模型"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, 
                 num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        return self.fc(out)


# ============================================================
# 数据加载器
# ============================================================

class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def load_station_data(data_dir: Path, data_type: str, 
                          max_stations: int = None,
                          samples_per_station: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载基站数据
        
        Args:
            data_dir: 数据目录
            data_type: '4g' 或 '5g'
            max_stations: 最大基站数（None表示全部）
            samples_per_station: 每个基站取多少样本（None表示全部）
        
        Returns:
            (features, target, hour) 特征矩阵、目标向量、小时标签
        """
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
            
            # 注意：字段名是 X_train_norm, y_train_norm
            features = data.get('X_train_norm', None)
            target = data.get('y_train_norm', None)
            
            # 兼容旧格式
            if features is None:
                features = data.get('features_norm', None)
            if target is None:
                target = data.get('target_norm', None)
            
            if features is None or target is None:
                continue
            
            if samples_per_station:
                features = features[:samples_per_station]
                target = target[:samples_per_station]
            
            # 提取小时（从特征中）
            # 特征: [PRB, Traffic, Users, Hour_sin, Hour_cos]
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
        
        if not all_features:
            raise ValueError(f"没有成功加载任何数据: {station_dir}")
        
        features = np.concatenate(all_features, axis=0)
        target = np.concatenate(all_targets, axis=0)
        hour = np.concatenate(all_hours, axis=0)
        
        return features, target, hour


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
        
    def train_model(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """训练模型"""
        self.logger.info("训练模型...")
        
        dataset = TimeSeriesDataset(X, y, seq_len=24, pred_len=1)
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
                self.logger.info(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
        
        self.model = model
        return model
    
    def compute_shap(self, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        """计算 SHAP 值"""
        self.logger.info("计算 SHAP 值...")
        
        # 转换为 tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        background_tensor = torch.FloatTensor(background).to(self.device)
        
        # 创建解释器
        self.model.eval()
        explainer = shap.DeepExplainer(self.model, background_tensor)
        
        # 计算 SHAP 值
        shap_values = explainer.shap_values(X_tensor)
        
        # 处理返回格式
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        return shap_values
    
    def compute_hourly_importance(self, shap_values: np.ndarray, 
                                   hour: np.ndarray) -> Dict[int, np.ndarray]:
        """计算每小时的特征重要性"""
        self.logger.info("计算每小时特征重要性...")
        
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
    
    def compute_segment_importance(self, shap_values: np.ndarray,
                                    hour: np.ndarray) -> Dict[str, np.ndarray]:
        """计算各时段的特征重要性"""
        self.logger.info("计算时段特征重要性...")
        
        segment_importance = {}
        
        for segment_name, hours in self.config.hour_segments.items():
            mask = np.isin(hour, hours)
            if mask.sum() > 0:
                if shap_values.ndim == 3:
                    shap_segment = shap_values[:, mask, :]
                    importance = np.abs(shap_segment).mean(axis=(0, 1))
                else:
                    shap_segment = shap_values[mask, :]
                    importance = np.abs(shap_segment).mean(axis=0)
                segment_importance[segment_name] = importance
            else:
                segment_importance[segment_name] = np.zeros(self.config.input_dim)
        
        return segment_importance


# ============================================================
# 可视化
# ============================================================

class ShapVisualizer:
    """SHAP 可视化"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def plot_hourly_importance(self, hourly_importance: Dict[int, np.ndarray],
                                feature_names: List[str], output_path: Path):
        """绘制每小时特征重要性热力图"""
        self.logger.info("绘制每小时特征重要性热力图...")
        
        # 构建矩阵
        hours = list(range(24))
        importance_matrix = np.array([hourly_importance[h] for h in hours])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(importance_matrix.T, aspect='auto', cmap='YlOrRd')
        
        ax.set_xlabel('小时')
        ax.set_ylabel('特征')
        ax.set_title(f'{self.config.data_type.upper()} 基站 - 每小时特征重要性')
        ax.set_xticks(range(0, 24, 2))
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        
        plt.colorbar(im, ax=ax, label='SHAP 重要性')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        self.logger.info(f"  保存: {output_path}")
        plt.close()
    
    def plot_segment_importance(self, segment_importance: Dict[str, np.ndarray],
                                 feature_names: List[str], output_path: Path):
        """绘制时段特征重要性条形图"""
        self.logger.info("绘制时段特征重要性条形图...")
        
        segments = list(segment_importance.keys())
        n_segments = len(segments)
        n_features = len(feature_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_features)
        width = 0.2
        
        colors = ['#2E8B57', '#2E86AB', '#F4A261', '#E76F51']
        
        for i, (segment, importance) in enumerate(segment_importance.items()):
            ax.bar(x + i * width, importance, width, label=segment, color=colors[i % len(colors)])
        
        ax.set_xlabel('特征')
        ax.set_ylabel('SHAP 重要性')
        ax.set_title(f'{self.config.data_type.upper()} 基站 - 各时段特征重要性')
        ax.set_xticks(x + width * (n_segments - 1) / 2)
        ax.set_xticklabels(feature_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        self.logger.info(f"  保存: {output_path}")
        plt.close()
    
    def plot_importance_comparison(self, importance_4g: Dict, importance_5g: Dict,
                                    feature_names: List[str], output_path: Path):
        """绘制4G和5G对比图"""
        self.logger.info("绘制4G vs 5G 对比图...")
        
        segments = list(importance_4g.keys())
        n_segments = len(segments)
        n_features = len(feature_names)
        
        fig, axes = plt.subplots(n_segments, 1, figsize=(10, 12))
        if n_segments == 1:
            axes = [axes]
        
        for i, segment in enumerate(segments):
            ax = axes[i]
            imp_4g = importance_4g[segment]
            imp_5g = importance_5g[segment]
            
            x = np.arange(n_features)
            width = 0.35
            
            ax.bar(x - width/2, imp_4g, width, label='4G', color='#2E8B57')
            ax.bar(x + width/2, imp_5g, width, label='5G', color='#E76F51')
            
            ax.set_xlabel('特征')
            ax.set_ylabel('SHAP 重要性')
            ax.set_title(f'{segment}')
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        self.logger.info(f"  保存: {output_path}")
        plt.close()


# ============================================================
# 主分析流程
# ============================================================

class ShapAnalysisRunner:
    """SHAP 分析运行器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__, config)
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """运行 SHAP 分析"""
        self.logger.info("="*60)
        self.logger.info(f"SHAP 小时级特征重要性分析 - {self.config.data_type.upper()}")
        self.logger.info("="*60)
        self.logger.info(f"配置:")
        self.logger.info(f"  数据目录: {self.config.data_dir}")
        self.logger.info(f"  数据类型: {self.config.data_type}")
        self.logger.info(f"  最大基站数: {self.config.max_stations if self.config.max_stations else '全部'}")
        self.logger.info(f"  每基站样本数: {self.config.samples_per_station if self.config.samples_per_station else '全部'}")
        self.logger.info(f"  训练轮数: {self.config.epochs}")
        
        # 加载数据
        self.logger.info("\n1. 加载数据...")
        X, y, hour = DataLoaderFactory.load_station_data(
            self.config.data_dir, self.config.data_type,
            self.config.max_stations, self.config.samples_per_station
        )
        self.logger.info(f"   样本数: {len(X):,}")
        self.logger.info(f"   特征维度: {X.shape[1]}")
        
        # 训练模型
        self.logger.info("\n2. 训练模型...")
        analyzer = ShapAnalyzer(self.config, self.logger)
        model = analyzer.train_model(X, y)
        
        # 计算 SHAP
        self.logger.info("\n3. 计算 SHAP 值...")
        background = X[:self.config.shap_background_samples]
        test_samples = X[self.config.shap_background_samples:
                         self.config.shap_background_samples + self.config.shap_test_samples]
        
        shap_values = analyzer.compute_shap(test_samples, background)
        
        # 计算每小时重要性
        self.logger.info("\n4. 计算每小时特征重要性...")
        test_hour = hour[self.config.shap_background_samples:
                         self.config.shap_background_samples + self.config.shap_test_samples]
        hourly_importance = analyzer.compute_hourly_importance(shap_values, test_hour)
        
        # 计算时段重要性
        self.logger.info("\n5. 计算时段特征重要性...")
        segment_importance = analyzer.compute_segment_importance(shap_values, test_hour)
        
        # 可视化
        self.logger.info("\n6. 生成可视化...")
        visualizer = ShapVisualizer(self.config, self.logger)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        hourly_path = self.config.output_dir / f"hourly_importance_{self.config.data_type}_{timestamp}.png"
        visualizer.plot_hourly_importance(hourly_importance, self.config.feature_names, hourly_path)
        
        segment_path = self.config.output_dir / f"segment_importance_{self.config.data_type}_{timestamp}.png"
        visualizer.plot_segment_importance(segment_importance, self.config.feature_names, segment_path)
        
        # 保存结果
        self.logger.info("\n7. 保存结果...")
        results = {
            'config': asdict(self.config),
            'timestamp': timestamp,
            'hourly_importance': {str(k): v.tolist() for k, v in hourly_importance.items()},
            'segment_importance': {k: v.tolist() for k, v in segment_importance.items()},
            'feature_names': self.config.feature_names,
            'hour_segments': self.config.hour_segments
        }
        
        result_path = self.config.output_dir / f"shap_results_{self.config.data_type}_{timestamp}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"   保存: {result_path}")
        
        # 打印时段重要性
        self.logger.info("\n" + "="*50)
        self.logger.info("时段特征重要性汇总")
        self.logger.info("="*50)
        for segment, importance in segment_importance.items():
            self.logger.info(f"\n{segment}:")
            for name, imp in zip(self.config.feature_names, importance):
                self.logger.info(f"  {name}: {imp:.4f}")
        
        self.results = results
        return results


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SHAP 小时级特征重要性分析')
    parser.add_argument('--data_type', type=str, default='4g', choices=['4g', '5g'],
                        help='数据类型: 4g 或 5g')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='最大基站数（默认全部）')
    parser.add_argument('--samples_per_station', type=int, default=None,
                        help='每个基站取多少样本（默认全部）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='日志级别')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    config = Config(
        data_type=args.data_type,
        max_stations=args.max_stations,
        samples_per_station=args.samples_per_station,
        epochs=args.epochs,
        seed=args.seed,
        log_level=args.log_level
    )
    
    runner = ShapAnalysisRunner(config)
    results = runner.run()
    
    return 0


if __name__ == "__main__":
    exit(main())
