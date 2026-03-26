#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP 特征重要性分析 - GradientExplainer (PyTorch 原生)
五星级专业版，支持 3D 输入，无需 TensorFlow
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
from dataclasses import dataclass, field, asdict
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
    data_type: str = '4g'                      # '4g' or '5g'
    max_stations: int = 50                     # 最大基站数（None表示全部）
    samples_per_station: int = 100             # 每基站样本数（None表示全部）
    
    # 模型配置
    seq_len: int = 24                          # 输入序列长度
    pred_len: int = 1                          # 预测长度
    input_dim: int = 5                         # 输入特征维度
    hidden_dim: int = 64                       # LSTM 隐藏层维度
    num_layers: int = 2                        # LSTM 层数
    dropout: float = 0.0                       # Dropout（SHAP 推理时关闭）
    
    # 训练配置
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    device: str = 'cpu'                        # 'cuda' or 'cpu'
    
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
        """初始化后处理"""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "processed" / "tsinghua_v2"
        if self.output_dir is None:
            self.output_dir = self.project_root / "results" / "shap_analysis"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # 检测设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 日志系统
# ============================================================

def setup_logger(name: str = __name__, config: Config = None) -> logging.Logger:
    """配置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
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
# 数据加载器
# ============================================================

class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def load_station_data(data_dir: Path, data_type: str,
                          max_stations: Optional[int] = None,
                          samples_per_station: Optional[int] = None) -> np.ndarray:
        """加载基站数据并创建序列"""
        station_dir = data_dir / data_type
        if not station_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {station_dir}")
        
        station_dirs = [d for d in station_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('station_')]
        
        if max_stations:
            station_dirs = station_dirs[:max_stations]
        
        all_features = []
        
        for s_dir in tqdm(station_dirs, desc=f"加载 {data_type} 数据"):
            with open(s_dir / 'data.pkl', 'rb') as f:
                data = pickle.load(f)
            
            features = data.get('X_train_norm', data.get('features_norm', None))
            if features is None:
                continue
            
            if samples_per_station:
                features = features[:samples_per_station]
            
            all_features.append(features)
        
        if not all_features:
            raise ValueError(f"没有成功加载任何数据: {station_dir}")
        
        X = np.concatenate(all_features, axis=0)
        
        # 创建序列 [n_samples, seq_len, input_dim]
        seq_len = 24
        X_seq = []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
        X_seq = np.array(X_seq, dtype=np.float32)
        
        # 目标：下一个时间点的第一个特征（PRB）
        y = X_seq[:, -1, 0:1]
        
        return X_seq, y


# ============================================================
# 模型定义
# ============================================================

class LSTMPredictor(nn.Module):
    """LSTM 预测模型"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============================================================
# SHAP 分析器
# ============================================================

class ShapAnalyzer:
    """SHAP 分析器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.model = None
        self.logger.info(f"设备: {self.device}")
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """训练模型"""
        self.logger.info("训练模型...")
        
        # 转换为 tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # 创建模型
        model = LSTMPredictor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练
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
    
    def compute_shap(self, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        """计算 SHAP 值"""
        self.logger.info("计算 SHAP 值...")
        
        self.model.eval()
        
        # 转换为 tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        background_tensor = torch.FloatTensor(background).to(self.device)
        
        # 创建 GradientExplainer
        explainer = shap.GradientExplainer(self.model, background_tensor)
        
        # 计算 SHAP 值
        shap_values = explainer.shap_values(X_tensor)
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        self.logger.info(f"  SHAP 值形状: {shap_values.shape}")
        
        return shap_values


# ============================================================
# 可视化
# ============================================================

class ShapVisualizer:
    """SHAP 可视化"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def plot_feature_importance(self, importance: np.ndarray, 
                                 feature_names: List[str],
                                 output_path: Path, data_type: str):
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
        self.logger.info(f"  保存: {output_path}")
        plt.close()
    
    def plot_hourly_importance(self, shap_values: np.ndarray,
                                feature_names: List[str],
                                output_path: Path, data_type: str):
        """绘制时间步特征重要性热力图"""
        # shap_values: [n_samples, seq_len, n_features]
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
        self.logger.info(f"  保存: {output_path}")
        plt.close()


# ============================================================
# 主流程
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
        self.logger.info(f"SHAP 特征重要性分析 - {self.config.data_type.upper()} (GradientExplainer)")
        self.logger.info("="*60)
        self.logger.info(f"配置:")
        self.logger.info(f"  数据目录: {self.config.data_dir}")
        self.logger.info(f"  数据类型: {self.config.data_type}")
        self.logger.info(f"  最大基站数: {self.config.max_stations if self.config.max_stations else '全部'}")
        self.logger.info(f"  每基站样本数: {self.config.samples_per_station if self.config.samples_per_station else '全部'}")
        self.logger.info(f"  训练轮数: {self.config.epochs}")
        self.logger.info(f"  设备: {self.config.device}")
        
        # 1. 加载数据
        self.logger.info("\n1. 加载数据...")
        X_seq, y = DataLoaderFactory.load_station_data(
            self.config.data_dir, self.config.data_type,
            self.config.max_stations, self.config.samples_per_station
        )
        self.logger.info(f"   序列样本数: {len(X_seq):,}")
        self.logger.info(f"   序列形状: {X_seq.shape}")
        
        # 2. 训练模型
        self.logger.info("\n2. 训练模型...")
        analyzer = ShapAnalyzer(self.config, self.logger)
        model = analyzer.train_model(X_seq, y)
        
        # 3. 计算 SHAP 值
        self.logger.info("\n3. 计算 SHAP 值...")
        background = X_seq[:self.config.shap_background_samples]
        test_samples = X_seq[self.config.shap_background_samples:
                             self.config.shap_background_samples + self.config.shap_test_samples]
        
        self.logger.info(f"   背景样本: {background.shape}")
        self.logger.info(f"   测试样本: {test_samples.shape}")
        
        shap_values = analyzer.compute_shap(test_samples, background)
        
        # 4. 计算特征重要性
        self.logger.info("\n4. 计算特征重要性...")
        importance = np.abs(shap_values).mean(axis=(0, 1))
        
        # 打印结果
        self.logger.info("\n" + "="*50)
        self.logger.info("特征重要性排序")
        self.logger.info("="*50)
        sorted_idx = np.argsort(importance)[::-1]
        for idx in sorted_idx:
            self.logger.info(f"  {self.config.feature_names[idx]}: {importance[idx]:.4f}")
        
        # 5. 可视化
        self.logger.info("\n5. 生成可视化...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualizer = ShapVisualizer(self.config, self.logger)
        
        # 条形图
        bar_path = self.config.output_dir / f"feature_importance_{self.config.data_type}_{timestamp}.png"
        visualizer.plot_feature_importance(
            importance, self.config.feature_names, bar_path, self.config.data_type
        )
        
        # 热力图
        heat_path = self.config.output_dir / f"hourly_importance_{self.config.data_type}_{timestamp}.png"
        visualizer.plot_hourly_importance(
            shap_values, self.config.feature_names, heat_path, self.config.data_type
        )
        
        # 6. 保存结果
        self.logger.info("\n6. 保存结果...")
        results = {
            'config': {
                'data_type': self.config.data_type,
                'max_stations': self.config.max_stations,
                'samples_per_station': self.config.samples_per_station,
                'epochs': self.config.epochs,
                'seed': self.config.seed
            },
            'timestamp': timestamp,
            'feature_importance': {
                name: float(imp) for name, imp in zip(self.config.feature_names, importance)
            },
            'sorted_features': [
                {self.config.feature_names[i]: float(importance[i])} for i in sorted_idx
            ],
            'shap_values_shape': list(shap_values.shape)
        }
        
        result_path = self.config.output_dir / f"shap_results_{self.config.data_type}_{timestamp}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"   保存: {result_path}")
        
        self.logger.info("\n" + "="*50)
        self.logger.info("分析完成")
        self.logger.info("="*50)
        self.logger.info(f"特征重要性排序: {results['sorted_features']}")
        
        return results


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='SHAP 特征重要性分析 - GradientExplainer (PyTorch 原生)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_type', type=str, default='4g',
                        choices=['4g', '5g'],
                        help='数据类型: 4g 或 5g (默认: 4g)')
    parser.add_argument('--max_stations', type=int, default=50,
                        help='最大基站数，0表示全部 (默认: 50)')
    parser.add_argument('--samples_per_station', type=int, default=100,
                        help='每基站样本数，0表示全部 (默认: 100)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数 (默认: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    config = Config(
        data_type=args.data_type,
        max_stations=args.max_stations if args.max_stations > 0 else None,
        samples_per_station=args.samples_per_station if args.samples_per_station > 0 else None,
        epochs=args.epochs,
        seed=args.seed
    )
    
    runner = ShapAnalysisRunner(config)
    results = runner.run()
    
    return 0


if __name__ == "__main__":
    exit(main())
