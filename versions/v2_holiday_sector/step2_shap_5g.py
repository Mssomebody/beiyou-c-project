#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版：4G和5G合并分析（合理采样）
- 训练样本：100,000（足够模型收敛）
- SHAP样本：10,000（置信区间稳定）
- 预计时间：2-3小时
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
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


# ============================================================
# 配置管理
# ============================================================

@dataclass
class Config:
    """集中配置管理"""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    data_type: str = "5g"

    seq_len: int = 48                    # 48个时间步 = 24小时
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 0.001
    patience: int = 5
    
    # 合理采样（不是全量）
    train_samples: int = 100000          # 训练用 10万样本（足够收敛）
    shap_samples: int = 10000            # SHAP用 1万样本（稳定）
    shap_background_samples: int = 100
    n_bootstrap: int = 1000
    ci_percentile: int = 95
    
    feature_names: List[str] = field(default_factory=lambda: [
        'PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos'
    ])
    
    seed: int = 42
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "processed" / "tsinghua_full"
        if self.output_dir is None:
            self.output_dir = self.project_root / "results" / "shap_comparison"
        if self.models_dir is None:
            self.models_dir = self.output_dir / "models"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


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
        log_file = config.logs_dir / f"shap_comparison_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# 数据集和模型
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, features, target, seq_len=48, pred_len=1):
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============================================================
# 数据加载器（采样）
# ============================================================

def load_sampled_data(data_dir: Path, data_type: str, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """加载采样数据"""
    station_dir = data_dir / data_type
    all_X = []
    all_y = []
    
    stations = list(station_dir.iterdir())
    np.random.shuffle(stations)
    
    total_loaded = 0
    for s_dir in stations:
        if total_loaded >= n_samples:
            break
        if s_dir.is_dir() and s_dir.name.startswith('station_'):
            with open(s_dir / 'data.pkl', 'rb') as f:
                data = pickle.load(f)
            X = data['X_train_norm']
            y = data['y_train_norm']
            
            need = n_samples - total_loaded
            X = X[:need]
            y = y[:need]
            
            all_X.append(X)
            all_y.append(y)
            total_loaded += len(X)
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # 创建序列
    seq_len = 48
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    
    return np.array(X_seq), np.array(y_seq)


# ============================================================
# 模型训练器
# ============================================================

class ModelTrainer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, X: np.ndarray, y: np.ndarray, data_type: str) -> nn.Module:
        self.logger.info(f"训练模型 - {data_type.upper()}")
        self.logger.info(f"  样本数: {len(X):,}")
        
        dataset = TimeSeriesDataset(X, y, seq_len=self.config.seq_len)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        model = LSTMPredictor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.config.models_dir / f"model_{data_type}_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    self.logger.info(f"  早停于 epoch {epoch+1}")
                    break
        
        model.load_state_dict(torch.load(self.config.models_dir / f"model_{data_type}_best.pth"))
        return model


# ============================================================
# SHAP分析器
# ============================================================

class ShapAnalyzer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_shap(self, model: nn.Module, X: np.ndarray, data_type: str) -> Dict:
        """计算SHAP值"""
        self.logger.info(f"计算 SHAP 值 - {data_type.upper()}")
        model.eval()
        model.to(self.device)
        
        n_total = len(X)
        sample_idx = np.random.choice(n_total, self.config.shap_samples, replace=False)
        X_sample = X[sample_idx]
        
        self.logger.info(f"  采样样本数: {len(X_sample):,}")
        
        background = X_sample[:self.config.shap_background_samples]
        
        X_tensor = torch.FloatTensor(X_sample).to(self.device)
        background_tensor = torch.FloatTensor(background).to(self.device)
        
        explainer = shap.GradientExplainer(model, background_tensor)
        shap_values = explainer.shap_values(X_tensor)
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        if shap_values.ndim == 4:
            shap_values = shap_values.squeeze(-1)
        
        self.logger.info(f"  SHAP形状: {shap_values.shape}")
        
        # 保存原始SHAP值
        with open(self.config.output_dir / f"shap_raw_{data_type}.pkl", 'wb') as f:
            pickle.dump({
                'shap_values': shap_values,
                'X_sample': X_sample,
                'indices': sample_idx
            }, f)
        
        return {
            'shap_values': shap_values,
            'X_sample': X_sample,
            'indices': sample_idx
        }


# ============================================================
# 对比分析器
# ============================================================

class ComparisonAnalyzer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def compute_weights_30min(self, shap_values: np.ndarray) -> np.ndarray:
        return np.abs(shap_values).mean(axis=0)
    
    def compute_confidence_intervals(self, shap_values: np.ndarray) -> Dict:
        n_samples, n_timesteps, n_features = shap_values.shape
        ci_lower = np.zeros((n_timesteps, n_features))
        ci_upper = np.zeros((n_timesteps, n_features))
        
        for t in tqdm(range(n_timesteps), desc="Bootstrap"):
            for f in range(n_features):
                samples = shap_values[:, t, f]
                bootstrap_means = []
                for _ in range(self.config.n_bootstrap):
                    resample = np.random.choice(samples, len(samples), replace=True)
                    bootstrap_means.append(np.mean(resample))
                alpha = (100 - self.config.ci_percentile) / 2
                ci_lower[t, f] = np.percentile(bootstrap_means, alpha)
                ci_upper[t, f] = np.percentile(bootstrap_means, 100 - alpha)
        
        return {'lower': ci_lower, 'upper': ci_upper}
    
    def detect_anomalies(self, shap_values: np.ndarray) -> Dict:
        n_samples = shap_values.shape[0]
        X_flat = shap_values.reshape(n_samples, -1)
        
        iso_forest = IsolationForest(contamination=0.05, random_state=self.config.seed)
        labels = iso_forest.fit_predict(X_flat)
        anomaly_indices = np.where(labels == -1)[0].tolist()
        
        return {
            'count': len(anomaly_indices),
            'indices': anomaly_indices,
            'ratio': len(anomaly_indices) / n_samples
        }
    
    def compute_hourly_variability(self, shap_values: np.ndarray) -> Dict:
        n_samples, n_timesteps, n_features = shap_values.shape
        n_hours = n_timesteps // 2
        shap_hourly = np.zeros((n_samples, n_hours, n_features))
        for h in range(n_hours):
            shap_hourly[:, h, :] = (shap_values[:, 2*h, :] + shap_values[:, 2*h+1, :]) / 2
        
        hourly_mean = np.abs(shap_hourly).mean(axis=0)
        hourly_std = np.abs(shap_hourly).std(axis=0)
        hourly_cv = hourly_std / (hourly_mean + 1e-8)
        
        return {
            'mean': hourly_mean.tolist(),
            'std': hourly_std.tolist(),
            'cv': hourly_cv.tolist()
        }
    
    def cluster_time_segments(self, weights_30min: np.ndarray) -> Dict:
        n_timesteps, n_features = weights_30min.shape
        
        best_k = 2
        best_score = -1
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=self.config.seed, n_init=10)
            labels = kmeans.fit_predict(weights_30min)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(weights_30min, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        kmeans = KMeans(n_clusters=best_k, random_state=self.config.seed, n_init=10)
        labels = kmeans.fit_predict(weights_30min)
        
        segments = []
        for label in range(best_k):
            indices = np.where(labels == label)[0]
            start_hour = indices[0] / 2
            end_hour = (indices[-1] + 1) / 2
            segments.append({
                'label': label,
                'hours': indices.tolist(),
                'time_range': f'{start_hour:.1f}-{end_hour:.1f}',
                'center': kmeans.cluster_centers_[label].tolist()
            })
        
        return {
            'n_clusters': best_k,
            'silhouette_score': best_score,
            'labels': labels.tolist(),
            'segments': segments
        }
    
    def significance_test(self, shap_4g: np.ndarray, shap_5g: np.ndarray) -> Dict:
        n_timesteps = shap_4g.shape[1]
        n_features = shap_4g.shape[2]
        
        p_values = np.zeros((n_timesteps, n_features))
        effect_sizes = np.zeros((n_timesteps, n_features))
        
        for t in range(n_timesteps):
            for f in range(n_features):
                x = shap_4g[:, t, f]
                y = shap_5g[:, t, f]
                _, p = stats.ttest_ind(x, y)
                p_values[t, f] = p
                
                pooled_std = np.sqrt((np.var(x) + np.var(y)) / 2)
                effect_sizes[t, f] = (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)
        
        return {
            'p_values': p_values.tolist(),
            'effect_sizes': effect_sizes.tolist(),
            'significant_count': int(np.sum(p_values < 0.05))
        }
    
    def compute_barcelona_weights(self, weights_4g: np.ndarray, weights_5g: np.ndarray) -> Dict:
        segments = {
            '00-06': list(range(0, 12)),
            '06-12': list(range(12, 24)),
            '12-18': list(range(24, 36)),
            '18-24': list(range(36, 48))
        }
        
        industrial = {}
        commercial = {}
        
        for seg_name, indices in segments.items():
            industrial[seg_name] = {
                self.config.feature_names[i]: float(np.mean([weights_4g[t][i] for t in indices]))
                for i in range(len(self.config.feature_names))
            }
            commercial[seg_name] = {
                self.config.feature_names[i]: float(np.mean([weights_5g[t][i] for t in indices]))
                for i in range(len(self.config.feature_names))
            }
        
        return {
            'industrial_like': industrial,
            'commercial_like': commercial,
            'segments': list(segments.keys())
        }
    
    def generate_report(self, results_4g: Dict, results_5g: Dict, 
                        comparison: Dict, barcelona_weights: Dict) -> str:
        report = []
        report.append("# 4G vs 5G SHAP 权重分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        weights_4g = np.array(results_4g['weights_30min'])
        weights_5g = np.array(results_5g['weights_30min'])
        
        prb_ratio = weights_5g[:, 0].mean() / (weights_4g[:, 0].mean() + 1e-8)
        report.append("## 核心发现")
        report.append(f"1. 5G PRB 权重是 4G 的 {prb_ratio:.2f} 倍")
        
        peak_4g = np.argmax(weights_4g[:, 0]) / 2
        peak_5g = np.argmax(weights_5g[:, 0]) / 2
        report.append(f"2. 峰值时段: 4G 在 {peak_4g:.1f}点, 5G 在 {peak_5g:.1f}点")
        
        report.append(f"\n## 时段聚类")
        report.append(f"4G 最优聚类数: {results_4g['clustering']['n_clusters']}")
        report.append(f"5G 最优聚类数: {results_5g['clustering']['n_clusters']}")
        
        report.append(f"\n## 异常检测")
        report.append(f"4G 异常样本比例: {results_4g['anomalies']['ratio']*100:.2f}%")
        report.append(f"5G 异常样本比例: {results_5g['anomalies']['ratio']*100:.2f}%")
        
        report.append(f"\n## 4G vs 5G 差异显著性")
        report.append(f"显著差异时段数: {comparison['significance']['significant_count']}")
        
        report.append(f"\n## 巴塞罗那权重表 (PRB)")
        for seg in barcelona_weights['segments']:
            w4 = barcelona_weights['industrial_like'][seg]['PRB']
            w5 = barcelona_weights['commercial_like'][seg]['PRB']
            report.append(f"  {seg}: 工业区={w4:.6f}, 商业区={w5:.6f}, 比值={w5/w4:.2f}x")
        
        return '\n'.join(report)


# ============================================================
# 可视化器
# ============================================================

class Visualizer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = config.output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_weights_comparison(self, weights_4g: np.ndarray, weights_5g: np.ndarray, filename: str):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        hours = np.arange(48) / 2
        
        for i, (weights, title, color) in enumerate([
            (weights_4g, '4G', '#2E8B57'),
            (weights_5g, '5G', '#E76F51')
        ]):
            ax = axes[i]
            for f_idx, name in enumerate(self.config.feature_names):
                ax.plot(hours, weights[:, f_idx], label=name, linewidth=1.5)
            ax.set_xlabel('小时')
            ax.set_ylabel('SHAP 重要性')
            ax.set_title(f'{title} 30分钟粒度特征重要性')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
    
    def plot_diff_heatmap(self, diff: np.ndarray, filename: str):
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-0.05, vmax=0.05)
        ax.set_xlabel('小时')
        ax.set_ylabel('特征')
        ax.set_title('5G - 4G 权重差异')
        ax.set_yticks(range(len(self.config.feature_names)))
        ax.set_yticklabels(self.config.feature_names)
        ax.set_xticks(range(0, 48, 4))
        ax.set_xticklabels([f'{i/2:.0f}' for i in range(0, 48, 4)])
        plt.colorbar(im, ax=ax, label='差异')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
    
    def plot_barcelona_weights(self, barcelona_weights: Dict, filename: str):
        segments = barcelona_weights['segments']
        industrial = [barcelona_weights['industrial_like'][s]['PRB'] for s in segments]
        commercial = [barcelona_weights['commercial_like'][s]['PRB'] for s in segments]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(segments))
        width = 0.35
        ax.bar(x - width/2, industrial, width, label='工业区 (4G-like)', color='#2E8B57')
        ax.bar(x + width/2, commercial, width, label='商业区 (5G-like)', color='#E76F51')
        ax.set_xlabel('时段')
        ax.set_ylabel('PRB 权重')
        ax.set_title('巴塞罗那6小时时段权重')
        ax.set_xticks(x)
        ax.set_xticklabels(segments)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()


# ============================================================
# 主流程
# ============================================================

def process_data_type(config: Config, data_type: str, logger: logging.Logger) -> Dict:
    """处理单个数据类型"""
    logger.info(f"\n{'='*60}")
    logger.info(f"处理 {data_type.upper()} 数据")
    logger.info(f"{'='*60}")
    
    # 加载采样数据
    logger.info(f"加载采样数据 ({config.train_samples:,} 样本)...")
    X, y = load_sampled_data(config.data_dir, data_type, config.train_samples)
    logger.info(f"  样本数: {len(X):,}")
    
    # 训练模型
    logger.info("训练模型...")
    trainer = ModelTrainer(config, logger)
    model = trainer.train(X, y, data_type)
    
    # 计算SHAP
    logger.info("计算SHAP...")
    analyzer = ShapAnalyzer(config, logger)
    shap_result = analyzer.compute_shap(model, X, data_type)
    shap_values = shap_result['shap_values']
    
    # 分析
    comp_analyzer = ComparisonAnalyzer(config, logger)
    
    weights = comp_analyzer.compute_weights_30min(shap_values)
    ci = comp_analyzer.compute_confidence_intervals(shap_values)
    anomalies = comp_analyzer.detect_anomalies(shap_values)
    variability = comp_analyzer.compute_hourly_variability(shap_values)
    clustering = comp_analyzer.cluster_time_segments(weights)
    
    results = {
        'data_type': data_type,
        'weights_30min': weights.tolist(),
        'confidence_intervals': ci,
        'anomalies': anomalies,
        'variability': variability,
        'clustering': clustering,
        'model': model,
        'shap_values': shap_values
    }
    
    with open(config.output_dir / f"results_{data_type}.json", 'w') as f:
        json.dump({k: v for k, v in results.items() if k not in ['model', 'shap_values']}, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='4G和5G合并SHAP分析（优化版）')
    parser.add_argument('--train_samples', type=int, default=100000, help='训练样本数')
    parser.add_argument('--shap_samples', type=int, default=10000, help='SHAP样本数')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    args = parser.parse_args()
    
    config = Config(
        train_samples=args.train_samples,
        shap_samples=args.shap_samples,
        epochs=args.epochs
    )
    
    logger = setup_logger(__name__, config)
    
    logger.info("="*70)
    logger.info("4G和5G合并SHAP分析（优化版）")
    logger.info("="*70)
    logger.info(f"训练样本数: {config.train_samples:,}")
    logger.info(f"SHAP样本数: {config.shap_samples:,}")
    logger.info(f"预计时间: 2-3 小时")
    logger.info("="*70)
    
    # 分别处理4G和5G
    # results_4g = process_data_type(config, '4g', logger)  # 4G 已跑过，跳过
    results_5g = process_data_type(config, '5g', logger)
    
    # 对比分析
    logger.info("\n" + "="*60)
    logger.info("4G vs 5G 对比分析")
    logger.info("="*60)
    
    comp_analyzer = ComparisonAnalyzer(config, logger)
    
    # 显著性检验
    significance = comp_analyzer.significance_test(
        results_4g['shap_values'], 
        results_5g['shap_values']
    )
    
    # 巴塞罗那权重表
    barcelona_weights = comp_analyzer.compute_barcelona_weights(
        np.array(results_4g['weights_30min']),
        np.array(results_5g['weights_30min'])
    )
    
    # 可视化
    logger.info("生成可视化...")
    viz = Visualizer(config, logger)
    
    viz.plot_weights_comparison(
        np.array(results_4g['weights_30min']),
        np.array(results_5g['weights_30min']),
        'weights_comparison.png'
    )
    
    diff = np.array(results_5g['weights_30min']) - np.array(results_4g['weights_30min'])
    viz.plot_diff_heatmap(diff, 'diff_heatmap.png')
    
    viz.plot_barcelona_weights(barcelona_weights, 'barcelona_weights.png')
    
    # 保存对比结果
    comparison_results = {
        'significance': significance,
        'barcelona_weights': barcelona_weights,
        'diff_30min': diff.tolist()
    }
    
    with open(config.output_dir / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # 生成报告
    report = comp_analyzer.generate_report(
        results_4g, results_5g, 
        comparison_results, barcelona_weights
    )
    
    with open(config.output_dir / 'report.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n报告内容:\n{report}")
    logger.info(f"\n✅ 全部完成！结果保存至: {config.output_dir}")
    
    # 打印巴塞罗那权重表
    print("\n" + "="*70)
    print("巴塞罗那6小时时段权重表 (PRB)")
    print("="*70)
    print(f"{'时段':<10} {'工业区(4G-like)':>18} {'商业区(5G-like)':>18} {'比值':>10}")
    print("-"*60)
    for seg in barcelona_weights['segments']:
        w4 = barcelona_weights['industrial_like'][seg]['PRB']
        w5 = barcelona_weights['commercial_like'][seg]['PRB']
        print(f"{seg:<10} {w4:>18.6f} {w5:>18.6f} {w5/w4:>9.2f}x")


if __name__ == "__main__":
    main()
