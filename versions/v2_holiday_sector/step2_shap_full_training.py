#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级方案C：全量训练 + 采样SHAP
完整版 - 包含所有专业分析模块
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
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')

# 设置中文字体
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
    
    data_type: str = '4g'
    seq_len: int = 48                    # 48个时间步 = 24小时
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 0.001
    patience: int = 5
    
    shap_sample_size: int = 10000        # SHAP采样样本数
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
            self.output_dir = self.project_root / "results" / "shap_full_training"
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
    def __init__(self, features, target, seq_len=48, pred_len=1):
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.target[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y.flatten())


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============================================================
# 数据加载器
# ============================================================

class DataLoaderFactory:
    @staticmethod
    def load_all_data(data_dir: Path, data_type: str, max_samples: int = None):
        """加载所有基站数据"""
        station_dir = data_dir / data_type
        all_X = []
        all_y = []
        
        stations = list(station_dir.iterdir())
        for s_dir in tqdm(stations, desc=f"加载 {data_type} 数据"):
            if s_dir.is_dir() and s_dir.name.startswith('station_'):
                with open(s_dir / 'data.pkl', 'rb') as f:
                    data = pickle.load(f)
                X = data['X_train_norm']
                y = data['y_train_norm']
                all_X.append(X)
                all_y.append(y)
        
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # 创建序列
        seq_len = 48
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        if max_samples:
            X_seq = X_seq[:max_samples]
            y_seq = y_seq[:max_samples]
        
        return X_seq, y_seq


# ============================================================
# 模型训练器
# ============================================================

class ModelTrainer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        self.logger.info(f"训练模型 - {self.config.data_type.upper()}")
        self.logger.info(f"  样本数: {len(X):,}")
        self.logger.info(f"  设备: {self.device}")
        
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
        history = {'train_loss': []}
        
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
            history['train_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.config.models_dir / f"model_{self.config.data_type}_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    self.logger.info(f"  早停于 epoch {epoch+1}")
                    break
        
        model.load_state_dict(torch.load(self.config.models_dir / f"model_{self.config.data_type}_best.pth"))
        
        # 保存训练历史
        with open(self.config.output_dir / f"training_history_{self.config.data_type}.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        return model


# ============================================================
# SHAP分析器
# ============================================================

class ShapAnalyzer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_shap(self, model: nn.Module, X: np.ndarray) -> Dict:
        """计算SHAP值并返回所有分析结果"""
        self.logger.info("计算 SHAP 值...")
        model.eval()
        model.to(self.device)
        
        # 采样
        n_total = len(X)
        sample_idx = np.random.choice(n_total, self.config.shap_sample_size, replace=False)
        X_sample = X[sample_idx]
        
        self.logger.info(f"  采样样本数: {len(X_sample):,}")
        
        # 背景样本
        background = X_sample[:self.config.shap_background_samples]
        
        # 计算SHAP值
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
        with open(self.config.output_dir / f"shap_raw_{self.config.data_type}.pkl", 'wb') as f:
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
# 统计分析器
# ============================================================

class StatisticalAnalyzer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results = {}
    
    def compute_confidence_intervals(self, shap_values: np.ndarray) -> Dict:
        """2.1 真实置信区间 (Bootstrap)"""
        self.logger.info("计算置信区间...")
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
    
    def compute_weights_30min(self, shap_values: np.ndarray) -> np.ndarray:
        """2.2 30分钟粒度全天权重"""
        self.logger.info("计算30分钟粒度权重...")
        return np.abs(shap_values).mean(axis=0)  # [48, 5]
    
    def detect_anomalies(self, shap_values: np.ndarray) -> Dict:
        """2.3 异常样本检测"""
        self.logger.info("检测异常样本...")
        n_samples, n_timesteps, n_features = shap_values.shape
        
        # 方法1：总SHAP值异常
        total_shap = np.abs(shap_values).sum(axis=(1, 2))
        z_score = (total_shap - total_shap.mean()) / (total_shap.std() + 1e-8)
        anomaly_by_total = np.where(np.abs(z_score) > 3)[0].tolist()
        
        # 方法2：孤立森林
        X_flat = shap_values.reshape(n_samples, -1)
        iso_forest = IsolationForest(contamination=0.05, random_state=self.config.seed)
        labels = iso_forest.fit_predict(X_flat)
        anomaly_by_iforest = np.where(labels == -1)[0].tolist()
        
        return {
            'by_total': anomaly_by_total,
            'by_iforest': anomaly_by_iforest,
            'union': list(set(anomaly_by_total) | set(anomaly_by_iforest))
        }
    
    def compute_interaction_effects(self, model: nn.Module, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        """2.4 特征交互效应"""
        self.logger.info("计算特征交互效应...")
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        X_tensor = torch.FloatTensor(X[:1000]).to(device)
        background_tensor = torch.FloatTensor(background[:100]).to(device)
        
        explainer = shap.GradientExplainer(model, background_tensor)
        interaction_values = explainer.shap_interaction_values(X_tensor)
        
        if isinstance(interaction_values, list):
            interaction_values = np.array(interaction_values)
        if interaction_values.ndim == 5:
            interaction_values = interaction_values.squeeze(-1)
        
        return interaction_values  # [n_samples, 48, 5, 5]
    
    def compute_hourly_variability(self, shap_values: np.ndarray) -> Dict:
        """2.5 小时级波动性"""
        self.logger.info("计算波动性...")
        n_samples, n_timesteps, n_features = shap_values.shape
        
        # 聚合到小时（每2个30分钟点取平均）
        n_hours = n_timesteps // 2
        shap_hourly = np.zeros((n_samples, n_hours, n_features))
        for h in range(n_hours):
            shap_hourly[:, h, :] = (shap_values[:, 2*h, :] + shap_values[:, 2*h+1, :]) / 2
        
        hourly_mean = np.abs(shap_hourly).mean(axis=0)
        hourly_std = np.abs(shap_hourly).std(axis=0)
        hourly_cv = hourly_std / (hourly_mean + 1e-8)
        
        stability = {
            'stable': np.where(hourly_cv < 0.3),
            'moderate': np.where((hourly_cv >= 0.3) & (hourly_cv < 0.6)),
            'unstable': np.where(hourly_cv >= 0.6)
        }
        
        return {
            'mean': hourly_mean,
            'std': hourly_std,
            'cv': hourly_cv,
            'stability': stability
        }
    
    def cluster_time_segments(self, weights_30min: np.ndarray) -> Dict:
        """2.6 时段聚类优化"""
        self.logger.info("时段聚类...")
        n_timesteps, n_features = weights_30min.shape
        
        # 确定最优聚类数
        best_k = 2
        best_score = -1
        for k in range(2, 9):
            kmeans = KMeans(n_clusters=k, random_state=self.config.seed, n_init=10)
            labels = kmeans.fit_predict(weights_30min)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(weights_30min, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # 最终聚类
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
                'center': kmeans.cluster_centers_[label].tolist(),
                'size': len(indices)
            })
        
        return {
            'n_clusters': best_k,
            'silhouette_score': best_score,
            'labels': labels.tolist(),
            'segments': segments
        }
    
    def compute_diff_30min(self, weights_4g: np.ndarray, weights_5g: np.ndarray) -> Dict:
        """2.7 逐半小时差异分析"""
        self.logger.info("计算逐半小时差异...")
        diff = weights_5g - weights_4g
        
        # 显著性检验（需要原始SHAP值，这里用权重近似）
        # 实际需要原始SHAP值做t检验
        p_values = np.ones_like(diff)
        
        return {
            'diff': diff,
            'p_values': p_values
        }
    
    def time_series_decomposition(self, weights_30min: np.ndarray) -> Dict:
        """2.8 时序分解 (STL)"""
        self.logger.info("时序分解...")
        n_timesteps, n_features = weights_30min.shape
        
        # 聚合到小时
        n_hours = n_timesteps // 2
        weights_hourly = np.zeros((n_hours, n_features))
        for h in range(n_hours):
            weights_hourly[h] = (weights_30min[2*h] + weights_30min[2*h+1]) / 2
        
        results = {}
        for f in range(n_features):
            try:
                stl = STL(weights_hourly[:, f], period=24)
                result = stl.fit()
                results[self.config.feature_names[f]] = {
                    'trend': result.trend.tolist(),
                    'seasonal': result.seasonal.tolist(),
                    'residual': result.resid.tolist()
                }
            except:
                results[self.config.feature_names[f]] = None
        
        return results
    
    def feature_ranking_dynamic(self, weights_30min: np.ndarray) -> Dict:
        """2.9 特征重要性动态排序"""
        self.logger.info("特征重要性排序...")
        n_timesteps, n_features = weights_30min.shape
        
        rankings = {}
        for t in range(n_timesteps):
            weights = weights_30min[t]
            sorted_idx = np.argsort(weights)[::-1]
            rankings[t] = [self.config.feature_names[i] for i in sorted_idx]
        
        return rankings
    
    def cross_segment_similarity(self, weights_30min: np.ndarray) -> Dict:
        """2.10 跨时段相似性"""
        self.logger.info("计算时段相似性...")
        similarity = cosine_similarity(weights_30min)
        
        # 找出最相似的时段
        most_similar = {}
        for t in range(len(weights_30min)):
            sim = similarity[t]
            sim[t] = -1
            top5 = np.argsort(sim)[-5:][::-1]
            most_similar[t] = top5.tolist()
        
        return {
            'similarity_matrix': similarity.tolist(),
            'most_similar': most_similar
        }
    
    def significance_test(self, shap_4g: np.ndarray, shap_5g: np.ndarray) -> Dict:
        """2.11 显著性检验"""
        self.logger.info("显著性检验...")
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
                
                # Cohen's d
                pooled_std = np.sqrt((np.var(x) + np.var(y)) / 2)
                effect_sizes[t, f] = (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)
        
        return {
            'p_values': p_values,
            'effect_sizes': effect_sizes
        }
    
    def effect_size_interpretation(self, effect_sizes: np.ndarray) -> Dict:
        """2.12 效应量解释"""
        interpretation = {
            'very_small': np.where(effect_sizes < 0.2),
            'small': np.where((effect_sizes >= 0.2) & (effect_sizes < 0.5)),
            'medium': np.where((effect_sizes >= 0.5) & (effect_sizes < 0.8)),
            'large': np.where(effect_sizes >= 0.8)
        }
        return interpretation
    
    def robustness_check(self, shap_values: np.ndarray) -> Dict:
        """2.13 稳健性检验"""
        self.logger.info("稳健性检验...")
        n_samples, n_timesteps, n_features = shap_values.shape
        
        # 不同采样比例
        ratios = [0.1, 0.2, 0.5, 0.8, 1.0]
        weights_by_ratio = {}
        
        for ratio in ratios:
            n = int(n_samples * ratio)
            idx = np.random.choice(n_samples, n, replace=False)
            weights_by_ratio[ratio] = np.abs(shap_values[idx]).mean(axis=0)
        
        # 计算相关性
        correlations = {}
        base = weights_by_ratio[1.0].flatten()
        for ratio in ratios:
            if ratio < 1.0:
                corr = np.corrcoef(base, weights_by_ratio[ratio].flatten())[0, 1]
                correlations[ratio] = corr
        
        return {
            'weights_by_ratio': {str(k): v.tolist() for k, v in weights_by_ratio.items()},
            'correlations': correlations
        }
    
    def generate_report(self, all_results: Dict, config: Config) -> str:
        """2.14 自动生成可解释性报告"""
        self.logger.info("生成报告...")
        
        report = []
        report.append("# 4G/5G 权重分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 核心发现
        weights_4g = all_results.get('weights_30min_4g')
        weights_5g = all_results.get('weights_30min_5g')
        if weights_4g is not None and weights_5g is not None:
            prb_ratio = weights_5g[:, 0].mean() / (weights_4g[:, 0].mean() + 1e-8)
            report.append("## 核心发现")
            report.append(f"1. 5G PRB 权重是 4G 的 {prb_ratio:.2f} 倍")
            
            peak_4g = np.argmax(weights_4g[:, 0]) / 2
            peak_5g = np.argmax(weights_5g[:, 0]) / 2
            report.append(f"2. 峰值时段: 4G 在 {peak_4g:.1f}点, 5G 在 {peak_5g:.1f}点")
        
        # 聚类结果
        clustering = all_results.get('clustering')
        if clustering:
            report.append(f"\n## 时段聚类")
            report.append(f"最优聚类数: {clustering['n_clusters']}")
            report.append(f"轮廓系数: {clustering['silhouette_score']:.4f}")
        
        # 异常检测
        anomalies = all_results.get('anomalies')
        if anomalies:
            report.append(f"\n## 异常检测")
            report.append(f"异常样本数: {len(anomalies.get('union', []))}")
        
        # 显著差异
        sig_test = all_results.get('significance_test')
        if sig_test:
            p_values = sig_test['p_values']
            sig_count = np.sum(p_values < 0.05)
            report.append(f"\n## 显著性检验")
            report.append(f"显著差异时段数: {sig_count}")
        
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
    
    def plot_weights_30min(self, weights: np.ndarray, title: str, filename: str):
        """绘制30分钟权重热力图"""
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(weights.T, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('时间步 (30分钟)')
        ax.set_ylabel('特征')
        ax.set_title(title)
        ax.set_yticks(range(len(self.config.feature_names)))
        ax.set_yticklabels(self.config.feature_names)
        
        # 添加小时刻度
        xticks = range(0, 48, 4)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{i/2:.0f}' for i in xticks])
        
        plt.colorbar(im, ax=ax, label='SHAP 重要性')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
        self.logger.info(f"  保存: {filename}")
    
    def plot_confidence_intervals(self, weights: np.ndarray, ci_lower: np.ndarray, 
                                   ci_upper: np.ndarray, filename: str):
        """绘制置信区间"""
        fig, ax = plt.subplots(figsize=(12, 6))
        hours = np.arange(48) / 2
        ax.plot(hours, weights[:, 0], 'b-', linewidth=2, label='PRB权重')
        ax.fill_between(hours, ci_lower[:, 0], ci_upper[:, 0], alpha=0.3, color='blue')
        ax.set_xlabel('小时')
        ax.set_ylabel('PRB 权重')
        ax.set_title('PRB权重及95%置信区间')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
    
    def plot_interaction_heatmap(self, interaction: np.ndarray, filename: str):
        """绘制交互效应热力图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        mean_interaction = np.mean(np.abs(interaction), axis=(0, 1))
        im = ax.imshow(mean_interaction, cmap='RdBu_r', vmin=-0.01, vmax=0.01)
        ax.set_xticks(range(len(self.config.feature_names)))
        ax.set_yticks(range(len(self.config.feature_names)))
        ax.set_xticklabels(self.config.feature_names)
        ax.set_yticklabels(self.config.feature_names)
        ax.set_title('特征交互效应')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
    
    def plot_clustering(self, weights: np.ndarray, labels: np.ndarray, filename: str):
        """绘制聚类结果"""
        fig, ax = plt.subplots(figsize=(12, 5))
        hours = np.arange(48) / 2
        colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
        for i, label in enumerate(np.unique(labels)):
            mask = labels == label
            ax.scatter(hours[mask], weights[mask, 0], c=[colors[i]], 
                      label=f'时段{i+1}', s=30, alpha=0.7)
        ax.plot(hours, weights[:, 0], 'k-', alpha=0.3, linewidth=1)
        ax.set_xlabel('小时')
        ax.set_ylabel('PRB 权重')
        ax.set_title('PRB权重时段聚类')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
    
    def plot_feature_ranking(self, rankings: Dict, filename: str):
        """绘制特征重要性排序热力图"""
        n_timesteps = len(rankings)
        ranking_matrix = np.zeros((n_timesteps, len(self.config.feature_names)))
        for t, ranked in rankings.items():
            for rank, feat in enumerate(ranked):
                ranking_matrix[t, self.config.feature_names.index(feat)] = rank + 1
        
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(ranking_matrix, aspect='auto', cmap='viridis_r')
        ax.set_xlabel('特征')
        ax.set_ylabel('时间步 (30分钟)')
        ax.set_title('特征重要性排序 (1=最重要)')
        ax.set_yticks(range(0, n_timesteps, 4))
        ax.set_yticklabels([f'{i/2:.0f}' for i in range(0, n_timesteps, 4)])
        ax.set_xticks(range(len(self.config.feature_names)))
        ax.set_xticklabels(self.config.feature_names)
        plt.colorbar(im, ax=ax, label='排名')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()


# ============================================================
# 主流程
# ============================================================

class ShapFullTraining:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__, config)
        
    def run(self):
        self.logger.info("="*70)
        self.logger.info("五星级方案C：全量训练 + 采样SHAP")
        self.logger.info(f"数据类型: {self.config.data_type.upper()}")
        self.logger.info("="*70)
        
        # 1. 加载数据
        self.logger.info("\n1. 加载数据...")
        X, y = DataLoaderFactory.load_all_data(
            self.config.data_dir, self.config.data_type,
            max_samples=None
        )
        self.logger.info(f"   样本数: {len(X):,}")
        self.logger.info(f"   形状: {X.shape}")
        
        # 2. 训练模型
        self.logger.info("\n2. 训练模型...")
        trainer = ModelTrainer(self.config, self.logger)
        model = trainer.train(X, y)
        
        # 3. 计算SHAP
        self.logger.info("\n3. 计算SHAP...")
        analyzer = ShapAnalyzer(self.config, self.logger)
        shap_result = analyzer.compute_shap(model, X)
        shap_values = shap_result['shap_values']
        
        # 4. 统计分析
        self.logger.info("\n4. 统计分析...")
        stat_analyzer = StatisticalAnalyzer(self.config, self.logger)
        
        # 4.1 置信区间
        ci = stat_analyzer.compute_confidence_intervals(shap_values)
        
        # 4.2 30分钟权重
        weights_30min = stat_analyzer.compute_weights_30min(shap_values)
        
        # 4.3 异常检测
        anomalies = stat_analyzer.detect_anomalies(shap_values)
        
        # 4.4 波动性
        variability = stat_analyzer.compute_hourly_variability(shap_values)
        
        # 4.5 时段聚类
        clustering = stat_analyzer.cluster_time_segments(weights_30min)
        
        # 4.6 特征排名
        rankings = stat_analyzer.feature_ranking_dynamic(weights_30min)
        
        # 4.7 时段相似性
        similarity = stat_analyzer.cross_segment_similarity(weights_30min)
        
        # 4.8 稳健性检验
        robustness = stat_analyzer.robustness_check(shap_values)
        
        # 5. 可视化
        self.logger.info("\n5. 生成可视化...")
        viz = Visualizer(self.config, self.logger)
        
        viz.plot_weights_30min(weights_30min, 
                               f'{self.config.data_type.upper()} 30分钟粒度权重',
                               f'weights_30min_{self.config.data_type}.png')
        
        viz.plot_confidence_intervals(weights_30min, ci['lower'], ci['upper'],
                                      f'confidence_intervals_{self.config.data_type}.png')
        
        viz.plot_clustering(weights_30min, np.array(clustering['labels']),
                            f'clustering_{self.config.data_type}.png')
        
        viz.plot_feature_ranking(rankings, f'feature_ranking_{self.config.data_type}.png')
        
        # 6. 保存所有结果
        self.logger.info("\n6. 保存结果...")
        
        results = {
            'data_type': self.config.data_type,
            'weights_30min': weights_30min.tolist(),
            'confidence_intervals': {
                'lower': ci['lower'].tolist(),
                'upper': ci['upper'].tolist()
            },
            'anomalies': {
                'count': len(anomalies['union']),
                'indices': anomalies['union']
            },
            'variability': {
                'cv': variability['cv'].tolist(),
                'stability': {k: v[0].tolist() if len(v[0]) > 0 else [] 
                             for k, v in variability['stability'].items()}
            },
            'clustering': clustering,
            'feature_rankings': {str(k): v for k, v in rankings.items()},
            'similarity': {
                'most_similar': {str(k): v for k, v in similarity['most_similar'].items()}
            },
            'robustness': robustness
        }
        
        with open(self.config.output_dir / f'analysis_results_{self.config.data_type}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # 7. 生成报告
        self.logger.info("\n7. 生成报告...")
        
        all_results = {
            'weights_30min_4g': weights_30min if self.config.data_type == '4g' else None,
            'weights_30min_5g': weights_30min if self.config.data_type == '5g' else None,
            'clustering': clustering,
            'anomalies': anomalies,
            'significance_test': None  # 需要两个数据集
        }
        
        report = stat_analyzer.generate_report(all_results, self.config)
        with open(self.config.output_dir / f'report_{self.config.data_type}.md', 'w') as f:
            f.write(report)
        
        self.logger.info(f"\n✅ 完成！结果保存至: {self.config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SHAP全量分析')
    parser.add_argument('--data_type', type=str, default='4g', choices=['4g', '5g'])
    parser.add_argument('--sample_size', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    config = Config(
        data_type=args.data_type,
        shap_sample_size=args.sample_size,
        epochs=args.epochs
    )
    
    runner = ShapFullTraining(config)
    runner.run()


if __name__ == "__main__":
    main()
