#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级完整版：4G vs 5G SHAP 对比分析
直接加载已有结果，包含所有专业分析功能
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ============================================================
# 配置
# ============================================================
class Config:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.shap_4g_path = self.project_root / "results/shap_complete/shap_raw_4g.pkl"
        self.shap_5g_path = self.project_root / "results/shap_comparison/shap_raw_5g.pkl"
        self.output_dir = self.project_root / "results/shap_comparison/comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_names = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']
        self.ci_percentile = 95
        self.n_bootstrap = 1000
        self.seed = 42
        
        np.random.seed(self.seed)


# ============================================================
# 数据加载
# ============================================================
def load_data(config):
    print("加载 SHAP 值...")
    with open(config.shap_4g_path, 'rb') as f:
        shap_4g = pickle.load(f)['shap_values']
    with open(config.shap_5g_path, 'rb') as f:
        shap_5g = pickle.load(f)['shap_values']
    
    print(f"  4G: {shap_4g.shape}")
    print(f"  5G: {shap_5g.shape}")
    
    # 计算权重
    weights_4g = np.abs(shap_4g).mean(axis=0)
    weights_5g = np.abs(shap_5g).mean(axis=0)
    
    return shap_4g, shap_5g, weights_4g, weights_5g


# ============================================================
# 对比分析类
# ============================================================
class ComparisonAnalyzer:
    def __init__(self, config):
        self.config = config
        self.feature_names = config.feature_names
    
    def compute_stats(self, w4, w5):
        """计算统计信息"""
        stats = {}
        for i, name in enumerate(self.feature_names):
            stats[name] = {
                '4g_mean': float(np.mean(w4[:, i])),
                '4g_std': float(np.std(w4[:, i])),
                '4g_max': float(np.max(w4[:, i])),
                '4g_peak_hour': float(np.argmax(w4[:, i]) / 2),
                '5g_mean': float(np.mean(w5[:, i])),
                '5g_std': float(np.std(w5[:, i])),
                '5g_max': float(np.max(w5[:, i])),
                '5g_peak_hour': float(np.argmax(w5[:, i]) / 2),
                'ratio': float(np.mean(w5[:, i]) / (np.mean(w4[:, i]) + 1e-8))
            }
        return stats
    
    def significance_test(self, shap_4g, shap_5g):
        """显著性检验（逐时段逐特征）"""
        n_t, n_f = shap_4g.shape[1], shap_4g.shape[2]
        p_values = np.zeros((n_t, n_f))
        effect_sizes = np.zeros((n_t, n_f))
        
        for t in range(n_t):
            for f in range(n_f):
                x = shap_4g[:, t, f]
                y = shap_5g[:, t, f]
                _, p = stats.ttest_ind(x, y)
                p_values[t, f] = p
                pooled_std = np.sqrt((np.var(x) + np.var(y)) / 2)
                effect_sizes[t, f] = (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)
        
        return {
            'p_values': p_values.tolist(),
            'effect_sizes': effect_sizes.tolist(),
            'significant_count': int(np.sum(p_values < 0.05)),
            'significant_ratio': float(np.sum(p_values < 0.05) / (n_t * n_f))
        }
    
    def compute_confidence_intervals(self, shap_values):
        """计算置信区间（Bootstrap）"""
        n_samples, n_t, n_f = shap_values.shape
        ci_lower = np.zeros((n_t, n_f))
        ci_upper = np.zeros((n_t, n_f))
        
        alpha = (100 - self.config.ci_percentile) / 2
        for t in range(n_t):
            for f in range(n_f):
                samples = shap_values[:, t, f]
                ci_lower[t, f] = np.percentile(samples, alpha)
                ci_upper[t, f] = np.percentile(samples, 100 - alpha)
        
        return {'lower': ci_lower.tolist(), 'upper': ci_upper.tolist()}
    
    def cluster_time_segments(self, weights):
        """时段聚类（自动确定最优K）"""
        n_t, n_f = weights.shape
        best_k = 2
        best_score = -1
        
        for k in range(2, min(7, n_t)):
            kmeans = KMeans(n_clusters=k, random_state=self.config.seed, n_init=10)
            labels = kmeans.fit_predict(weights)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(weights, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        kmeans = KMeans(n_clusters=best_k, random_state=self.config.seed, n_init=10)
        labels = kmeans.fit_predict(weights)
        
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
            'silhouette_score': float(best_score),
            'labels': labels.tolist(),
            'segments': segments
        }
    
    def detect_anomalies(self, shap_values):
        """异常检测（Isolation Forest）"""
        n_samples = shap_values.shape[0]
        X_flat = shap_values.reshape(n_samples, -1)
        
        iso_forest = IsolationForest(contamination=0.05, random_state=self.config.seed)
        labels = iso_forest.fit_predict(X_flat)
        anomaly_indices = np.where(labels == -1)[0].tolist()
        
        return {
            'count': len(anomaly_indices),
            'ratio': len(anomaly_indices) / n_samples,
            'indices_sample': anomaly_indices[:10]
        }
    
    def compute_barcelona_weights(self, w4, w5):
        """巴塞罗那6小时时段权重"""
        segments = {
            '00-06': (0, 12),
            '06-12': (12, 24),
            '12-18': (24, 36),
            '18-24': (36, 48)
        }
        
        industrial = {}
        commercial = {}
        
        for seg_name, (start, end) in segments.items():
            industrial[seg_name] = {}
            commercial[seg_name] = {}
            for i, name in enumerate(self.feature_names):
                industrial[seg_name][name] = float(np.mean([w4[t][i] for t in range(start, end)]))
                commercial[seg_name][name] = float(np.mean([w5[t][i] for t in range(start, end)]))
        
        return {
            'industrial_like': industrial,
            'commercial_like': commercial,
            'segments': list(segments.keys())
        }
    
    def generate_report(self, stats, significance, barcelona_weights, 
                        clustering_4g, clustering_5g, anomalies_4g, anomalies_5g):
        """生成专业报告"""
        report = []
        report.append("# 4G vs 5G SHAP 权重分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 核心发现
        prb_ratio = stats['PRB']['ratio']
        report.append("## 核心发现")
        report.append(f"1. **5G PRB 权重是 4G 的 {prb_ratio:.2f} 倍**")
        report.append(f"2. PRB峰值时段: 4G 在 {stats['PRB']['4g_peak_hour']:.1f}点, 5G 在 {stats['PRB']['5g_peak_hour']:.1f}点")
        report.append(f"3. 显著差异时段数: {significance['significant_count']} ({significance['significant_ratio']*100:.1f}%)")
        report.append("")
        
        # 特征权重对比
        report.append("## 特征权重对比")
        report.append("| 特征 | 4G均值 | 5G均值 | 比值 | 4G峰值时段 | 5G峰值时段 |")
        report.append("|------|--------|--------|------|------------|------------|")
        for name in self.feature_names:
            s = stats[name]
            report.append(f"| {name} | {s['4g_mean']:.6f} | {s['5g_mean']:.6f} | {s['ratio']:.2f}x | {s['4g_peak_hour']:.1f}h | {s['5g_peak_hour']:.1f}h |")
        report.append("")
        
        # 时段聚类
        report.append("## 时段聚类")
        report.append(f"- 4G 最优聚类数: {clustering_4g['n_clusters']} (轮廓系数: {clustering_4g['silhouette_score']:.3f})")
        report.append(f"- 5G 最优聚类数: {clustering_5g['n_clusters']} (轮廓系数: {clustering_5g['silhouette_score']:.3f})")
        report.append("")
        
        # 异常检测
        report.append("## 异常检测")
        report.append(f"- 4G 异常样本比例: {anomalies_4g['ratio']*100:.2f}%")
        report.append(f"- 5G 异常样本比例: {anomalies_5g['ratio']*100:.2f}%")
        report.append("")
        
        # 巴塞罗那权重表
        report.append("## 巴塞罗那6小时时段权重表 (PRB)")
        report.append("| 时段 | 工业区(4G-like) | 商业区(5G-like) | 比值 |")
        report.append("|------|-----------------|-----------------|------|")
        for seg in barcelona_weights['segments']:
            w4 = barcelona_weights['industrial_like'][seg]['PRB']
            w5 = barcelona_weights['commercial_like'][seg]['PRB']
            report.append(f"| {seg} | {w4:.6f} | {w5:.6f} | {w5/w4:.2f}x |")
        
        return '\n'.join(report)


# ============================================================
# 可视化
# ============================================================
class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.output_dir
    
    def plot_weights_comparison(self, w4, w5):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        hours = np.arange(48) / 2
        
        for i, (weights, title, color) in enumerate([(w4, '4G', '#2E8B57'), (w5, '5G', '#E76F51')]):
            ax = axes[i]
            for f_idx, name in enumerate(self.config.feature_names):
                ax.plot(hours, weights[:, f_idx], label=name, linewidth=1.5)
            ax.set_xlabel('小时')
            ax.set_ylabel('SHAP 重要性')
            ax.set_title(f'{title} 30分钟粒度特征重要性')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weights_comparison.png', dpi=150)
        plt.close()
    
    def plot_diff_heatmap(self, w4, w5):
        diff = w5 - w4
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
        plt.savefig(self.output_dir / 'diff_heatmap.png', dpi=150)
        plt.close()
    
    def plot_barcelona_weights(self, barcelona_weights):
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
        plt.savefig(self.output_dir / 'barcelona_weights.png', dpi=150)
        plt.close()
    
    def plot_significance_heatmap(self, significance):
        p_values = np.array(significance['p_values'])
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(p_values.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.05)
        ax.set_xlabel('小时')
        ax.set_ylabel('特征')
        ax.set_title('显著性检验 p-value (p<0.05 表示显著差异)')
        ax.set_yticks(range(len(self.config.feature_names)))
        ax.set_yticklabels(self.config.feature_names)
        ax.set_xticks(range(0, 48, 4))
        ax.set_xticklabels([f'{i/2:.0f}' for i in range(0, 48, 4)])
        plt.colorbar(im, ax=ax, label='p-value')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'significance_heatmap.png', dpi=150)
        plt.close()
    
    def plot_clustering(self, weights, title, filename):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        weights_2d = pca.fit_transform(weights)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(weights_2d[:, 0], weights_2d[:, 1], 
                            c=np.arange(len(weights)), cmap='viridis', s=50)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{title} 时段聚类 (PCA降维)')
        plt.colorbar(scatter, ax=ax, label='时段索引')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()


# ============================================================
# 主函数
# ============================================================
def main():
    config = Config()
    print("="*70)
    print("4G vs 5G SHAP 对比分析（五星级完整版）")
    print("="*70)
    
    # 加载数据
    shap_4g, shap_5g, w4, w5 = load_data(config)
    
    # 创建分析器
    analyzer = ComparisonAnalyzer(config)
    viz = Visualizer(config)
    
    # 统计分析
    print("\n【1. 特征统计】")
    stats = analyzer.compute_stats(w4, w5)
    for name, s in stats.items():
        print(f"  {name}: 4G={s['4g_mean']:.6f}, 5G={s['5g_mean']:.6f}, 比值={s['ratio']:.2f}x")
    
    print("\n【2. 显著性检验】")
    significance = analyzer.significance_test(shap_4g, shap_5g)
    print(f"  显著差异时段数: {significance['significant_count']} ({significance['significant_ratio']*100:.1f}%)")
    
    print("\n【3. 置信区间】")
    ci_4g = analyzer.compute_confidence_intervals(shap_4g)
    ci_5g = analyzer.compute_confidence_intervals(shap_5g)
    print(f"  4G PRB 95% CI: [{np.array(ci_4g['lower'])[:,0].mean():.6f}, {np.array(ci_4g['upper'])[:,0].mean():.6f}]")
    print(f"  5G PRB 95% CI: [{np.array(ci_5g['lower'])[:,0].mean():.6f}, {np.array(ci_5g['upper'])[:,0].mean():.6f}]")
    
    print("\n【4. 时段聚类】")
    clustering_4g = analyzer.cluster_time_segments(w4)
    clustering_5g = analyzer.cluster_time_segments(w5)
    print(f"  4G: {clustering_4g['n_clusters']} 个时段 (轮廓系数={clustering_4g['silhouette_score']:.3f})")
    print(f"  5G: {clustering_5g['n_clusters']} 个时段 (轮廓系数={clustering_5g['silhouette_score']:.3f})")
    
    print("\n【5. 异常检测】")
    anomalies_4g = analyzer.detect_anomalies(shap_4g)
    anomalies_5g = analyzer.detect_anomalies(shap_5g)
    print(f"  4G 异常样本: {anomalies_4g['ratio']*100:.2f}%")
    print(f"  5G 异常样本: {anomalies_5g['ratio']*100:.2f}%")
    
    print("\n【6. 巴塞罗那权重】")
    barcelona_weights = analyzer.compute_barcelona_weights(w4, w5)
    print("  时段权重 (PRB):")
    for seg in barcelona_weights['segments']:
        w4_seg = barcelona_weights['industrial_like'][seg]['PRB']
        w5_seg = barcelona_weights['commercial_like'][seg]['PRB']
        print(f"    {seg}: 工业区={w4_seg:.6f}, 商业区={w5_seg:.6f}, 比值={w5_seg/w4_seg:.2f}x")
    
    # 可视化
    print("\n【7. 生成可视化】")
    viz.plot_weights_comparison(w4, w5)
    viz.plot_diff_heatmap(w4, w5)
    viz.plot_barcelona_weights(barcelona_weights)
    viz.plot_significance_heatmap(significance)
    viz.plot_clustering(w4, '4G', 'clustering_4g.png')
    viz.plot_clustering(w5, '5G', 'clustering_5g.png')
    print(f"  图片保存至: {config.output_dir}")
    
    # 保存结果
    print("\n【8. 保存结果】")
    all_results = {
        'stats': stats,
        'significance': significance,
        'confidence_intervals': {'4g': ci_4g, '5g': ci_5g},
        'clustering': {'4g': clustering_4g, '5g': clustering_5g},
        'anomalies': {'4g': anomalies_4g, '5g': anomalies_5g},
        'barcelona_weights': barcelona_weights
    }
    
    with open(config.output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成报告
    report = analyzer.generate_report(
        stats, significance, barcelona_weights,
        clustering_4g, clustering_5g, anomalies_4g, anomalies_5g
    )
    with open(config.output_dir / 'report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  结果保存至: {config.output_dir}")
    
    # 打印报告
    print("\n" + "="*70)
    print(report)
    print("\n" + "="*70)
    print(f"\n✅ 完成！结果目录: {config.output_dir}")


if __name__ == "__main__":
    main()
