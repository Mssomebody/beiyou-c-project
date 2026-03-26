#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接加载4G和5G的SHAP结果进行对比分析
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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ============================================================
# 配置
# ============================================================
RESULTS_DIR = Path(__file__).parent.parent.parent / "results/shap_comparison"
OUTPUT_DIR = RESULTS_DIR / 'comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

feature_names = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']

print("="*70)
print("4G vs 5G SHAP 对比分析")
print("="*70)

# ============================================================
# 加载数据
# ============================================================
print("\n1. 加载4G结果...")
with open(RESULTS_DIR / 'results_4g.json', 'r') as f:
    results_4g = json.load(f)
weights_4g = np.array(results_4g['weights_30min'])
print(f"   4G权重形状: {weights_4g.shape}")

with open(RESULTS_DIR / 'shap_raw_4g.pkl', 'rb') as f:
    shap_4g_data = pickle.load(f)
shap_4g = shap_4g_data['shap_values']
print(f"   4G SHAP形状: {shap_4g.shape}")

print("\n2. 加载5G结果...")
with open(RESULTS_DIR / 'results_5g.json', 'r') as f:
    results_5g = json.load(f)
weights_5g = np.array(results_5g['weights_30min'])
print(f"   5G权重形状: {weights_5g.shape}")

with open(RESULTS_DIR / 'shap_raw_5g.pkl', 'rb') as f:
    shap_5g_data = pickle.load(f)
shap_5g = shap_5g_data['shap_values']
print(f"   5G SHAP形状: {shap_5g.shape}")

# ============================================================
# 对比分析类
# ============================================================
class ComparisonAnalyzer:
    def __init__(self):
        self.feature_names = feature_names
    
    def significance_test(self, shap_4g, shap_5g):
        """显著性检验"""
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
    
    def compute_barcelona_weights(self, weights_4g, weights_5g):
        """计算巴塞罗那6小时时段权重"""
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
                self.feature_names[i]: float(np.mean([weights_4g[t][i] for t in indices]))
                for i in range(len(self.feature_names))
            }
            commercial[seg_name] = {
                self.feature_names[i]: float(np.mean([weights_5g[t][i] for t in indices]))
                for i in range(len(self.feature_names))
            }
        
        return {
            'industrial_like': industrial,
            'commercial_like': commercial,
            'segments': list(segments.keys())
        }
    
    def compute_weights_stats(self, weights_4g, weights_5g):
        """计算权重统计"""
        stats_4g = {}
        stats_5g = {}
        
        for i, name in enumerate(self.feature_names):
            stats_4g[name] = {
                'mean': float(np.mean(weights_4g[:, i])),
                'std': float(np.std(weights_4g[:, i])),
                'max': float(np.max(weights_4g[:, i])),
                'peak_hour': int(np.argmax(weights_4g[:, i])) / 2
            }
            stats_5g[name] = {
                'mean': float(np.mean(weights_5g[:, i])),
                'std': float(np.std(weights_5g[:, i])),
                'max': float(np.max(weights_5g[:, i])),
                'peak_hour': int(np.argmax(weights_5g[:, i])) / 2
            }
        
        return stats_4g, stats_5g
    
    def generate_report(self, weights_4g, weights_5g, comparison, barcelona_weights):
        """生成报告"""
        report = []
        report.append("# 4G vs 5G SHAP 权重分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 核心发现
        prb_ratio = weights_5g[:, 0].mean() / (weights_4g[:, 0].mean() + 1e-8)
        report.append("## 核心发现")
        report.append(f"1. 5G PRB 权重是 4G 的 {prb_ratio:.2f} 倍")
        
        peak_4g = np.argmax(weights_4g[:, 0]) / 2
        peak_5g = np.argmax(weights_5g[:, 0]) / 2
        report.append(f"2. PRB峰值时段: 4G 在 {peak_4g:.1f}点, 5G 在 {peak_5g:.1f}点")
        
        report.append(f"\n## 特征权重对比")
        for i, name in enumerate(feature_names):
            mean_4g = weights_4g[:, i].mean()
            mean_5g = weights_5g[:, i].mean()
            ratio = mean_5g / (mean_4g + 1e-8)
            report.append(f"  {name}: 4G={mean_4g:.6f}, 5G={mean_5g:.6f}, 比值={ratio:.2f}x")
        
        report.append(f"\n## 4G vs 5G 差异显著性")
        report.append(f"显著差异时段数: {comparison['significance']['significant_count']}")
        
        report.append(f"\n## 巴塞罗那权重表 (PRB)")
        for seg in barcelona_weights['segments']:
            w4 = barcelona_weights['industrial_like'][seg]['PRB']
            w5 = barcelona_weights['commercial_like'][seg]['PRB']
            report.append(f"  {seg}: 工业区={w4:.6f}, 商业区={w5:.6f}, 比值={w5/w4:.2f}x")
        
        return '\n'.join(report)


# ============================================================
# 可视化
# ============================================================
def plot_weights_comparison(weights_4g, weights_5g, output_dir):
    """绘制权重对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    hours = np.arange(48) / 2
    
    for i, (weights, title, color) in enumerate([
        (weights_4g, '4G', '#2E8B57'),
        (weights_5g, '5G', '#E76F51')
    ]):
        ax = axes[i]
        for f_idx, name in enumerate(feature_names):
            ax.plot(hours, weights[:, f_idx], label=name, linewidth=1.5)
        ax.set_xlabel('小时')
        ax.set_ylabel('SHAP 重要性')
        ax.set_title(f'{title} 30分钟粒度特征重要性')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weights_comparison.png', dpi=150)
    plt.close()
    print(f"✅ 图片保存: {output_dir / 'weights_comparison.png'}")


def plot_diff_heatmap(weights_4g, weights_5g, output_dir):
    """绘制差异热力图"""
    diff = weights_5g - weights_4g
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    ax.set_xlabel('小时')
    ax.set_ylabel('特征')
    ax.set_title('5G - 4G 权重差异')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([f'{i/2:.0f}' for i in range(0, 48, 4)])
    plt.colorbar(im, ax=ax, label='差异')
    plt.tight_layout()
    plt.savefig(output_dir / 'diff_heatmap.png', dpi=150)
    plt.close()
    print(f"✅ 图片保存: {output_dir / 'diff_heatmap.png'}")


def plot_barcelona_weights(barcelona_weights, output_dir):
    """绘制巴塞罗那权重图"""
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
    plt.savefig(output_dir / 'barcelona_weights.png', dpi=150)
    plt.close()
    print(f"✅ 图片保存: {output_dir / 'barcelona_weights.png'}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n开始对比分析...")
    
    # 创建分析器
    analyzer = ComparisonAnalyzer()
    
    # 1. 显著性检验
    print("\n3. 显著性检验...")
    significance = analyzer.significance_test(shap_4g, shap_5g)
    print(f"   显著差异时段数: {significance['significant_count']}")
    
    # 2. 巴塞罗那权重
    print("\n4. 计算巴塞罗那权重...")
    barcelona_weights = analyzer.compute_barcelona_weights(weights_4g, weights_5g)
    print("   时段权重:")
    for seg in barcelona_weights['segments']:
        w4 = barcelona_weights['industrial_like'][seg]['PRB']
        w5 = barcelona_weights['commercial_like'][seg]['PRB']
        print(f"     {seg}: 工业区={w4:.6f}, 商业区={w5:.6f}, 比值={w5/w4:.2f}x")
    
    # 3. 统计
    print("\n5. 计算统计...")
    stats_4g, stats_5g = analyzer.compute_weights_stats(weights_4g, weights_5g)
    print("   特征均值对比:")
    for name in feature_names:
        print(f"     {name}: 4G={stats_4g[name]['mean']:.6f}, 5G={stats_5g[name]['mean']:.6f}, 比值={stats_5g[name]['mean']/stats_4g[name]['mean']:.2f}x")
    
    # 4. 可视化
    print("\n6. 生成可视化...")
    plot_weights_comparison(weights_4g, weights_5g, OUTPUT_DIR)
    plot_diff_heatmap(weights_4g, weights_5g, OUTPUT_DIR)
    plot_barcelona_weights(barcelona_weights, OUTPUT_DIR)
    
    # 5. 保存结果
    print("\n7. 保存结果...")
    comparison_results = {
        'significance': significance,
        'barcelona_weights': barcelona_weights,
        'stats_4g': stats_4g,
        'stats_5g': stats_5g,
        'diff_30min': (weights_5g - weights_4g).tolist()
    }
    
    with open(OUTPUT_DIR / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"✅ 结果保存: {OUTPUT_DIR / 'comparison_results.json'}")
    
    # 6. 生成报告
    print("\n8. 生成报告...")
    report = analyzer.generate_report(weights_4g, weights_5g, comparison_results, barcelona_weights)
    with open(OUTPUT_DIR / 'report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 报告保存: {OUTPUT_DIR / 'report.md'}")
    
    # 打印报告
    print("\n" + "="*70)
    print("分析报告")
    print("="*70)
    print(report)
    
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
    
    print(f"\n✅ 全部完成！结果保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
