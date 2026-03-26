#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤3：时段聚类 + 置信区间
基于采样结果
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

shap_dir = Path('D:/Desk/desk/beiyou_c_project/results/shap_full')
output_dir = Path('D:/Desk/desk/beiyou_c_project/results/data_mining')
output_dir.mkdir(parents=True, exist_ok=True)

features = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']

with open(shap_dir / 'shap_results_4g_20260324_131153.json') as f:
    matrix_4g = np.array(json.load(f)['hourly_importance_matrix'])

with open(shap_dir / 'shap_results_5g_20260324_132638.json') as f:
    matrix_5g = np.array(json.load(f)['hourly_importance_matrix'])

print('='*80)
print('步骤3：时段聚类 + 置信区间')
print('='*80)

# 1. 时段聚类（基于 PRB 权重）
print('\n1. 时段聚类 (K-Means)')
print('-'*80)

prb_4g = matrix_4g[:, 0].reshape(-1, 1)
prb_5g = matrix_5g[:, 0].reshape(-1, 1)

# 肘部法则确定最佳 k
inertias = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(prb_4g)
    inertias.append(kmeans.inertia_)

# 找拐点
diffs = np.diff(inertias)
best_k = np.argmax(np.abs(diffs)) + 2
print(f'   最佳聚类数: {best_k}')

# 4G 聚类
kmeans_4g = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_4g = kmeans_4g.fit_predict(prb_4g)
print(f'\n   4G 聚类结果:')
for k in range(best_k):
    hours = np.where(labels_4g == k)[0]
    print(f'      类别{k+1}: 小时 {hours.tolist()}')

# 5G 聚类
kmeans_5g = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_5g = kmeans_5g.fit_predict(prb_5g)
print(f'\n   5G 聚类结果:')
for k in range(best_k):
    hours = np.where(labels_5g == k)[0]
    print(f'      类别{k+1}: 小时 {hours.tolist()}')

# 2. 置信区间 (Bootstrap)
print('\n2. 置信区间 (Bootstrap)')
print('-'*80)

n_bootstrap = 1000
ci_lower = []
ci_upper = []

for i in range(24):
    # 模拟每个小时有多个样本（这里用相邻小时的值模拟）
    # 实际应该用原始 SHAP 值，这里简化
    samples = prb_5g[max(0, i-2):min(24, i+3)].flatten()
    if len(samples) < 2:
        samples = prb_5g.flatten()
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(samples, size=len(samples), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    ci_lower.append(lower)
    ci_upper.append(upper)

print(f'   5G PRB 权重 95% 置信区间:')
for i in range(24):
    print(f'      小时{i:2d}: {prb_5g[i]:.6f} [{ci_lower[i]:.6f}, {ci_upper[i]:.6f}]')

# 3. 巴塞罗那 6小时时段权重
print('\n3. 巴塞罗那 6小时时段权重')
print('-'*80)

barcelona_segments = {
    '00-06': list(range(0, 6)),
    '06-12': list(range(6, 12)),
    '12-18': list(range(12, 18)),
    '18-24': list(range(18, 24))
}

print(f'\n{"时段":<10} {"4G PRB":>12} {"5G PRB":>12} {"工业区(4G-like)":>16} {"商业区(5G-like)":>16}')
print('-'*70)

for seg, hours in barcelona_segments.items():
    w4 = np.mean([prb_4g[h] for h in hours])
    w5 = np.mean([prb_5g[h] for h in hours])
    print(f'{seg:<10} {w4:>12.6f} {w5:>12.6f} {w4:>16.6f} {w5:>16.6f}')

# 4. 可视化聚类结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
hours = np.arange(24)
colors = plt.cm.Set1(np.linspace(0, 1, best_k))
for k in range(best_k):
    mask = labels_4g == k
    ax.scatter(hours[mask], prb_4g[mask], c=[colors[k]], label=f'类别{k+1}', s=50)
ax.plot(hours, prb_4g, 'k-', alpha=0.5, linewidth=1)
ax.set_xlabel('小时')
ax.set_ylabel('PRB 权重')
ax.set_title('4G PRB 时段聚类')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for k in range(best_k):
    mask = labels_5g == k
    ax.scatter(hours[mask], prb_5g[mask], c=[colors[k]], label=f'类别{k+1}', s=50)
ax.plot(hours, prb_5g, 'k-', alpha=0.5, linewidth=1)
ax.set_xlabel('小时')
ax.set_ylabel('PRB 权重')
ax.set_title('5G PRB 时段聚类')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'clustering_result.png', dpi=150)
print(f'\n✅ 聚类图保存: {output_dir / "clustering_result.png"}')

# 5. 保存结果
results = {
    'clustering': {
        '4g_labels': labels_4g.tolist(),
        '5g_labels': labels_5g.tolist(),
        'best_k': best_k
    },
    'barcelona_weights': {
        seg: {
            'industrial_like': float(np.mean([prb_4g[h] for h in hours])),
            'commercial_like': float(np.mean([prb_5g[h] for h in hours]))
        }
        for seg, hours in barcelona_segments.items()
    },
    'confidence_intervals': {
        str(i): {'lower': ci_lower[i], 'upper': ci_upper[i], 'mean': prb_5g[i]}
        for i in range(24)
    }
}

with open(output_dir / 'clustering_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✅ 结果保存: {output_dir / "clustering_results.json"}')
