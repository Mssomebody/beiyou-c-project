#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级高级聚类：多维度特征 + PCA降维 + 自动寻优
"""

import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed/barcelona_ready_2019_2022"
OUTPUT_DIR = PROJECT_ROOT / "results/barcelona_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("巴塞罗那42节点多维度高级聚类")
print("="*80)

# ============================================================
# 1. 加载所有节点的训练数据
# ============================================================
nodes = list(range(8001, 8043))
node_data = {}

print("\n【1/6】加载节点数据...")
for node in nodes:
    data_path = DATA_DIR / f"node_{node}" / "train.pkl"
    if not data_path.exists():
        continue
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    node_data[node] = df
print(f"成功加载 {len(node_data)} 个节点")

# ============================================================
# 2. 特征提取
# ============================================================
print("\n【2/6】提取多维度特征...")

features = []  # 每个节点的特征向量
feature_names = []

for node, df in node_data.items():
    feat = []
    
    # (1) 时段能耗（4个6小时时段）
    segment_means = []
    for hour_code in range(4):
        val = df[df['hour_code'] == hour_code]['Valor_norm'].mean()
        segment_means.append(val)
    feat.extend(segment_means)
    if not feature_names:
        feature_names.extend([f'seg{i}' for i in range(4)])
    
    # (2) 节假日效应
    holiday_vals = df.groupby('is_holiday')['Valor_norm'].mean()
    holiday_effect = (holiday_vals.get(1, 0) - holiday_vals.get(0, 0)) / (holiday_vals.get(0, 1) + 1e-8)
    feat.append(holiday_effect)
    if not feature_names: feature_names.append('holiday_effect')
    
    # (3) 周末效应
    weekend_vals = df.groupby('is_weekend')['Valor_norm'].mean()
    weekend_effect = (weekend_vals.get(1, 0) - weekend_vals.get(0, 0)) / (weekend_vals.get(0, 1) + 1e-8)
    feat.append(weekend_effect)
    if not feature_names: feature_names.append('weekend_effect')
    
    # (4) 月度模式（从Data列提取月份）
    df['month'] = pd.to_datetime(df['Data']).dt.month
    monthly_means = df.groupby('month')['Valor_norm'].mean().reindex(range(1,13), fill_value=0)
    feat.extend(monthly_means.values)
    if not feature_names: feature_names.extend([f'month{i}' for i in range(1,13)])
    
    # (5) 波动性（跨年标准差）
    df['year'] = pd.to_datetime(df['Data']).dt.year
    yearly_means = df.groupby('year')['Valor_norm'].mean()
    volatility = yearly_means.std() / (yearly_means.mean() + 1e-8)
    feat.append(volatility)
    if not feature_names: feature_names.append('volatility')
    
    # (6) 部门-时段交互：计算每个节点每个时段中工业/商业的能耗占比（需要原始未聚合数据）
    # 由于我们已有每个时段内各个部门的数据，可以从df中提取
    # 部门列: sector_0(工业), sector_1(住宅), sector_2(商业), sector_3(其他)
    # 但df中每个记录对应一个部门？实际上每个时段内只有一个能耗值，部门信息已被编码到sector_0-3列作为one-hot，但每行只有一个部门为1。
    # 因此可以按部门分组计算每个时段的平均能耗。
    sector_cols = ['sector_0', 'sector_1', 'sector_2', 'sector_3']
    sector_names = ['industry', 'residential', 'commercial', 'other']
    for hour_code in range(4):
        hour_df = df[df['hour_code'] == hour_code]
        for s_idx, s_name in enumerate(sector_names):
            s_val = hour_df[hour_df[sector_cols[s_idx]] == 1]['Valor_norm'].mean()
            if np.isnan(s_val):
                s_val = 0
            feat.append(s_val)
            if not feature_names:
                feature_names.append(f'seg{hour_code}_{s_name}')
    
    features.append(feat)

# 转为 numpy 数组
X = np.array(features)
print(f"特征矩阵形状: {X.shape}")

# 处理 NaN
X = np.nan_to_num(X, nan=0.0)

# ============================================================
# 3. 标准化 + PCA 降维
# ============================================================
print("\n【3/6】PCA降维...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 选择保留95%方差的维度
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"原始维度: {X.shape[1]}, 降维后: {X_pca.shape[1]}")

# ============================================================
# 4. 确定最优聚类数
# ============================================================
print("\n【4/6】确定最优聚类数...")
K_range = range(2, 9)
silhouettes = []
inertias = []
ch_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    silhouettes.append(silhouette_score(X_pca, labels))
    inertias.append(kmeans.inertia_)
    ch_scores.append(calinski_harabasz_score(X_pca, labels))

# 综合得分（归一化后平均）
inertia_norm = 1 - (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias))
sil_norm = np.array(silhouettes)
ch_norm = (np.array(ch_scores) - min(ch_scores)) / (max(ch_scores) - min(ch_scores))
composite = (inertia_norm + sil_norm + ch_norm) / 3
optimal_k = K_range[np.argmax(composite)]

print(f"最优聚类数: {optimal_k}")
print(f"  轮廓系数: {silhouettes[optimal_k-2]:.4f}")
print(f"  CH指数: {ch_scores[optimal_k-2]:.1f}")

# ============================================================
# 5. 最终聚类（KMeans + 层次聚类投票）
# ============================================================
print("\n【5/6】执行聚类...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_pca)

# 层次聚类
agg = AgglomerativeClustering(n_clusters=optimal_k)
labels_agg = agg.fit_predict(X_pca)

# 投票融合（简单多数）
final_labels = []
for i in range(len(labels_kmeans)):
    if labels_kmeans[i] == labels_agg[i]:
        final_labels.append(labels_kmeans[i])
    else:
        final_labels.append(labels_kmeans[i])  # 以KMeans为准

# ============================================================
# 6. 聚类分析
# ============================================================
print("\n【6/6】聚类分析...")

segments = ['00-06', '06-12', '12-18', '18-24']
sector_names = ['工业', '住宅', '商业', '其他']

# 整理每个簇的数据
clusters = {}
nodes_list = list(node_data.keys())
for i, node in enumerate(nodes_list):
    cid = final_labels[i]
    if cid not in clusters:
        clusters[cid] = {
            'nodes': [],
            'features': [],
            'segment_means': [],
            'holiday_effect': [],
            'weekend_effect': [],
            'monthly': [],
            'volatility': [],
            'sector_segment': []  # 部门-时段交互
        }
    clusters[cid]['nodes'].append(node)
    clusters[cid]['features'].append(features[i])
    clusters[cid]['segment_means'].append(X[i][:4])
    clusters[cid]['holiday_effect'].append(X[i][4])
    clusters[cid]['weekend_effect'].append(X[i][5])
    clusters[cid]['monthly'].append(X[i][6:18])
    clusters[cid]['volatility'].append(X[i][18])
    # 部门-时段交互特征从索引19开始
    clusters[cid]['sector_segment'].append(X[i][19:])

# 输出簇摘要
print("\n" + "="*80)
print("聚类结果摘要")
print("="*80)

cluster_summary = []
for cid in sorted(clusters.keys()):
    data = clusters[cid]
    avg_segment = np.mean(data['segment_means'], axis=0)
    avg_holiday = np.mean(data['holiday_effect'])
    avg_weekend = np.mean(data['weekend_effect'])
    avg_volatility = np.mean(data['volatility'])
    
    # 找出峰值时段
    peak_idx = np.argmax(avg_segment)
    peak_seg = segments[peak_idx]
    
    # 判断类型
    if peak_idx == 3:
        type_label = "夜间高峰型"
    elif peak_idx == 2:
        type_label = "下午高峰型"
    elif peak_idx == 1:
        type_label = "上午高峰型"
    else:
        type_label = "凌晨高峰型"
    
    # 补充部门-时段特征（简化为主要部门占优）
    avg_sector_seg = np.mean(data['sector_segment'], axis=0).reshape(4, 4)  # 4时段 x 4部门
    # 找出每个时段最主要的部门
    dominant_sectors = []
    for t in range(4):
        sec_idx = np.argmax(avg_sector_seg[t])
        dominant_sectors.append(sector_names[sec_idx])
    type_label += f" / 主要部门时段: {', '.join(dominant_sectors)}"
    
    cluster_summary.append({
        'cluster': cid,
        'node_count': len(data['nodes']),
        'nodes': data['nodes'],
        'avg_segment': avg_segment.tolist(),
        'peak_hour': peak_seg,
        'peak_value': float(avg_segment[peak_idx]),
        'avg_holiday_effect': float(avg_holiday),
        'avg_weekend_effect': float(avg_weekend),
        'avg_volatility': float(avg_volatility),
        'dominant_sectors': dominant_sectors,
        'type': type_label
    })
    
    print(f"\n聚类 {cid} ({type_label}):")
    print(f"  节点数: {len(data['nodes'])}")
    print(f"  节点列表: {data['nodes'][:10]}{'...' if len(data['nodes']) > 10 else ''}")
    print(f"  6小时能耗: {', '.join([f'{seg}={val:.4f}' for seg, val in zip(segments, avg_segment)])}")
    print(f"  峰值时段: {peak_seg} ({avg_segment[peak_idx]:.4f})")
    print(f"  节假日效应: {avg_holiday:.4f} (正表示节假日更高)")
    print(f"  周末效应: {avg_weekend:.4f} (正表示周末更高)")
    print(f"  波动性: {avg_volatility:.4f}")
    print(f"  各时段主要部门: {', '.join([f'{seg}:{sec}' for seg, sec in zip(segments, dominant_sectors)])}")

# ============================================================
# 7. 保存结果
# ============================================================
print("\n保存结果...")

# 节点分类表
classification = []
for i, node in enumerate(nodes_list):
    cid = final_labels[i]
    cluster_info = cluster_summary[cid]
    classification.append({
        'node': node,
        'cluster': cid,
        'type': cluster_info['type'],
        'peak_hour': cluster_info['peak_hour'],
        'holiday_effect': float(features[i][4]),
        'weekend_effect': float(features[i][5]),
        'volatility': float(features[i][18]),
        'energy_00_06': float(features[i][0]),
        'energy_06_12': float(features[i][1]),
        'energy_12_18': float(features[i][2]),
        'energy_18_24': float(features[i][3]),
    })
df_class = pd.DataFrame(classification)
df_class.to_csv(OUTPUT_DIR / 'node_classification_advanced.csv', index=False)
print(f"✅ 节点分类表: {OUTPUT_DIR / 'node_classification_advanced.csv'}")

# 聚类摘要
with open(OUTPUT_DIR / 'cluster_summary_advanced.json', 'w') as f:
    json.dump(cluster_summary, f, indent=2, default=str)
print(f"✅ 聚类摘要: {OUTPUT_DIR / 'cluster_summary_advanced.json'}")

# ============================================================
# 8. 可视化
# ============================================================
print("\n生成可视化...")

# 图1: 各簇能耗曲线
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_summary)))
for idx, info in enumerate(cluster_summary):
    ax.plot(segments, info['avg_segment'], 'o-', 
            label=f"聚类{info['cluster']} ({info['type'][:12]})", 
            color=colors[idx], linewidth=2, markersize=8)
ax.set_xlabel('时段')
ax.set_ylabel('归一化能耗')
ax.set_title('各聚类6小时能耗曲线')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cluster_curves_advanced.png', dpi=150)
plt.close()

# 图2: 最优K选择
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(list(K_range), inertias, 'bo-')
axes[0].set_xlabel('K')
axes[0].set_ylabel('惯性')
axes[0].set_title('肘部法则')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), silhouettes, 'ro-')
axes[1].axvline(optimal_k, color='green', linestyle='--', label=f'最优 K={optimal_k}')
axes[1].set_xlabel('K')
axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(list(K_range), composite, 'go-')
axes[2].axvline(optimal_k, color='green', linestyle='--')
axes[2].set_xlabel('K')
axes[2].set_ylabel('综合得分')
axes[2].set_title('综合得分')
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'optimal_k_advanced.png', dpi=150)
plt.close()

# 图3: 部门-时段热力图（每个簇）
fig, axes = plt.subplots(1, len(cluster_summary), figsize=(4*len(cluster_summary), 4))
if len(cluster_summary) == 1:
    axes = [axes]
for idx, info in enumerate(cluster_summary):
    avg_sector_seg = np.mean([f[19:] for f in clusters[info['cluster']]['features']], axis=0).reshape(4, 4)
    ax = axes[idx]
    im = ax.imshow(avg_sector_seg, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(sector_names, rotation=45, ha='right')
    ax.set_yticks(range(4))
    ax.set_yticklabels(segments)
    ax.set_title(f'聚类{info["cluster"]}')
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sector_segment_heatmap.png', dpi=150)
plt.close()

print(f"\n✅ 完成！结果保存至: {OUTPUT_DIR}")
print("\n" + "="*80)
print("节点分类示例")
print("="*80)
print(df_class[['node', 'cluster', 'type', 'peak_hour', 'holiday_effect', 'weekend_effect', 'volatility']].head(15).to_string())
