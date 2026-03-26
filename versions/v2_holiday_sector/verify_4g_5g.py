"""
验证4G和5G数据的相似性
目的：判断联邦学习是否可行
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(data_dir, max_stations=100):
    """加载数据，取公共特征前5维"""
    station_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
    station_dirs = station_dirs[:max_stations]
    
    all_features = []
    all_targets = []
    
    for station_dir in station_dirs:
        with open(station_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        # 只取前5维公共特征
        features = data['features'][:, :5]
        target = data['target']
        all_features.append(features)
        all_targets.append(target)
    
    features = np.concatenate(all_features, axis=0)
    target = np.concatenate(all_targets, axis=0)
    
    return features, target


print("="*60)
print("4G vs 5G 数据相似性分析")
print("="*60)

# 加载数据
data_dir = Path("D:/Desk/desk/beiyou_c_project/data/processed/tsinghua")

print("\n1. 加载数据...")
features_4g, target_4g = load_data(data_dir / '4g', max_stations=200)
features_5g, target_5g = load_data(data_dir / '5g', max_stations=200)

print(f"   4G: {len(features_4g)} 样本, 特征维度 {features_4g.shape[1]}")
print(f"   5G: {len(features_5g)} 样本, 特征维度 {features_5g.shape[1]}")

# ============================================================
# 1. 能耗分布对比
# ============================================================
print("\n2. 能耗分布对比...")

# 归一化后比较
target_4g_norm = (target_4g - target_4g.mean()) / target_4g.std()
target_5g_norm = (target_5g - target_5g.mean()) / target_5g.std()

print(f"   4G能耗: mean={target_4g.mean():.4f}, std={target_4g.std():.4f}")
print(f"   5G能耗: mean={target_5g.mean():.4f}, std={target_5g.std():.4f}")

# 分布相似性
from scipy.stats import ks_2samp
ks_stat, p_value = ks_2samp(target_4g_norm, target_5g_norm)
print(f"   KS检验: statistic={ks_stat:.4f}, p={p_value:.4f}")
if p_value > 0.05:
    print("   → 能耗分布相似 (p>0.05)，联邦学习可行")
else:
    print("   → 能耗分布不同 (p<0.05)，联邦学习可能无效")

# ============================================================
# 2. 特征相关性
# ============================================================
print("\n3. 特征与能耗的相关性...")

corr_4g = np.corrcoef(features_4g[:10000, 0], target_4g[:10000])[0, 1]
corr_5g = np.corrcoef(features_5g[:10000, 0], target_5g[:10000])[0, 1]

print(f"   4G PRB-能耗相关性: {corr_4g:.4f}")
print(f"   5G PRB-能耗相关性: {corr_5g:.4f}")

if abs(corr_4g - corr_5g) < 0.2:
    print("   → 特征关系相似，联邦学习有意义")
else:
    print("   → 特征关系差异大，联邦学习可能无效")

# ============================================================
# 3. 数据量差异（必要性）
# ============================================================
print("\n4. 数据量差异（必要性）")
print(f"   4G样本数: {len(features_4g):,}")
print(f"   5G样本数: {len(features_5g):,}")
print(f"   5G数据量是4G的 {len(features_5g)/len(features_4g)*100:.1f}%")

if len(features_5g) < len(features_4g):
    print("   → 5G数据少，需要向4G学习，联邦学习有必要")
else:
    print("   → 5G数据不少，联邦学习必要性降低")

# ============================================================
# 4. 画图对比
# ============================================================
print("\n5. 生成对比图...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 能耗分布直方图
axes[0, 0].hist(target_4g_norm, bins=50, alpha=0.5, label='4G', density=True)
axes[0, 0].hist(target_5g_norm, bins=50, alpha=0.5, label='5G', density=True)
axes[0, 0].set_xlabel('归一化能耗')
axes[0, 0].set_ylabel('密度')
axes[0, 0].set_title('能耗分布对比')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 时间序列示例
sample_4g = target_4g[:200]
sample_5g = target_5g[:200]
axes[0, 1].plot(sample_4g, label='4G', alpha=0.7)
axes[0, 1].plot(sample_5g, label='5G', alpha=0.7)
axes[0, 1].set_xlabel('时间点')
axes[0, 1].set_ylabel('能耗')
axes[0, 1].set_title('能耗时间序列对比')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# PRB-能耗散点图
axes[1, 0].scatter(features_4g[:2000, 0], target_4g[:2000], alpha=0.3, s=1, label='4G')
axes[1, 0].scatter(features_5g[:2000, 0], target_5g[:2000], alpha=0.3, s=1, label='5G')
axes[1, 0].set_xlabel('PRB使用率')
axes[1, 0].set_ylabel('能耗')
axes[1, 0].set_title('PRB-能耗关系对比')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 数据量对比
labels = ['4G', '5G']
sizes = [len(features_4g), len(features_5g)]
colors = ['#2E8B57', '#E76F51']
axes[1, 1].bar(labels, sizes, color=colors)
axes[1, 1].set_ylabel('样本数')
axes[1, 1].set_title('数据量对比')
axes[1, 1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(sizes):
    axes[1, 1].text(i, v + 5000, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/4g_5g_comparison.png', dpi=150)
print("   ✅ 图片保存: results/4g_5g_comparison.png")

# ============================================================
# 结论
# ============================================================
print("\n" + "="*60)
print("结论")
print("="*60)

if p_value > 0.05 and abs(corr_4g - corr_5g) < 0.2:
    print("✅ 4G和5G能耗模式相似，联邦学习可行")
    print("✅ 5G数据量少，需要向4G学习，联邦学习有必要")
    print("\n建议：继续做4G+5G联邦学习")
else:
    print("⚠️ 4G和5G能耗模式差异较大，联邦学习可能无效")
    print("\n建议：放弃4G+5G联邦，转向跨国协同或粒度协同")

