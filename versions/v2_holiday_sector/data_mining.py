#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据挖掘：从 SHAP 结果提取完整的小时级权重信息
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

SHAP_DIR = Path("D:/Desk/desk/beiyou_c_project/results/shap_analysis")
OUTPUT_DIR = Path("D:/Desk/desk/beiyou_c_project/results/data_mining")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']


def load_hourly_data(data_type):
    """加载包含小时级权重的 JSON"""
    files = list(SHAP_DIR.glob(f"shap_results_{data_type}_*.json"))
    # 筛选包含 hourly_importance_matrix 的文件
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            if 'hourly_importance_matrix' in data:
                print(f"加载 {data_type}: {f.name}")
                return data
    print(f"警告: 未找到 {data_type} 的小时级数据")
    return None


def main():
    print("="*70)
    print("数据挖掘：4G/5G 小时级权重完整分析")
    print("="*70)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_4g = load_hourly_data('4g')
    data_5g = load_hourly_data('5g')
    
    if data_4g is None or data_5g is None:
        print("\n❌ 缺少小时级数据，请先运行 shap_pytorch_hourly.py")
        print("   命令: python shap_pytorch_hourly.py --data_type 4g --max_stations 0 --samples_per_station 0")
        return
    
    hourly_4g = np.array(data_4g['hourly_importance_matrix'])  # [24, 5]
    hourly_5g = np.array(data_5g['hourly_importance_matrix'])  # [24, 5]
    
    print(f"   4G 小时级权重形状: {hourly_4g.shape}")
    print(f"   5G 小时级权重形状: {hourly_5g.shape}")
    
    # 2. 统计摘要
    print("\n2. 统计摘要...")
    print(f"\n   {'特征':<12} {'4G 均值':>12} {'4G 标准差':>12} {'5G 均值':>12} {'5G 标准差':>12}")
    print("   " + "-"*60)
    for i, name in enumerate(FEATURE_NAMES):
        mean_4g = hourly_4g[:, i].mean()
        std_4g = hourly_4g[:, i].std()
        mean_5g = hourly_5g[:, i].mean()
        std_5g = hourly_5g[:, i].std()
        print(f"   {name:<12} {mean_4g:>12.6f} {std_4g:>12.6f} {mean_5g:>12.6f} {std_5g:>12.6f}")
    
    # 3. 特征排名
    print("\n3. 特征重要性排名...")
    total_4g = hourly_4g.mean(axis=0)
    total_5g = hourly_5g.mean(axis=0)
    
    print(f"\n   4G:")
    for i, name in sorted(zip(total_4g, FEATURE_NAMES), reverse=True):
        print(f"      {name}: {i:.6f}")
    
    print(f"\n   5G:")
    for i, name in sorted(zip(total_5g, FEATURE_NAMES), reverse=True):
        print(f"      {name}: {i:.6f}")
    
    # 4. 4G vs 5G 差异分析（PRB 特征）
    print("\n4. 4G vs 5G PRB 权重差异分析...")
    prb_4g = hourly_4g[:, 0]
    prb_5g = hourly_5g[:, 0]
    
    # 计算每个小时的差异
    diff = prb_5g - prb_4g
    # 计算整体差异的显著性（配对 t 检验）
    t_stat, p_value = stats.ttest_rel(prb_5g, prb_4g)
    
    print(f"   配对 t 检验: t={t_stat:.4f}, p={p_value:.6f}")
    if p_value < 0.05:
        print("   ✅ 4G 和 5G 的 PRB 权重有显著差异 (p<0.05)")
    else:
        print("   ⚠️ 4G 和 5G 的 PRB 权重无显著差异 (p>=0.05)")
    
    # 找出差异最大的小时
    print("\n   差异最大的小时:")
    for h in np.argsort(np.abs(diff))[-5:][::-1]:
        print(f"      小时 {h}: 4G={prb_4g[h]:.6f}, 5G={prb_5g[h]:.6f}, 差={diff[h]:+.6f}")
    
    # 5. 关键时段识别
    print("\n5. 关键时段识别（基于 PRB 权重）...")
    
    # 找出 PRB 权重最高的 3 个时段
    top_3_4g = np.argsort(prb_4g)[-3:][::-1]
    top_3_5g = np.argsort(prb_5g)[-3:][::-1]
    
    print(f"\n   4G 高峰时段: {sorted(top_3_4g)}")
    print(f"   5G 高峰时段: {sorted(top_3_5g)}")
    
    # 6. 可视化
    print("\n6. 生成可视化...")
    
    # 图1: 24小时 × 5特征 热力图（4G vs 5G 并排）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    im1 = axes[0].imshow(hourly_4g.T, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('小时')
    axes[0].set_ylabel('特征')
    axes[0].set_title('4G 小时级特征重要性')
    axes[0].set_yticks(range(len(FEATURE_NAMES)))
    axes[0].set_yticklabels(FEATURE_NAMES)
    plt.colorbar(im1, ax=axes[0], label='SHAP 重要性')
    
    im2 = axes[1].imshow(hourly_5g.T, aspect='auto', cmap='YlOrRd')
    axes[1].set_xlabel('小时')
    axes[1].set_ylabel('特征')
    axes[1].set_title('5G 小时级特征重要性')
    axes[1].set_yticks(range(len(FEATURE_NAMES)))
    axes[1].set_yticklabels(FEATURE_NAMES)
    plt.colorbar(im2, ax=axes[1], label='SHAP 重要性')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hourly_heatmap_comparison.png', dpi=150)
    print(f"   ✅ 热力图: {OUTPUT_DIR / 'hourly_heatmap_comparison.png'}")
    
    # 图2: PRB 小时级曲线对比（带置信区间）
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = np.arange(24)
    ax.plot(hours, prb_4g, label='4G PRB', color='#2E8B57', linewidth=2, marker='o')
    ax.plot(hours, prb_5g, label='5G PRB', color='#E76F51', linewidth=2, marker='s')
    ax.fill_between(hours, prb_4g - hourly_4g[:, 0].std(), prb_4g + hourly_4g[:, 0].std(), alpha=0.2, color='#2E8B57')
    ax.fill_between(hours, prb_5g - hourly_5g[:, 0].std(), prb_5g + hourly_5g[:, 0].std(), alpha=0.2, color='#E76F51')
    ax.set_xlabel('小时')
    ax.set_ylabel('PRB 重要性')
    ax.set_title('4G vs 5G PRB 小时级重要性对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'prb_hourly_comparison.png', dpi=150)
    print(f"   ✅ PRB 对比图: {OUTPUT_DIR / 'prb_hourly_comparison.png'}")
    
    # 图3: 各特征小时级曲线（4G）
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(FEATURE_NAMES):
        ax.plot(hours, hourly_4g[:, i], label=name, linewidth=1.5)
    ax.set_xlabel('小时')
    ax.set_ylabel('SHAP 重要性')
    ax.set_title('4G 各特征小时级重要性')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4g_all_features_hourly.png', dpi=150)
    print(f"   ✅ 4G 全特征图: {OUTPUT_DIR / '4g_all_features_hourly.png'}")
    
    # 图4: 各特征小时级曲线（5G）
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(FEATURE_NAMES):
        ax.plot(hours, hourly_5g[:, i], label=name, linewidth=1.5)
    ax.set_xlabel('小时')
    ax.set_ylabel('SHAP 重要性')
    ax.set_title('5G 各特征小时级重要性')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5g_all_features_hourly.png', dpi=150)
    print(f"   ✅ 5G 全特征图: {OUTPUT_DIR / '5g_all_features_hourly.png'}")
    
    # 7. 保存完整数据
    print("\n7. 保存完整数据...")
    results = {
        'feature_names': FEATURE_NAMES,
        'hourly_4g': hourly_4g.tolist(),
        'hourly_5g': hourly_5g.tolist(),
        'statistics': {
            '4g_mean': hourly_4g.mean(axis=0).tolist(),
            '4g_std': hourly_4g.std(axis=0).tolist(),
            '5g_mean': hourly_5g.mean(axis=0).tolist(),
            '5g_std': hourly_5g.std(axis=0).tolist(),
        },
        'prb_diff': {
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'peak_hours': {
            '4g': [int(h) for h in top_3_4g],
            '5g': [int(h) for h in top_3_5g]
        }
    }
    
    with open(OUTPUT_DIR / 'hourly_weights_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✅ 数据保存: {OUTPUT_DIR / 'hourly_weights_analysis.json'}")
    
    print("\n" + "="*70)
    print("数据挖掘完成")
    print("="*70)
    print("\n发现:")
    print(f"  - 4G 高峰时段: {sorted(top_3_4g)}")
    print(f"  - 5G 高峰时段: {sorted(top_3_5g)}")
    if p_value < 0.05:
        print("  - 4G 和 5G 的 PRB 权重有显著差异")
    else:
        print("  - 4G 和 5G 的 PRB 权重无显著差异")


if __name__ == "__main__":
    main()
