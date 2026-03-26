#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4G vs 5G SHAP 结果对比 + 映射到巴塞罗那权重
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SHAP_DIR = Path("D:/Desk/desk/beiyou_c_project/results/shap_analysis")
OUTPUT_DIR = Path("D:/Desk/desk/beiyou_c_project/results/barcelona_weights")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 巴塞罗那6小时时段
BARCELONA_SEGMENTS = {
    '00-06': list(range(0, 6)),
    '06-12': list(range(6, 12)),
    '12-18': list(range(12, 18)),
    '18-24': list(range(18, 24))
}

# 西班牙文化映射（时移）
CULTURAL_MAP = {
    '00-06': 0,      # 深夜（夜生活可能活跃）
    '06-12': 1,      # 上班晚1小时
    '12-18': 2,      # 午休长
    '18-24': 0       # 晚餐/夜生活
}

FEATURE_NAMES = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']


def load_latest_results(data_type):
    """加载最新的SHAP结果"""
    files = list(SHAP_DIR.glob(f"shap_results_{data_type}_*.json"))
    if not files:
        print(f"未找到 {data_type} 结果")
        return None
    latest = max(files, key=lambda x: x.stat().st_mtime)
    with open(latest, 'r') as f:
        return json.load(f)


def compute_hourly_weights(shap_values, hour_indices):
    """计算小时级权重"""
    # 从JSON中恢复hourly_importance需要shap_values
    # 简化：用特征重要性作为基础权重
    importance = shap_values.get('feature_importance', {})
    weights = np.array([importance.get(f, 0) for f in FEATURE_NAMES])
    return weights / (weights.sum() + 1e-8)


def map_to_barcelona(hourly_weights_4g, hourly_weights_5g):
    """映射到巴塞罗那6小时时段"""
    barcelona_weights = {}
    
    for segment, hours in BARCELONA_SEGMENTS.items():
        shift = CULTURAL_MAP[segment]
        # 中国小时 → 西班牙小时（时移）
        shifted_hours = [(h + shift) % 24 for h in hours]
        
        # 4G权重（类似工业区）
        w_4g = np.mean([hourly_weights_4g[h] for h in shifted_hours if h in hourly_weights_4g])
        # 5G权重（类似商业区）
        w_5g = np.mean([hourly_weights_5g[h] for h in shifted_hours if h in hourly_weights_5g])
        
        barcelona_weights[segment] = {
            'industrial_like': float(w_4g),
            'commercial_like': float(w_5g),
            'mixed': float((w_4g + w_5g) / 2)
        }
    
    return barcelona_weights


def main():
    print("="*60)
    print("4G vs 5G SHAP 对比分析")
    print("="*60)
    
    # 加载结果
    data_4g = load_latest_results('4g')
    data_5g = load_latest_results('5g')
    
    if data_4g is None or data_5g is None:
        print("请先运行全量分析")
        return
    
    print("\n1. 特征重要性对比")
    print("-"*40)
    print(f"{'特征':<12} {'4G':>12} {'5G':>12} {'差异':>12}")
    print("-"*40)
    
    importance_4g = data_4g['feature_importance']
    importance_5g = data_5g['feature_importance']
    
    for feat in FEATURE_NAMES:
        v4 = importance_4g.get(feat, 0)
        v5 = importance_5g.get(feat, 0)
        diff = v5 - v4
        print(f"{feat:<12} {v4:>12.6f} {v5:>12.6f} {diff:>+12.6f}")
    
    # 绘制对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(FEATURE_NAMES))
    width = 0.35
    
    v4 = [importance_4g.get(f, 0) for f in FEATURE_NAMES]
    v5 = [importance_5g.get(f, 0) for f in FEATURE_NAMES]
    
    ax.bar(x - width/2, v4, width, label='4G', color='#2E8B57')
    ax.bar(x + width/2, v5, width, label='5G', color='#E76F51')
    
    ax.set_ylabel('SHAP 重要性')
    ax.set_title('4G vs 5G 特征重要性对比')
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4g_5g_comparison.png', dpi=150)
    print(f"\n✅ 对比图保存: {OUTPUT_DIR / '4g_5g_comparison.png'}")
    
    # 计算小时级权重（简化：用特征重要性作为基础）
    hourly_weights_4g = {h: v4[0] for h in range(24)}  # 简化，实际应从SHAP小时级数据提取
    hourly_weights_5g = {h: v5[0] for h in range(24)}
    
    # 映射到巴塞罗那
    barcelona_weights = map_to_barcelona(hourly_weights_4g, hourly_weights_5g)
    
    print("\n2. 巴塞罗那6小时时段权重")
    print("-"*60)
    print(f"{'时段':<12} {'工业区(类似4G)':>18} {'商业区(类似5G)':>18} {'混合':>12}")
    print("-"*60)
    
    for segment, weights in barcelona_weights.items():
        print(f"{segment:<12} {weights['industrial_like']:>18.6f} "
              f"{weights['commercial_like']:>18.6f} {weights['mixed']:>12.6f}")
    
    # 保存权重
    output = {
        'timestamp': datetime.now().isoformat(),
        'feature_importance': {
            '4g': importance_4g,
            '5g': importance_5g
        },
        'barcelona_weights': barcelona_weights,
        'cultural_mapping': CULTURAL_MAP
    }
    
    with open(OUTPUT_DIR / 'barcelona_weights.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 权重保存: {OUTPUT_DIR / 'barcelona_weights.json'}")
    
    # 打印结论
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    if v4[0] > v5[0]:
        print("✅ 4G 基站中 PRB 重要性更高")
    else:
        print("✅ 5G 基站中 PRB 重要性更高")
    
    print("\n巴塞罗那联邦学习建议:")
    print("  - 工业区（类似4G）: 使用 4G 权重")
    print("  - 商业区（类似5G）: 使用 5G 权重")
    print("  - 混合区: 使用平均权重")


if __name__ == "__main__":
    main()
