#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
巴塞罗那42节点动态权重：每个时段独立判断
输出：每个节点每个时段的权重系数
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed/barcelona_ready_2019_2022"
OUTPUT_DIR = PROJECT_ROOT / "results/barcelona_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("巴塞罗那动态权重生成（时段级）")
print("="*70)

# ============================================================
# 1. 加载4G/5G权重曲线（24小时）
# ============================================================
print("\n加载4G/5G权重曲线...")
with open(PROJECT_ROOT / "results/shap_complete/shap_raw_4g.pkl", 'rb') as f:
    shap_4g = pickle.load(f)['shap_values']
with open(PROJECT_ROOT / "results/shap_comparison/shap_raw_5g.pkl", 'rb') as f:
    shap_5g = pickle.load(f)['shap_values']

# PRB权重（30分钟 → 24小时）
prb_4g_30min = np.abs(shap_4g[:, :, 0]).mean(axis=0)
prb_5g_30min = np.abs(shap_5g[:, :, 0]).mean(axis=0)
prb_4g_24h = np.array([prb_4g_30min[i*2:(i+1)*2].mean() for i in range(24)])
prb_5g_24h = np.array([prb_5g_30min[i*2:(i+1)*2].mean() for i in range(24)])

# 6小时时段划分（巴塞罗那粒度）
segments = {
    '00-06': (0, 6),
    '06-12': (6, 12),
    '12-18': (12, 18),
    '18-24': (18, 24)
}

# 每个时段的4G/5G权重
segment_weights = {}
for seg_name, (start, end) in segments.items():
    segment_weights[seg_name] = {
        '4g': prb_4g_24h[start:end].mean(),
        '5g': prb_5g_24h[start:end].mean()
    }

print("\n时段基准权重:")
for seg, w in segment_weights.items():
    print(f"  {seg}: 4G={w['4g']:.6f}, 5G={w['5g']:.6f}")

# ============================================================
# 2. 提取每个节点每个时段的能耗模式
# ============================================================
print("\n提取节点时段能耗...")
nodes = list(range(8001, 8043))
node_dynamic_weights = {}

for node in nodes:
    data_path = DATA_DIR / f"node_{node}" / "train.pkl"
    if not data_path.exists():
        continue
    
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    
    # 计算每个时段的平均能耗
    segment_energy = {}
    for seg_name, (start, end) in segments.items():
        # 该时段对应的小时范围
        hour_range = list(range(start, end))
        # 获取该节点这些小时的平均能耗
        energies = []
        for hour in hour_range:
            hour_idx = hour // 6
            val = df[df['hour_code'] == hour_idx]['Valor_norm'].mean()
            energies.append(val)
        segment_energy[seg_name] = np.mean(energies) if energies else 0
    
    # 计算该节点每个时段与4G/5G的相似度
    # 相似度 = 该节点时段能耗与4G/5G权重的相关性
    # 但这里用更简单的方式：比较该节点时段能耗与4G/5G的接近程度
    
    dynamic_weights = {}
    for seg_name in segments.keys():
        # 该节点此时段的能耗值
        node_val = segment_energy[seg_name]
        # 4G/5G基准权重
        w4 = segment_weights[seg_name]['4g']
        w5 = segment_weights[seg_name]['5g']
        
        # 计算该节点与该时段4G/5G的差异
        diff_4g = abs(node_val - w4)
        diff_5g = abs(node_val - w5)
        
        # 根据接近程度决定权重
        if diff_4g < diff_5g:
            # 更接近4G，用4G权重
            dynamic_weights[seg_name] = w4
        elif diff_5g < diff_4g:
            # 更接近5G，用5G权重
            dynamic_weights[seg_name] = w5
        else:
            # 相等，用平均
            dynamic_weights[seg_name] = (w4 + w5) / 2
        
        # 也记录相似度分数
        dynamic_weights[f'{seg_name}_type'] = '4g' if diff_4g < diff_5g else ('5g' if diff_5g < diff_4g else 'mixed')
    
    node_dynamic_weights[node] = dynamic_weights
    
    print(f"节点 {node}:")
    for seg in segments.keys():
        print(f"  {seg}: {dynamic_weights[seg]:.6f} ({dynamic_weights[f'{seg}_type']})")

# ============================================================
# 3. 保存动态权重表
# ============================================================
print("\n保存动态权重...")

# 转换为DataFrame
rows = []
for node, weights in node_dynamic_weights.items():
    row = {'node': node}
    for seg in segments.keys():
        row[f'{seg}_weight'] = weights[seg]
        row[f'{seg}_type'] = weights[f'{seg}_type']
    rows.append(row)

df_weights = pd.DataFrame(rows)
df_weights.to_csv(OUTPUT_DIR / 'dynamic_weights.csv', index=False)
print(f"✅ 动态权重表: {OUTPUT_DIR / 'dynamic_weights.csv'}")

# 保存为JSON
with open(OUTPUT_DIR / 'dynamic_weights.json', 'w') as f:
    json.dump(node_dynamic_weights, f, indent=2, default=str)
print(f"✅ 动态权重JSON: {OUTPUT_DIR / 'dynamic_weights.json'}")

# ============================================================
# 4. 统计每个时段各类型的节点数
# ============================================================
print("\n统计各时段类型分布:")
for seg in segments.keys():
    type_col = f'{seg}_type'
    counts = df_weights[type_col].value_counts()
    print(f"\n{seg}:")
    for t, cnt in counts.items():
        print(f"  {t}: {cnt}个节点")

# ============================================================
# 5. 生成联邦学习可用的权重映射
# ============================================================
print("\n生成联邦学习权重映射...")

# 格式: {node: {segment: weight}}
federated_weights = {}
for node, weights in node_dynamic_weights.items():
    federated_weights[str(node)] = {}
    for seg in segments.keys():
        federated_weights[str(node)][seg] = weights[seg]

with open(OUTPUT_DIR / 'federated_weights.json', 'w') as f:
    json.dump(federated_weights, f, indent=2)
print(f"✅ 联邦权重映射: {OUTPUT_DIR / 'federated_weights.json'}")

# ============================================================
# 6. 打印示例
# ============================================================
print("\n" + "="*70)
print("动态权重示例（前5个节点）")
print("="*70)
print(df_weights.head().to_string())

print(f"\n✅ 完成！结果保存至: {OUTPUT_DIR}")
