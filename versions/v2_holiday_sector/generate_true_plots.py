"""
基于真实数据生成图片
"""

import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 精度对比图（基于两阶段结果 JSON）
# ============================================================

with open('../../results/two_stage/two_stage_results_20260323_183730.json', 'r') as f:
    data = json.load(f)

models = ['混合口径', '旧口径', '新口径', '两阶段']
values = [
    float(data['baselines']['mixed']),      # 58.12
    float(data['baselines']['old_only']),   # 65.53
    float(data['baselines']['new_only']),   # 70.78
    float(data['metrics']['new_smape'])     # 60.54
]

colors = ['#2E8B57', '#2E86AB', '#E76F51', '#F4A261']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1)
ax.set_ylabel('sMAPE (%)')
ax.set_title('基站能耗预测精度对比（巴塞罗那数据）')
ax.set_ylim(0, 80)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=11)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../../results/accuracy_comparison.png', dpi=150)
print('✅ 精度对比图: results/accuracy_comparison.png')
print(f'   混合口径: {values[0]}%')
print(f'   旧口径: {values[1]}%')
print(f'   新口径: {values[2]}%')
print(f'   两阶段: {values[3]}%')

# ============================================================
# 2. 4G/5G 对比图（基于真实结果）
# ============================================================

categories = ['4G', '5G']
values_single = [47.57, 35.98]

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2E8B57', '#E76F51']
bars = ax.bar(categories, values_single, color=colors, edgecolor='black', linewidth=1)
ax.set_ylabel('sMAPE (%)')
ax.set_title('4G vs 5G 单独训练精度对比')
ax.set_ylim(0, 60)

for bar, val in zip(bars, values_single):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val:.2f}%', ha='center', fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../../results/4g_5g_comparison.png', dpi=150)
print('\n✅ 4G/5G对比图: results/4g_5g_comparison.png')
print(f'   4G: {values_single[0]}%')
print(f'   5G: {values_single[1]}%')

# ============================================================
# 3. 两阶段预测曲线图（基于两阶段模型）
# ============================================================
print('\n⚠️  预测曲线图需要运行模型预测')
print('   可运行: python plot_predictions_fixed.py')
