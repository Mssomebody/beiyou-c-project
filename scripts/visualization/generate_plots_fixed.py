#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成粒度融合实验的可视化图表"""
import re
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 从日志中提取训练损失（2节点粒度融合）
# ============================================================
def extract_loss_from_log(log_file):
    losses = []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = re.search(r'Round (\d+): avg_train_loss = ([\d\.]+)', line)
                if m:
                    losses.append(float(m.group(2)))
    except FileNotFoundError:
        print(f"文件 {log_file} 不存在")
    return losses

fusion_2node_loss = extract_loss_from_log('fusion_2node.log')
if fusion_2node_loss:
    print(f"提取到 {len(fusion_2node_loss)} 轮损失")
else:
    print("警告：未从 fusion_2node.log 中提取到损失，请检查文件内容")

# ============================================================
# 2. 精度对比数据
# ============================================================
smape_data = {
    '2节点不加权基线': 60.45,
    '2节点粒度融合': 33.88,
    '41节点粒度融合': 40.81,
}

# ============================================================
# 3. 绘制损失曲线
# ============================================================
if fusion_2node_loss:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(fusion_2node_loss)+1), fusion_2node_loss, marker='o', linestyle='-', color='#2E8B57')
    plt.xlabel('轮次')
    plt.ylabel('平均训练损失')
    plt.title('2节点粒度融合训练损失曲线')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fusion_2node_loss_curve.png', dpi=150)
    plt.close()
    print("✅ 损失曲线保存为 fusion_2node_loss_curve.png")

# ============================================================
# 4. 绘制精度对比柱状图
# ============================================================
labels = list(smape_data.keys())
values = list(smape_data.values())
colors = ['#2E8B57', '#E76F51', '#2E86AB']

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=colors, edgecolor='black')
plt.ylabel('sMAPE (%)')
plt.title('粒度融合精度对比')
plt.ylim(0, 80)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fusion_accuracy_comparison.png', dpi=150)
plt.close()
print("✅ 精度对比图保存为 fusion_accuracy_comparison.png")

print("\n提示：如需预测曲线图，请运行评估脚本输出预测值与真实值。")
