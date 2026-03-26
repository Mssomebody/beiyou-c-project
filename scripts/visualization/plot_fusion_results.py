#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成粒度融合实验的可视化图表"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "versions" / "v2_holiday_sector"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_loss_from_log(log_file):
    losses = []
    for enc in ['utf-8', 'gbk', 'gb2312']:
        try:
            with open(log_file, 'r', encoding=enc) as f:
                for line in f:
                    m = re.search(r'Round (\d+): avg_train_loss = ([\d\.]+)', line)
                    if m:
                        losses.append(float(m.group(2)))
            if losses:
                return losses
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return losses

def plot_loss_curve(losses, output_file):
    if not losses:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o', linestyle='-', color='#2E8B57')
    plt.xlabel('轮次')
    plt.ylabel('平均训练损失')
    plt.title('2节点粒度融合训练损失曲线')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✅ 损失曲线保存至 {output_file}")

def plot_accuracy_comparison(data, output_file):
    labels = list(data.keys())
    values = list(data.values())
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
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✅ 精度对比图保存至 {output_file}")

def main():
    log_file = LOG_DIR / "fusion_2node.log"
    if log_file.exists():
        losses = extract_loss_from_log(log_file)
        if losses:
            plot_loss_curve(losses, OUTPUT_DIR / "fusion_2node_loss_curve.png")
    else:
        print(f"日志文件 {log_file} 不存在")

    smape_data = {
        '2节点不加权基线': 60.45,
        '2节点粒度融合': 33.88,
        '41节点粒度融合': 40.81,
    }
    plot_accuracy_comparison(smape_data, OUTPUT_DIR / "fusion_accuracy_comparison.png")

if __name__ == "__main__":
    main()
