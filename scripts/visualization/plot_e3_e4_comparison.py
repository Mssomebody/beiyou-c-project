#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 E3 (基础粒度融合) 和 E4 (知识迁移加权) 的结果
- 从日志文件提取训练损失和最终 sMAPE
- 生成损失曲线对比图和精度柱状图
"""

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

def extract_smape_from_log(log_file):
    for enc in ['utf-8', 'gbk', 'gb2312']:
        try:
            with open(log_file, 'r', encoding=enc) as f:
                content = f.read()
                m = re.search(r'最终 sMAPE: ([\d\.]+)%', content)
                if m:
                    return float(m.group(1))
        except:
            continue
    return None

def main():
    # E3 日志（假设为 fusion_2node.log）
    e3_log = LOG_DIR / "fusion_2node.log"
    e4_log = LOG_DIR / "fusion_time_weighted_2node.log"  # 运行后生成

    e3_loss = extract_loss_from_log(e3_log) if e3_log.exists() else []
    e4_loss = extract_loss_from_log(e4_log) if e4_log.exists() else []

    e3_smape = extract_smape_from_log(e3_log) if e3_log.exists() else None
    e4_smape = extract_smape_from_log(e4_log) if e4_log.exists() else None

    # 损失曲线对比
    if e3_loss or e4_loss:
        plt.figure(figsize=(10, 6))
        if e3_loss:
            plt.plot(range(1, len(e3_loss)+1), e3_loss, marker='o', label='E3 (基础粒度融合)', color='#2E8B57')
        if e4_loss:
            plt.plot(range(1, len(e4_loss)+1), e4_loss, marker='s', label='E4 (知识迁移加权)', color='#E76F51')
        plt.xlabel('轮次')
        plt.ylabel('平均训练损失')
        plt.title('粒度融合损失曲线对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "e3_e4_loss_comparison.png", dpi=150)
        plt.close()
        print("✅ 损失对比图保存至 results/figures/e3_e4_loss_comparison.png")

    # 精度柱状图
    if e3_smape is not None and e4_smape is not None:
        labels = ['E3 (基础粒度融合)', 'E4 (知识迁移加权)']
        values = [e3_smape, e4_smape]
        colors = ['#2E8B57', '#E76F51']
        plt.figure(figsize=(6, 6))
        bars = plt.bar(labels, values, color=colors, edgecolor='black')
        plt.ylabel('sMAPE (%)')
        plt.title('粒度融合精度对比')
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom')
        plt.ylim(0, max(values) + 10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "e3_e4_smape_comparison.png", dpi=150)
        plt.close()
        print("✅ 精度对比图保存至 results/figures/e3_e4_smape_comparison.png")
    else:
        print("缺少 E3 或 E4 的 sMAPE 数据，无法生成对比图。")

if __name__ == "__main__":
    main()
