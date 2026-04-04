#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Analysis Script for Phase 1/2/3 Results
- Full comparison of accuracy, energy, cost, carbon
- Uncertainty evaluation for Phase 2/3 (MC Dropout)
- Decomposition by node, hour, decision type
- Statistical significance testing (McNemar)
- Threshold history visualization (if available)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar

# 设置 matplotlib
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "decision" / "analysis_outputs" / "advanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 文件路径
PHASE1_CSV = PROJECT_ROOT / "decision" / "outputs" / "decision_results_7day_all.csv"
PHASE2_CSV = PROJECT_ROOT / "decision" / "outputs" / "decision_results_7day_all_phase2.csv"
PHASE3_CSV = PROJECT_ROOT / "decision" / "outputs" / "decision_results_7day_all_phase3.csv"
THRESHOLD_HISTORY = PROJECT_ROOT / "decision" / "outputs" / "threshold_history_7day_all_phase3.csv"

def load_data():
    df1 = pd.read_csv(PHASE1_CSV) if PHASE1_CSV.exists() else None
    df2 = pd.read_csv(PHASE2_CSV) if PHASE2_CSV.exists() else None
    df3 = pd.read_csv(PHASE3_CSV) if PHASE3_CSV.exists() else None
    return df1, df2, df3

def add_confidence_from_std(df):
    """为 Phase 2/3 添加置信度列（基于预测标准差）"""
    if df is not None and 'pred_std_kw' in df.columns and 'pred_mean_kw' in df.columns:
        df['confidence'] = np.exp(-df['pred_std_kw'] / (df['pred_mean_kw'] + 1e-8))
        df['confidence'] = df['confidence'].clip(0, 1)
    return df

def plot_accuracy_comparison(df1, df2, df3):
    """准确率对比柱状图（带置信区间，bootstrap）"""
    def bootstrap_ci(series, n_bootstrap=1000):
        means = [np.mean(np.random.choice(series, size=len(series), replace=True)) for _ in range(n_bootstrap)]
        return np.percentile(means, [2.5, 97.5])
    
    accs, cis = [], []
    for df, name in [(df1, 'Phase 1'), (df2, 'Phase 2'), (df3, 'Phase 3')]:
        if df is not None:
            acc = df['decision_correct'].mean() * 100
            ci = bootstrap_ci(df['decision_correct']) * 100
            accs.append(acc)
            cis.append(ci)
        else:
            accs.append(np.nan)
            cis.append((np.nan, np.nan))
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Phase 1', 'Phase 2', 'Phase 3'], accs, yerr=[(ci[1]-ci[0])/2 for ci in cis], capsize=5, color=['#3498db', '#e67e22', '#2ecc71'])
    plt.ylabel('Decision Accuracy (%)')
    plt.title('Accuracy Comparison with 95% CI (Bootstrap)')
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.1f}%', ha='center')
    plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png', dpi=150)
    plt.close()

def plot_savings_comparison(df1, df2, df3):
    """节能、成本、碳排对比"""
    metrics = ['energy_saved_kwh', 'cost_saved_eur', 'carbon_saved_kg']
    labels = ['Energy Saved (MWh)', 'Cost Saved (k€)', 'Carbon Saved (t CO2)']
    factors = [1e6, 1e3, 1e3]
    values = {m: [] for m in metrics}
    for df, name in [(df1, 'Phase 1'), (df2, 'Phase 2'), (df3, 'Phase 3')]:
        if df is not None:
            for m in metrics:
                values[m].append(df[m].sum() / factors[metrics.index(m)])
        else:
            for m in metrics:
                values[m].append(np.nan)
    
    x = np.arange(3)
    width = 0.25
    plt.figure(figsize=(10, 6))
    for i, (m, lbl, fac) in enumerate(zip(metrics, labels, factors)):
        plt.bar(x + i*width, values[m], width, label=lbl)
    plt.xticks(x + width, ['Phase 1', 'Phase 2', 'Phase 3'])
    plt.ylabel('Amount')
    plt.title('Energy, Cost, Carbon Savings Comparison')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'savings_comparison.png', dpi=150)
    plt.close()

def plot_decision_distribution_stacked(df1, df2, df3):
    """决策类型分布堆叠图"""
    decisions = ['Sleep', 'Normal', 'Migration']
    props = []
    for df in [df1, df2, df3]:
        if df is not None:
            counts = df['decision'].value_counts()
            props.append([counts.get(d, 0)/len(df) for d in decisions])
        else:
            props.append([0,0,0])
    df_plot = pd.DataFrame(props, columns=decisions, index=['Phase 1', 'Phase 2', 'Phase 3'])
    df_plot.plot(kind='bar', stacked=True, figsize=(8,6), color=['#2ecc71', '#3498db', '#e67e22'])
    plt.ylabel('Proportion')
    plt.title('Decision Type Distribution')
    plt.legend(title='Decision')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'decision_distribution_stacked.png', dpi=150)
    plt.close()

def plot_node_accuracy(df, phase_name):
    if df is None: return
    node_acc = df.groupby('node_id')['decision_correct'].mean() * 100
    node_acc = node_acc.sort_values()
    plt.figure(figsize=(14,6))
    plt.bar(node_acc.index.astype(str), node_acc.values, color='steelblue')
    plt.axhline(y=node_acc.mean(), color='r', linestyle='--', label=f'Avg: {node_acc.mean():.1f}%')
    plt.xlabel('Node ID'); plt.ylabel('Accuracy (%)')
    plt.title(f'Node-wise Accuracy - {phase_name}')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'node_accuracy_{phase_name}.png', dpi=150)
    plt.close()

def plot_hour_accuracy_curve(df1, df2, df3):
    """按时段准确率折线图（三个阶段叠加）"""
    hour_accs = []
    labels = []
    for df, label in [(df1, 'Phase 1'), (df2, 'Phase 2'), (df3, 'Phase 3')]:
        if df is not None:
            acc = df.groupby('hour_code')['decision_correct'].mean() * 100
            hour_accs.append(acc)
            labels.append(label)
    if not hour_accs:
        return
    plt.figure(figsize=(12,6))
    for acc, label in zip(hour_accs, labels):
        plt.plot(acc.index, acc.values, marker='o', label=label)
    plt.xlabel('Hour Code'); plt.ylabel('Accuracy (%)')
    plt.title('Hourly Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hour_accuracy_curve.png', dpi=150)
    plt.close()

def plot_uncertainty_evaluation(df2, df3):
    """评估 MC Dropout 的不确定性质量"""
    for df, name in [(df2, 'Phase2'), (df3, 'Phase3')]:
        if df is None or 'pred_std_kw' not in df.columns:
            continue
        # 1. 标准差分桶 vs 绝对误差
        df['abs_error'] = np.abs(df['real_kw'] - df['pred_mean_kw'])
        df['std_bin'] = pd.qcut(df['pred_std_kw'], q=10, duplicates='drop')
        bin_stats = df.groupby('std_bin').agg({'abs_error': 'mean', 'pred_std_kw': 'mean'}).dropna()
        plt.figure(figsize=(8,6))
        plt.plot(bin_stats['pred_std_kw'], bin_stats['abs_error'], 'o-')
        plt.xlabel('Mean Predicted Std Dev (kWh)')
        plt.ylabel('Mean Absolute Error (kWh)')
        plt.title(f'Uncertainty Calibration - {name}')
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / f'uncertainty_calibration_{name}.png', dpi=150)
        plt.close()
        # 2. 散点图：预测均值 vs 真实值，颜色表示标准差
        sample = df.sample(min(5000, len(df)))
        plt.figure(figsize=(8,6))
        sc = plt.scatter(sample['pred_mean_kw'], sample['real_kw'], c=sample['pred_std_kw'], cmap='viridis', alpha=0.5, s=5)
        plt.colorbar(sc, label='Predicted Std Dev (kWh)')
        plt.plot([0, sample[['pred_mean_kw','real_kw']].max().max()], [0, sample[['pred_mean_kw','real_kw']].max().max()], 'r--')
        plt.xlabel('Predicted Mean (kWh)')
        plt.ylabel('True Value (kWh)')
        plt.title(f'Prediction Scatter with Uncertainty - {name}')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'prediction_scatter_{name}.png', dpi=150)
        plt.close()
        # 3. Spearman 相关系数
        corr, p = spearmanr(df['pred_std_kw'], df['abs_error'])
        print(f"{name}: Spearman correlation between std and abs error = {corr:.3f} (p={p:.4f})")

def plot_reliability_from_std(df, phase_name):
    """利用预测标准差构造置信度，绘制可靠性曲线"""
    if df is None or 'pred_std_kw' not in df.columns:
        return
    df = df.copy()
    df['confidence'] = np.exp(-df['pred_std_kw'] / (df['pred_mean_kw'] + 1e-8))
    df['confidence'] = df['confidence'].clip(0, 1)
    bins = np.linspace(0, 1, 11)
    df['conf_bin'] = pd.cut(df['confidence'], bins, include_lowest=True)
    reliability = df.groupby('conf_bin').agg(
        mean_conf=('confidence', 'mean'),
        accuracy=('decision_correct', 'mean')
    ).dropna()
    plt.figure(figsize=(8,6))
    plt.plot([0,1], [0,1], 'k--', label='Perfect')
    plt.plot(reliability['mean_conf'], reliability['accuracy'], 'o-', label='Model')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Actual Accuracy')
    plt.title(f'Reliability Diagram (from MC Dropout) - {phase_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / f'reliability_{phase_name}.png', dpi=150)
    plt.close()

def mcnemar_test(df1, df2, phase_pair):
    """McNemar 检验两个阶段决策正确性是否有显著差异"""
    if df1 is None or df2 is None:
        return
    # 确保两个 DataFrame 按相同顺序合并（按 node_id, date, hour_code, is_holiday）
    merge_cols = ['node_id', 'date', 'hour_code', 'is_holiday']
    merged = df1.merge(df2, on=merge_cols, suffixes=('_1', '_2'))
    # 构建列联表
    correct1 = merged['decision_correct_1'].astype(int)
    correct2 = merged['decision_correct_2'].astype(int)
    table = np.array([[((correct1==1) & (correct2==1)).sum(), ((correct1==1) & (correct2==0)).sum()],
                      [((correct1==0) & (correct2==1)).sum(), ((correct1==0) & (correct2==0)).sum()]])
    result = mcnemar(table, exact=False, correction=True)
    print(f"McNemar test {phase_pair}: p-value = {result.pvalue:.6f}")
    return result.pvalue

def plot_threshold_history():
    if not THRESHOLD_HISTORY.exists():
        return
    df = pd.read_csv(THRESHOLD_HISTORY)
    if df.empty:
        return
    # 选取前5个最常更新的 (hour_code, is_holiday)
    key_counts = df.groupby(['hour_code', 'is_holiday']).size().nlargest(5).index
    plt.figure(figsize=(12,8))
    for (hc, hd) in key_counts:
        subset = df[(df['hour_code']==hc) & (df['is_holiday']==hd)]
        if len(subset) > 0:
            plt.plot(pd.to_datetime(subset['timestamp']), subset['high'], label=f'Hour {hc}, Holiday={hd} (high)')
            plt.plot(pd.to_datetime(subset['timestamp']), subset['low'], '--', label=f'Hour {hc}, Holiday={hd} (low)')
    plt.xlabel('Time')
    plt.ylabel('Threshold (kWh)')
    plt.title('Dynamic Threshold Evolution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'threshold_history.png', dpi=150)
    plt.close()

def generate_summary_table(df1, df2, df3):
    """生成汇总统计表并保存为 CSV"""
    rows = []
    for df, name in [(df1, 'Phase 1'), (df2, 'Phase 2'), (df3, 'Phase 3')]:
        if df is None:
            continue
        row = {
            'Phase': name,
            'Accuracy (%)': df['decision_correct'].mean() * 100,
            'Total Energy Saved (MWh)': df['energy_saved_kwh'].sum() / 1e6,
            'Total Cost Saved (k€)': df['cost_saved_eur'].sum() / 1e3,
            'Total Carbon Saved (t CO2)': df['carbon_saved_kg'].sum() / 1e3,
            'Num Samples': len(df),
            'Num Nodes': df['node_id'].nunique(),
        }
        if 'pred_std_kw' in df.columns:
            row['Mean Pred Std (kWh)'] = df['pred_std_kw'].mean()
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / 'summary_table.csv', index=False)
    print("Summary table saved to", OUTPUT_DIR / 'summary_table.csv')

def main():
    print("Loading data...")
    df1, df2, df3 = load_data()
    df2 = add_confidence_from_std(df2)
    df3 = add_confidence_from_std(df3)
    
    print("Generating plots...")
    plot_accuracy_comparison(df1, df2, df3)
    plot_savings_comparison(df1, df2, df3)
    plot_decision_distribution_stacked(df1, df2, df3)
    plot_hour_accuracy_curve(df1, df2, df3)
    for df, name in [(df1, 'Phase1'), (df2, 'Phase2'), (df3, 'Phase3')]:
        if df is not None:
            plot_node_accuracy(df, name)
            plot_reliability_from_std(df, name)
    plot_uncertainty_evaluation(df2, df3)
    plot_threshold_history()
    generate_summary_table(df1, df2, df3)
    
    print("Statistical tests:")
    if df1 is not None and df2 is not None:
        mcnemar_test(df1, df2, "Phase1 vs Phase2")
    if df2 is not None and df3 is not None:
        mcnemar_test(df2, df3, "Phase2 vs Phase3")
    
    print("All analysis completed. Outputs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()