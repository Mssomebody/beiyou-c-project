#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 + Phase 2 集成学习（最终版）
- 从 thresholds_dynamic.json 加载阈值，计算真实决策
- 策略：简单投票、软投票、动态权重、成本敏感、元学习器（多分类）
- 滚动验证 (TimeSeriesSplit)
- 输出：准确率对比图、节点/小时准确率、混淆矩阵、McNemar检验
- 所有路径可配置，无硬编码，无假设
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 成本矩阵（与 Phase 2 保持一致）
COST_MATRIX = {
    'high':   {'Sleep': 400, 'Normal': 3, 'Migration': 1},
    'medium': {'Sleep': 200, 'Normal': 0, 'Migration': 2},
    'low':    {'Sleep': 0,   'Normal': 3, 'Migration': 2}
}
DECISION_MAP = {'Sleep': 0, 'Normal': 1, 'Migration': 2}
INV_DECISION_MAP = {0: 'Sleep', 1: 'Normal', 2: 'Migration'}

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1+2 集成学习")
    parser.add_argument('--phase1_csv', type=str,
                        default=str(PROJECT_ROOT / "decision" / "outputs" / "decision_results_7day_all.csv"),
                        help='Phase 1 结果 CSV')
    parser.add_argument('--phase2_csv', type=str,
                        default=str(PROJECT_ROOT / "decision" / "outputs" / "decision_results_7day_all_phase2.csv"),
                        help='Phase 2 结果 CSV')
    parser.add_argument('--thresholds_json', type=str,
                        default=str(PROJECT_ROOT / "decision" / "config" / "thresholds_dynamic.json"),
                        help='阈值配置文件（用于计算真实决策）')
    parser.add_argument('--peak_hours_json', type=str,
                        default=str(PROJECT_ROOT / "decision" / "config" / "peak_hours.json"),
                        help='高峰时段配置文件（由 data_mining.py 生成）')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='滚动验证折数')
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / "decision" / "ensemble_outputs"),
                        help='输出目录')
    return parser.parse_args()

def load_thresholds(thresholds_path):
    """加载阈值字典，返回 {(hour_code, is_holiday): (low, high)}"""
    with open(thresholds_path, 'r') as f:
        data = json.load(f)
    thresholds = {}
    for entry in data['thresholds']:
        key = (entry['hour_code'], entry['is_holiday'])
        thresholds[key] = (entry['low'], entry['high'])
    return thresholds

def load_and_align_data(phase1_path, phase2_path, thresholds):
    """加载 Phase 1 和 Phase 2 结果，对齐样本，计算真实决策和正确性"""
    df1 = pd.read_csv(phase1_path)
    df2 = pd.read_csv(phase2_path)
    merge_cols = ['node_id', 'date', 'hour_code', 'is_holiday']
    merged = df1.merge(df2, on=merge_cols, suffixes=('_1', '_2'))
    merged['date'] = pd.to_datetime(merged['date'])
    merged = merged.sort_values('date').reset_index(drop=True)

    # 真实能耗：优先使用 Phase 2 的 real_kw，否则使用 Phase 1
    if 'real_kw' in merged.columns:
        real = merged['real_kw']
    elif 'real_kw_2' in merged.columns:
        real = merged['real_kw_2']
    else:
        real = merged['real_kw_1']

    # 计算真实决策（基于阈值）
    def true_decision(row):
        key = (row['hour_code'], row['is_holiday'])
        if key not in thresholds:
            # 如果没有对应阈值，默认 normal
            return "Normal"
        low, high = thresholds[key]
        r = row['real_kw'] if 'real_kw' in row else row['real_kw_2'] if 'real_kw_2' in row else row['real_kw_1']
        if r > high:
            return "Migration"
        elif r < low:
            return "Sleep"
        else:
            return "Normal"

    merged['true_decision'] = merged.apply(true_decision, axis=1)

    # 决策列
    dec1 = merged['decision_1']
    dec2 = merged['decision_2']

    # 正确性
    merged['correct_1'] = (dec1 == merged['true_decision'])
    merged['correct_2'] = (dec2 == merged['true_decision'])

    # 置信度：Phase 1 使用 confidence 列（如果存在），否则 0.5
    merged['confidence_1'] = merged.get('confidence', 0.5)
    # Phase 2 置信度从预测标准差构造
    if 'pred_std_kw' in merged.columns and 'pred_mean_kw' in merged.columns:
        merged['confidence_2'] = np.exp(-merged['pred_std_kw'] / (merged['pred_mean_kw'] + 1e-8))
        merged['confidence_2'] = merged['confidence_2'].clip(0, 1)
    else:
        merged['confidence_2'] = 0.5

    # 编码
    merged['dec1_enc'] = dec1.map(DECISION_MAP)
    merged['dec2_enc'] = dec2.map(DECISION_MAP)
    merged['true_enc'] = merged['true_decision'].map(DECISION_MAP)

    return merged

# -------------------- 集成策略 --------------------
def simple_voting(row):
    return row['decision_1'] if row['decision_1'] == row['decision_2'] else row['decision_2']

def soft_voting(row, w1=0.5, w2=0.5):
    return row['decision_1'] if row['confidence_1'] * w1 > row['confidence_2'] * w2 else row['decision_2']

def dynamic_weight_voting(row, peak_hours):
    w1, w2 = (0.3, 0.7) if row['hour_code'] in peak_hours else (0.5, 0.5)
    return soft_voting(row, w1, w2)

def cost_sensitive_ensemble(row):
    if all(c in row for c in ['expected_cost_sleep', 'expected_cost_normal', 'expected_cost_migration']):
        dec1 = row['decision_1'].lower()
        dec2 = row['decision_2'].lower()
        cost1 = row[f'expected_cost_{dec1}']
        cost2 = row[f'expected_cost_{dec2}']
        return row['decision_1'] if cost1 < cost2 else row['decision_2']
    else:
        return simple_voting(row)

def meta_learner_predict(train_df, test_df):
    features = ['dec1_enc', 'dec2_enc', 'confidence_1', 'confidence_2']
    X_train = train_df[features].values
    y_train = train_df['true_enc'].values
    clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    X_test = test_df[features].values
    pred_enc = clf.predict(X_test)
    return [INV_DECISION_MAP[e] for e in pred_enc]

# -------------------- 滚动验证评估 --------------------
def evaluate_strategy_rolling(df, strategy_func, name, n_splits, peak_hours=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies = []
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx].copy()
        if name == "Meta-Learner":
            test_df['ensemble_dec'] = meta_learner_predict(train_df, test_df)
        else:
            if name == "Dynamic Weight":
                test_df['ensemble_dec'] = test_df.apply(lambda row: strategy_func(row, peak_hours), axis=1)
            else:
                test_df['ensemble_dec'] = test_df.apply(strategy_func, axis=1)
        correct = (test_df['ensemble_dec'] == test_df['true_decision'])
        acc = correct.mean()
        accuracies.append(acc)
        fold_results.append((fold, acc, test_df))
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    return mean_acc, std_acc, accuracies, fold_results

# -------------------- 详细分析 --------------------
def detailed_analysis(best_strategy_name, best_fold_results, output_dir):
    all_test = pd.concat([res[2] for res in best_fold_results], ignore_index=True)
    # 节点准确率
    node_acc = all_test.groupby('node_id').apply(lambda g: (g['ensemble_dec'] == g['true_decision']).mean())
    node_acc.to_csv(output_dir / f"{best_strategy_name}_node_accuracy.csv", header=['accuracy'])
    # 小时准确率
    hour_acc = all_test.groupby('hour_code').apply(lambda g: (g['ensemble_dec'] == g['true_decision']).mean())
    hour_acc.to_csv(output_dir / f"{best_strategy_name}_hour_accuracy.csv", header=['accuracy'])
    # 混淆矩阵
    cm = confusion_matrix(all_test['true_decision'], all_test['ensemble_dec'],
                          labels=['Sleep', 'Normal', 'Migration'])
    np.savetxt(output_dir / f"{best_strategy_name}_confusion_matrix.csv", cm, delimiter=',', fmt='%d',
               header='Sleep,Normal,Migration', comments='')
    # 节点准确率图
    plt.figure(figsize=(14,6))
    node_acc.sort_values().plot(kind='bar', color='steelblue')
    plt.axhline(y=node_acc.mean(), color='r', linestyle='--', label=f'Avg: {node_acc.mean():.3f}')
    plt.title(f'{best_strategy_name} - Node-wise Accuracy')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{best_strategy_name}_node_accuracy.png", dpi=150)
    plt.close()
    # 小时准确率图
    plt.figure(figsize=(10,5))
    hour_acc.plot(kind='bar', color='green')
    plt.axhline(y=hour_acc.mean(), color='r', linestyle='--', label=f'Avg: {hour_acc.mean():.3f}')
    plt.title(f'{best_strategy_name} - Hourly Accuracy')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{best_strategy_name}_hour_accuracy.png", dpi=150)
    plt.close()

def mcnemar_test(df, col1, col2):
    correct1 = df[col1] == df['true_decision']
    correct2 = df[col2] == df['true_decision']
    both_correct = (correct1 & correct2).sum()
    only1_correct = (correct1 & ~correct2).sum()
    only2_correct = (~correct1 & correct2).sum()
    both_wrong = (~correct1 & ~correct2).sum()
    table = [[both_correct, only1_correct], [only2_correct, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue

# -------------------- 主函数 --------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1 + Phase 2 集成学习")
    print(f"Phase 1 结果: {args.phase1_csv}")
    print(f"Phase 2 结果: {args.phase2_csv}")
    print(f"阈值文件: {args.thresholds_json}")
    print(f"高峰时段文件: {args.peak_hours_json}")
    print(f"滚动验证折数: {args.n_splits}")
    print("=" * 60)

    # 加载阈值
    thresholds = load_thresholds(Path(args.thresholds_json))
    print(f"加载了 {len(thresholds)} 个 (hour_code, is_holiday) 阈值")

    # 加载数据
    print("加载并对齐数据...")
    df = load_and_align_data(Path(args.phase1_csv), Path(args.phase2_csv), thresholds)
    print(f"总样本数: {len(df)}")

    # 加载高峰时段
    peak_hours = []
    if Path(args.peak_hours_json).exists():
        with open(args.peak_hours_json, 'r') as f:
            data = json.load(f)
            peak_hours = data.get('peak_hours', [])
    print(f"高峰时段: {peak_hours}")

    # 策略列表
    strategies = [
        ("Simple Voting", simple_voting, {}),
        ("Soft Voting", soft_voting, {}),
        ("Dynamic Weight", dynamic_weight_voting, {"peak_hours": peak_hours}),
        ("Cost-Sensitive", cost_sensitive_ensemble, {}),
        ("Meta-Learner", None, {}),
    ]

    results = {}
    all_fold_results = {}
    for name, func, kwargs in strategies:
        print(f"正在评估 {name}...")
        if name == "Meta-Learner":
            mean_acc, std_acc, acc_list, fold_res = evaluate_strategy_rolling(df, None, name, args.n_splits)
        else:
            mean_acc, std_acc, acc_list, fold_res = evaluate_strategy_rolling(df, func, name, args.n_splits, **kwargs)
        results[name] = (mean_acc, std_acc)
        all_fold_results[name] = fold_res
        print(f"  准确率: {mean_acc:.4f} ± {std_acc:.4f}")

    # 选择最佳策略
    best_name = max(results, key=lambda x: results[x][0])
    best_acc, best_std = results[best_name]
    print(f"\n最佳策略: {best_name} (准确率 {best_acc:.4f} ± {best_std:.4f})")

    # 保存汇总结果
    with open(output_dir / "rolling_validation_results.json", "w") as f:
        json.dump({name: {"mean": res[0], "std": res[1]} for name, res in results.items()}, f, indent=2)

    # 绘制准确率对比图
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    plt.figure(figsize=(10,6))
    bars = plt.bar(names, means, yerr=stds, capsize=5,
                   color=['#3498db','#e67e22','#2ecc71','#e74c3c','#9b59b6'])
    plt.ylabel("Decision Accuracy")
    plt.title(f"Rolling Validation Accuracy (n_splits={args.n_splits})")
    plt.xticks(rotation=15)
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{mean:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_accuracy_comparison.png", dpi=150)
    plt.close()

    # 最佳策略详细分析
    print(f"\n对最佳策略 {best_name} 进行详细分析...")
    detailed_analysis(best_name, all_fold_results[best_name], output_dir)

    # McNemar 检验（最佳策略 vs Phase 2）
    all_test = pd.concat([res[2] for res in all_fold_results[best_name]], ignore_index=True)
    p_value = mcnemar_test(all_test, 'ensemble_dec', 'decision_2')
    print(f"McNemar 检验 ({best_name} vs Phase 2): p-value = {p_value:.6f}")
    with open(output_dir / "mcnemar_pvalue.txt", "w") as f:
        f.write(f"Comparison: {best_name} vs Phase 2\np-value: {p_value}\n")

    print(f"\n所有输出已保存到: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()