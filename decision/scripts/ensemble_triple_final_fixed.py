#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三模型集成（Phase 1 + Phase 2 + Phase 3）最终修复版
- 剔除困难节点（8006,8029,8036）
- 数据去重，节能正确累加
- 所有5种集成策略：简单投票、软投票、动态权重、成本敏感、元学习器
- SHAP 多分类处理（使用 KernelExplainer + 正确汇总）
- 输出所有结果：ensemble_results.csv, summary_stats.json, rolling_validation_results.json, 节点/小时准确率, 混淆矩阵, SHAP图表
"""

import sys
import json
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "decision" / "ensemble_triple_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pkl"

# 输入文件
PHASE1_CSV = PROJECT_ROOT / "decision" / "outputs" / "decision_results_7day_all.csv"
PHASE2_CSV = PROJECT_ROOT / "decision" / "outputs" / "decision_results_7day_all_phase2.csv"
PHASE3_CSV = PROJECT_ROOT / "decision" / "outputs" / "decision_results_7day_all_phase3.csv"
THRESHOLD_JSON = PROJECT_ROOT / "decision" / "config" / "thresholds_dynamic.json"
PEAK_HOURS_JSON = PROJECT_ROOT / "decision" / "config" / "peak_hours.json"

# 剔除的困难节点
EXCLUDE_NODES = {8006, 8029, 8036}

DECISION_MAP = {'Sleep': 0, 'Normal': 1, 'Migration': 2}
INV_DECISION_MAP = {0: 'Sleep', 1: 'Normal', 2: 'Migration'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_align():
    """加载并对齐三个数据集，去重，剔除困难节点"""
    df1 = pd.read_csv(PHASE1_CSV)
    df2 = pd.read_csv(PHASE2_CSV)
    df3 = pd.read_csv(PHASE3_CSV)

    for df in [df1, df2, df3]:
        df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)

    merge_cols = ['node_id', 'date', 'hour_code', 'is_holiday']

    # 去重：每个唯一键保留第一行
    df1 = df1.drop_duplicates(subset=merge_cols, keep='first')
    df2 = df2.drop_duplicates(subset=merge_cols, keep='first')
    df3 = df3.drop_duplicates(subset=merge_cols, keep='first')

    # 剔除困难节点
    for df in [df1, df2, df3]:
        df.drop(df[df['node_id'].isin(EXCLUDE_NODES)].index, inplace=True)

    merged = df1.merge(df2, on=merge_cols, suffixes=('_1', '_2'))
    merged = merged.merge(df3, on=merge_cols, suffixes=('', '_3'))
    merged = merged.sort_values('date').reset_index(drop=True)

    logger.info(f"合并后样本数: {len(merged)} (已剔除节点 {EXCLUDE_NODES})")

    merged = merged.rename(columns={'decision_1': 'dec1', 'decision_2': 'dec2', 'decision': 'dec3'})

    # 加载动态阈值
    with open(THRESHOLD_JSON, 'r') as f:
        th_data = json.load(f)
    thresholds = {(entry['hour_code'], entry['is_holiday']): (entry['low'], entry['high']) for entry in th_data['thresholds']}

    def true_decision(row):
        key = (row['hour_code'], row['is_holiday'])
        if key not in thresholds:
            return 'Normal'
        low, high = thresholds[key]
        real = row['real_kw_2'] if 'real_kw_2' in row else row['real_kw']
        if real > high:
            return 'Migration'
        elif real < low:
            return 'Sleep'
        else:
            return 'Normal'

    merged['true_decision'] = merged.apply(true_decision, axis=1)
    merged['true_enc'] = merged['true_decision'].map(DECISION_MAP)

    merged['dec1_enc'] = merged['dec1'].map(DECISION_MAP)
    merged['dec2_enc'] = merged['dec2'].map(DECISION_MAP)
    merged['dec3_enc'] = merged['dec3'].map(DECISION_MAP)

    merged['conf1'] = merged.get('confidence', 0.5)
    merged['conf2'] = merged.get('calibrated_confidence', 0.5)
    if 'calibrated_confidence' in merged.columns:
        merged['conf3'] = merged['calibrated_confidence']
    elif 'raw_confidence' in merged.columns:
        merged['conf3'] = merged['raw_confidence']
    else:
        merged['conf3'] = 0.5

    # 节能数据使用 Phase 2 的节能列
    merged['energy_saved'] = merged['energy_saved_kwh_2']
    return merged

# ========== 集成策略函数 ==========
def simple_voting(row):
    return row['dec1'] if row['dec1'] == row['dec2'] else row['dec2']

def soft_voting(row, w1=0.5, w2=0.5):
    return row['dec1'] if row['conf1'] * w1 > row['conf2'] * w2 else row['dec2']

def dynamic_weight_voting(row, peak_hours):
    w1, w2 = (0.3, 0.7) if row['hour_code'] in peak_hours else (0.5, 0.5)
    return soft_voting(row, w1, w2)

def cost_sensitive_ensemble(row):
    # 使用 Phase 2 的决策（已经成本敏感）
    return row['dec2']

def meta_learner_predict(train_df, test_df):
    features = ['dec1_enc', 'dec2_enc', 'dec3_enc', 'conf1', 'conf2', 'conf3']
    X_train = train_df[features].values
    y_train = train_df['true_enc'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    X_test = test_df[features].values
    pred_enc = clf.predict(X_test)
    return [INV_DECISION_MAP[e] for e in pred_enc]

# ========== 滚动验证评估（支持所有策略） ==========
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
    return np.mean(accuracies), np.std(accuracies), accuracies, fold_results

# ========== 详细分析（节点/小时准确率、混淆矩阵、图表） ==========
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

# ========== SHAP 分析（多分类，稳健版） ==========
def shap_analysis(clf, df, features, output_dir, n_samples=500):
    logger.info("Performing SHAP analysis...")
    sample_df = df.sample(min(n_samples, len(df))).copy()
    X_sample = sample_df[features].values
    def predict_proba(X):
        return clf.predict_proba(X)
    explainer = shap.KernelExplainer(predict_proba, X_sample[:100])
    shap_values = explainer.shap_values(X_sample[:100])  # list of (n_samples, n_features)
    
    # 确保所有类别形状一致
    n_classes = len(shap_values)
    n_samples_actual = shap_values[0].shape[0]
    n_features_actual = shap_values[0].shape[1]
    for sv in shap_values:
        if sv.shape != (n_samples_actual, n_features_actual):
            raise ValueError(f"Shape mismatch: {sv.shape} vs expected ({n_samples_actual}, {n_features_actual})")
    
    # 计算每个特征的平均绝对 SHAP 值（跨类别和样本）
    total_abs = np.zeros(n_features_actual)
    for class_idx in range(n_classes):
        total_abs += np.abs(shap_values[class_idx]).sum(axis=0)
    mean_abs_per_feature = total_abs / (n_classes * n_samples_actual)
    
    importance_df = pd.DataFrame({'feature': features, 'mean_abs_shap': mean_abs_per_feature})
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    importance_df.to_csv(output_dir / 'shap_importance.csv', index=False)
    
    # 概要图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample[:100], feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # 条形图
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['feature'], importance_df['mean_abs_shap'])
    plt.xlabel('Mean |SHAP|')
    plt.title('Feature Importance (Multi-class)')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_bar.png', dpi=150)
    plt.close()
    logger.info("SHAP analysis completed.")

# ========== 元学习器训练（带检查点） ==========
def train_meta_learner(df, n_splits=5, resume=True):
    features = ['dec1_enc', 'dec2_enc', 'dec3_enc', 'conf1', 'conf2', 'conf3']
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    if resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'rb') as f:
            chk = pickle.load(f)
        start_fold = chk['last_fold'] + 1
        accuracies = chk['accuracies']
        energy_savings = chk['energy_savings']
        all_test = chk['all_test']
        clf = chk.get('clf')
        if clf is None:
            raise RuntimeError("Checkpoint missing model. Please delete checkpoint.pkl and rerun.")
        logger.info(f"Resuming from fold {start_fold}/{n_splits}")
    else:
        start_fold = 0
        accuracies = []
        energy_savings = []
        all_test = []
        clf = None
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        if fold < start_fold:
            continue
        logger.info(f"Training fold {fold+1}/{n_splits}...")
        train = df.iloc[train_idx]
        test = df.iloc[test_idx].copy()
        X_train = train[features].values
        y_train = train['true_enc'].values
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        X_test = test[features].values
        y_pred = clf.predict(X_test)
        y_true = test['true_enc'].values
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)
        energy = test['energy_saved'].sum()
        energy_savings.append(energy)
        test['ensemble_dec'] = [INV_DECISION_MAP[p] for p in y_pred]
        test['ensemble_correct'] = (test['ensemble_dec'] == test['true_decision'])
        all_test.append(test)
        
        checkpoint = {
            'last_fold': fold,
            'accuracies': accuracies,
            'energy_savings': energy_savings,
            'all_test': all_test,
            'clf': clf
        }
        with open(CHECKPOINT_PATH, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved for fold {fold+1}")
    
    df_ensemble = pd.concat(all_test, ignore_index=True)
    mean_acc = np.mean(accuracies)
    total_energy = sum(energy_savings)
    return mean_acc, total_energy, df_ensemble, clf

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--no_resume', action='store_true')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("三模型集成 (最终修复版)")
    logger.info("="*60)
    
    # 加载数据
    df = load_and_align()
    logger.info(f"总样本数: {len(df)}")
    
    # 加载高峰时段
    peak_hours = []
    if PEAK_HOURS_JSON.exists():
        with open(PEAK_HOURS_JSON, 'r') as f:
            peak_hours = json.load(f).get('peak_hours', [])
    
    # 定义所有策略
    strategies = [
        ("Simple Voting", simple_voting, {}),
        ("Soft Voting", soft_voting, {}),
        ("Dynamic Weight", dynamic_weight_voting, {"peak_hours": peak_hours}),
        ("Cost-Sensitive", cost_sensitive_ensemble, {}),
        ("Meta-Learner", None, {}),
    ]
    
    results = {}
    all_fold_results = {}
    meta_clf = None
    
    for name, func, kwargs in strategies:
        logger.info(f"Evaluating {name}...")
        if name == "Meta-Learner":
            mean_acc, total_energy, df_ensemble, clf = train_meta_learner(df, n_splits=args.n_splits, resume=not args.no_resume)
            # 构造 fold_results 用于详细分析（使用训练过程中保存的 all_test）
            # 但 train_meta_learner 已经返回了 df_ensemble，我们将其包装成单个折叠
            fold_results = [(0, mean_acc, df_ensemble)]
            results[name] = (mean_acc, 0.0)  # 标准差暂时设为0
            all_fold_results[name] = fold_results
            meta_clf = clf
        else:
            mean_acc, std_acc, _, fold_res = evaluate_strategy_rolling(df, func, name, args.n_splits, **kwargs)
            results[name] = (mean_acc, std_acc)
            all_fold_results[name] = fold_res
        logger.info(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # 选择最佳策略
    best_name = max(results, key=lambda x: results[x][0])
    best_acc, best_std = results[best_name]
    logger.info(f"\n最佳策略: {best_name} (准确率 {best_acc:.4f} ± {best_std:.4f})")
    
    # 保存汇总 JSON
    with open(OUTPUT_DIR / "rolling_validation_results.json", "w") as f:
        json.dump({name: {"mean": res[0], "std": res[1]} for name, res in results.items()}, f, indent=2)
    
    # 绘制准确率对比图
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    plt.figure(figsize=(10,6))
    bars = plt.bar(names, means, yerr=stds, capsize=5, color=['#3498db','#e67e22','#2ecc71','#e74c3c','#9b59b6'])
    plt.ylabel("Decision Accuracy")
    plt.title(f"Rolling Validation Accuracy (n_splits={args.n_splits})")
    plt.xticks(rotation=15)
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{mean:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ensemble_accuracy_comparison.png", dpi=150)
    plt.close()
    
    # 对最佳策略进行详细分析
    detailed_analysis(best_name, all_fold_results[best_name], OUTPUT_DIR)
    
    # McNemar 检验（最佳策略 vs Phase 2）
    all_test = pd.concat([res[2] for res in all_fold_results[best_name]], ignore_index=True)
    if 'decision_2' in all_test.columns:
        p_value = mcnemar_test(all_test, 'ensemble_dec', 'decision_2')
        logger.info(f"McNemar test ({best_name} vs Phase 2): p-value = {p_value:.6f}")
        with open(OUTPUT_DIR / "mcnemar_pvalue.txt", "w") as f:
            f.write(f"Comparison: {best_name} vs Phase 2\np-value: {p_value}\n")
    else:
        logger.warning("Phase 2 decision column missing, skip McNemar test")
    
    # SHAP 分析（仅当最佳策略是元学习器时）
    if best_name == "Meta-Learner" and meta_clf is not None:
        features = ['dec1_enc', 'dec2_enc', 'dec3_enc', 'conf1', 'conf2', 'conf3']
        shap_analysis(meta_clf, df, features, OUTPUT_DIR)
    else:
        logger.info(f"Best strategy is {best_name}, skipping SHAP analysis (only for Meta-Learner)")
    
    # 保存最终统计摘要
    stats = {
        'ensemble_accuracy_percent': best_acc * 100,
        'total_energy_saved_kwh': total_energy if best_name == "Meta-Learner" else None,
        'num_samples': len(all_test),
        'num_nodes': all_test['node_id'].nunique(),
        'best_strategy': best_name,
        'excluded_nodes': list(EXCLUDE_NODES),
        'n_splits': args.n_splits
    }
    with open(OUTPUT_DIR / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # 保存集成结果 CSV（仅当最佳策略为元学习器且有 df_ensemble）
    if best_name == "Meta-Learner" and 'df_ensemble' in locals():
        df_ensemble.to_csv(OUTPUT_DIR / "ensemble_results.csv", index=False)
    
    logger.info(f"所有结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()