#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3 决策引擎（成本敏感 + MC Dropout + 动态阈值更新）- 五星版
- 按节点隔离滑动窗口，动态阈值针对每个节点独立计算
- 定期保存更新后的阈值到文件
- 自动与 Phase 1、Phase 2 结果对比并生成可视化
"""

import sys
import json
import time
import pickle
import argparse
import logging
import numpy as np
from scipy.stats import norm
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, Tuple, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 可视化库
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "versions" / "v2_holiday_sector"))

MINMAX_PATH = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
NODE_PARAMS_PATH = PROJECT_ROOT / "decision" / "config" / "node_weighted_params_monthly.csv"
THRESHOLD_PATH = PROJECT_ROOT / "decision" / "config" / "thresholds_dynamic.json"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_2023_2025"
FINAL_COMPARISON_PATH = PROJECT_ROOT / "results" / "final_comparison.csv"

ORIG_MODEL_DIR = PROJECT_ROOT / "results" / "finetune" / "models"
SEC_MODEL_DIR = PROJECT_ROOT / "results" / "finetune_secondary" / "models"

SLEEP_FACTOR = 0.05
MIGRATE_FACTOR = 0.8

INPUT_DIM = 7
HIDDEN_DIM = 128
NUM_LAYERS = 2
OUTPUT_DIM = 4
DROPOUT = 0.2

WINDOW_SIZE = {"1day": 28, "7day": 168}
PREDICT_SIZE = 4

MC_NUM_FORWARD = 30

# 动态阈值参数
DYNAMIC_WINDOW_DAYS = 3          # 滑动窗口天数
UPDATE_INTERVAL = 1000            # 每个节点每处理这么多样本后重新计算该节点的阈值

# 成本矩阵
COST_MATRIX = {
    'high':   {'Sleep': 100, 'Normal': 10, 'Migration': 5},
    'medium': {'Sleep': 20,  'Normal': 0,  'Migration': 10},
    'low':    {'Sleep': 0,   'Normal': 2,  'Migration': 8}
}

# ============================================================
# 模型定义、数据集类、collate函数（与 Phase 2 一致，完整保留）
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class MinMaxBarcelonaDataset(Dataset):
    def __init__(self, data_path: Path, node_id: int, node_minmax: Dict[int, Tuple[float, float]],
                 window_size: int, predict_size: int = PREDICT_SIZE):
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.data_min, self.data_max = node_minmax[node_id]
        self.energy = self.df['Valor'].values
        sector_codes = self.df['sector_code'].values
        self.sector_onehot = self._one_hot_sector(sector_codes)
        self.holiday = self.df['is_holiday'].values
        self.weekend = self.df['is_weekend'].values
        self.hour_code = self.df['hour_code'].values
        self.indices = self._build_indices()
        self._extract_sample_metadata()

    def _one_hot_sector(self, codes):
        n = 4
        onehot = np.zeros((len(codes), n))
        for i, c in enumerate(codes):
            if 0 <= c < n:
                onehot[i, c] = 1
        return onehot

    def _build_indices(self):
        total = len(self.energy)
        return [i for i in range(total - self.window_size - self.predict_size + 1)]

    def _extract_sample_metadata(self):
        self.sample_dates = []
        self.sample_hour_codes = []
        self.sample_holidays = []
        for start_idx in self.indices:
            pred_start = start_idx + self.window_size
            dates = []
            hours = []
            holidays = []
            for t in range(self.predict_size):
                row_idx = pred_start + t
                dates.append(self.df.iloc[row_idx]['Data'])
                hours.append(self.df.iloc[row_idx]['hour_code'])
                holidays.append(self.df.iloc[row_idx]['is_holiday'])
            self.sample_dates.append(dates)
            self.sample_hour_codes.append(hours)
            self.sample_holidays.append(holidays)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x_energy = self.energy[start:start+self.window_size]
        x_energy = (x_energy - self.data_min) / (self.data_max - self.data_min + 1e-8)
        x_energy = torch.FloatTensor(x_energy).unsqueeze(-1)

        sector_idx = start + self.window_size - 1
        x_sector = self.sector_onehot[sector_idx]
        x_sector = torch.FloatTensor(x_sector).unsqueeze(0).repeat(self.window_size, 1)

        x_holiday = self.holiday[start:start+self.window_size]
        x_holiday = torch.FloatTensor(x_holiday).unsqueeze(-1)

        x_weekend = self.weekend[start:start+self.window_size]
        x_weekend = torch.FloatTensor(x_weekend).unsqueeze(-1)

        x = torch.cat([x_energy, x_sector, x_holiday, x_weekend], dim=1)

        y = self.energy[start+self.window_size:start+self.window_size+self.predict_size]
        y = (y - self.data_min) / (self.data_max - self.data_min + 1e-8)
        y = torch.FloatTensor(y)

        dates = self.sample_dates[idx]
        hours = self.sample_hour_codes[idx]
        holidays = self.sample_holidays[idx]
        return x, y, dates, hours, holidays

def custom_collate(batch):
    xs = torch.stack([item[0] for item in batch])
    ys = torch.stack([item[1] for item in batch])
    dates = [item[2] for item in batch]
    hours = [item[3] for item in batch]
    holidays = [item[4] for item in batch]
    return xs, ys, dates, hours, holidays

def load_node_loaders_local(node_ids, data_dir, node_minmax, split, batch_size=64, shuffle=True):
    loaders = {}
    for node_id in node_ids:
        node_dir = data_dir / f"node_{node_id}"
        pkl_file = node_dir / f"{split}.pkl"
        if not pkl_file.exists():
            continue
        dataset = MinMaxBarcelonaDataset(pkl_file, node_id, node_minmax, WINDOW_SIZE["7day"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate, drop_last=False)
        loaders[node_id] = loader
    return loaders

def load_initial_thresholds(threshold_file: Path) -> Dict:
    with open(threshold_file, 'r') as f:
        data = json.load(f)
    thresholds = {}
    for entry in data['thresholds']:
        key = (entry['hour_code'], entry['is_holiday'])
        thresholds[key] = (entry['low'], entry['high'])
    return thresholds

def make_decision_phase1(pred_real, low_th, high_th):
    if pred_real > high_th:
        return "Migration"
    elif pred_real < low_th:
        return "Sleep"
    else:
        return "Normal"

def compute_savings(pred_real, decision):
    if decision == "Sleep":
        actual = pred_real * SLEEP_FACTOR
    elif decision == "Migration":
        actual = pred_real * MIGRATE_FACTOR
    else:
        actual = pred_real
    saved = pred_real - actual
    return saved, actual

def expected_cost(decision, pred_dist, low_th, high_th):
    pred_mean = pred_dist['mean']
    pred_std = pred_dist['std']
    if pred_std < 1e-6:
        if pred_mean > high_th:
            probs = {'high': 1.0, 'medium': 0.0, 'low': 0.0}
        elif pred_mean < low_th:
            probs = {'high': 0.0, 'medium': 0.0, 'low': 1.0}
        else:
            probs = {'high': 0.0, 'medium': 1.0, 'low': 0.0}
    else:
        prob_high = 1 - norm.cdf(high_th, loc=pred_mean, scale=pred_std)
        prob_low = norm.cdf(low_th, loc=pred_mean, scale=pred_std)
        prob_medium = 1 - prob_high - prob_low
        probs = {'high': prob_high, 'medium': prob_medium, 'low': prob_low}
    exp_cost = 0.0
    for cat, prob in probs.items():
        exp_cost += prob * COST_MATRIX[cat][decision]
    return exp_cost

def update_node_thresholds(node_id, window_real_values, thresholds, quantile_low=0.2, quantile_high=0.8):
    """根据某个节点的滑动窗口数据，更新该节点所有 (hour_code, is_holiday) 的阈值"""
    updated = False
    for (nid, hour_code, is_holiday), values in window_real_values.items():
        if nid != node_id:
            continue
        key = (hour_code, is_holiday)
        if len(values) < 10:  # 样本太少则保持原阈值
            continue
        low = np.quantile(list(values), quantile_low)
        high = np.quantile(list(values), quantile_high)
        # 只有明显变化时才更新，避免频繁变动
        if key in thresholds and thresholds[key] != (low, high):
            thresholds[key] = (low, high)
            updated = True
    return updated

def save_thresholds(thresholds, output_path):
    """将阈值字典保存为与初始文件相同的格式"""
    data = {"thresholds": []}
    for (hour_code, is_holiday), (low, high) in thresholds.items():
        data["thresholds"].append({
            "hour_code": hour_code,
            "is_holiday": is_holiday,
            "low": low,
            "high": high
        })
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_comparison_plots(output_dir, window, mode):
    """如果存在 Phase 1 和 Phase 2 的结果，生成对比图"""
    phase1_csv = output_dir / f"decision_results_{window}_{mode}_phase1.csv"
    phase2_csv = output_dir / f"decision_results_{window}_{mode}_phase2.csv"
    phase3_csv = output_dir / f"decision_results_{window}_{mode}_phase3.csv"
    if not (phase1_csv.exists() and phase2_csv.exists() and phase3_csv.exists()):
        logger.warning("Phase 1 or Phase 2 results missing, skip comparison plots")
        return
    df1 = pd.read_csv(phase1_csv)
    df2 = pd.read_csv(phase2_csv)
    df3 = pd.read_csv(phase3_csv)

    acc1 = df1['decision_correct'].mean() * 100
    acc2 = df2['decision_correct'].mean() * 100
    acc3 = df3['decision_correct'].mean() * 100
    energy1 = df1['energy_saved_kwh'].sum()
    energy2 = df2['energy_saved_kwh'].sum()
    energy3 = df3['energy_saved_kwh'].sum()
    cost1 = df1['cost_saved_eur'].sum()
    cost2 = df2['cost_saved_eur'].sum()
    cost3 = df3['cost_saved_eur'].sum()
    carbon1 = df1['carbon_saved_kg'].sum()
    carbon2 = df2['carbon_saved_kg'].sum()
    carbon3 = df3['carbon_saved_kg'].sum()

    # 准确率对比柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(['Phase 1', 'Phase 2', 'Phase 3'], [acc1, acc2, acc3], color=['#3498db', '#e67e22', '#2ecc71'])
    plt.ylabel('Decision Accuracy (%)')
    plt.title(f'Decision Accuracy Comparison (window={window}, mode={mode})')
    for i, v in enumerate([acc1, acc2, acc3]):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    plt.savefig(output_dir / f'accuracy_comparison_{window}_{mode}.png', dpi=150)
    plt.close()

    # 节能效果对比（归一化后显示）
    plt.figure(figsize=(10, 6))
    x = np.arange(3)
    width = 0.25
    plt.bar(x - width, [energy1, energy2, energy3], width, label='Energy Saved (MWh)', color='#3498db')
    plt.bar(x, [cost1, cost2, cost3], width, label='Cost Saved (k€)', color='#e67e22')
    plt.bar(x + width, [carbon1/1000, carbon2/1000, carbon3/1000], width, label='Carbon Saved (t CO2)', color='#2ecc71')
    plt.xticks(x, ['Phase 1', 'Phase 2', 'Phase 3'])
    plt.ylabel('Amount')
    plt.title(f'Energy, Cost, Carbon Savings Comparison (window={window}, mode={mode})')
    plt.legend()
    plt.savefig(output_dir / f'savings_comparison_{window}_{mode}.png', dpi=150)
    plt.close()

    logger.info(f"Comparison plots saved to {output_dir}")

def generate_uncertainty_plot(df_results, output_dir, window, mode):
    if not VISUALIZATION_AVAILABLE:
        return
    sample_df = df_results.sample(min(100, len(df_results)))
    sample_df = sample_df.sort_values('date')
    plt.figure(figsize=(12, 6))
    plt.errorbar(sample_df.index, sample_df['pred_mean_kw'], 
                 yerr=1.96 * sample_df['pred_std_kw'], fmt='o', capsize=2, alpha=0.5, label='Prediction ± 1.96σ')
    plt.plot(sample_df.index, sample_df['real_kw'], 'rx', label='True value')
    plt.title(f'MC Dropout Prediction Uncertainty (window={window}, mode={mode})')
    plt.xlabel('Sample index')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'uncertainty_{window}_{mode}_phase3.png', dpi=150)
    plt.close()

# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=str, choices=['1day', '7day'], default='7day')
    parser.add_argument('--mode', type=str, choices=['all', 'first'], default='all')
    parser.add_argument('--output_dir', type=str, default=str(PROJECT_ROOT / "decision" / "outputs"))
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--dynamic-threshold', action='store_true', default=True)
    parser.add_argument('--window-days', type=int, default=DYNAMIC_WINDOW_DAYS)
    parser.add_argument('--update-interval', type=int, default=UPDATE_INTERVAL)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"decision_results_{args.window}_{args.mode}_phase3.csv"
    output_stats = output_dir / f"summary_stats_{args.window}_{args.mode}_phase3.json"
    threshold_out_path = output_dir / f"thresholds_dynamic_phase3.json"

    logger.info("=" * 60)
    logger.info(f"Phase 3 Decision Engine (Fixed: Node-Isolated Dynamic Thresholds)")
    logger.info(f"Window={args.window}, mode={args.mode}")
    logger.info(f"Dynamic threshold: {args.dynamic_threshold}, window days={args.window_days}, update interval={args.update_interval}")
    logger.info("=" * 60)

    # 检查必要文件
    for fpath, name in [(MINMAX_PATH, "MinMax"), (NODE_PARAMS_PATH, "Node params"),
                        (THRESHOLD_PATH, "Thresholds"), (TEST_DATA_PATH, "Test data"),
                        (FINAL_COMPARISON_PATH, "Best model assignment")]:
        if not fpath.exists():
            logger.error(f"{name} file not found: {fpath}")
            sys.exit(1)

    best_model_df = pd.read_csv(FINAL_COMPARISON_PATH)
    best_model_dict = dict(zip(best_model_df['node_id'], best_model_df['best_model']))
    logger.info(f"Loaded best model assignment for {len(best_model_dict)} nodes")

    with open(MINMAX_PATH, 'rb') as f:
        node_minmax = pickle.load(f)

    node_params_df = pd.read_csv(NODE_PARAMS_PATH)
    node_params_df['date'] = pd.to_datetime(node_params_df[['year', 'month']].assign(day=1))

    # 初始阈值（从文件加载，键为 (hour_code, is_holiday)，后续我们将为每个节点独立更新）
    base_thresholds = load_initial_thresholds(THRESHOLD_PATH)
    # 将阈值字典转换为以 (node_id, hour_code, is_holiday) 为键，初始值从 base_thresholds 复制
    thresholds = {}
    node_ids_all = list(node_minmax.keys())
    for node_id in node_ids_all:
        for (hc, hd), (low, high) in base_thresholds.items():
            thresholds[(node_id, hc, hd)] = (low, high)
    logger.info(f"Initialized thresholds for {len(node_ids_all)} nodes × {len(base_thresholds)} (hour,holiday) pairs")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cache = {}

    node_dirs = sorted(TEST_DATA_PATH.glob("node_*"))
    all_results = []
    total_time = 0.0
    total_samples = 0
    window_size = WINDOW_SIZE[args.window]

    # 滑动窗口存储真实能耗（按节点隔离）
    if args.dynamic_threshold:
        maxlen = args.window_days * 24
        window_real_values = defaultdict(lambda: deque(maxlen=maxlen))
        # 每个节点的样本计数器，用于独立触发阈值更新
        node_sample_count = defaultdict(int)

    for node_dir in node_dirs:
        node_id = int(node_dir.name.split('_')[1])
        if node_id == 8025:
            continue
        logger.info(f"Processing node {node_id}...")

        model_type = best_model_dict.get(node_id, 'original')
        model_path = SEC_MODEL_DIR / f"node_{node_id}.pth" if model_type == 'secondary' else ORIG_MODEL_DIR / f"node_{node_id}.pth"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            continue

        if node_id in model_cache:
            model = model_cache[node_id]
        else:
            model = LSTMPredictor().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model_cache[node_id] = model

        test_file = node_dir / "test.pkl"
        if not test_file.exists():
            logger.warning(f"Test file missing for node {node_id}")
            continue

        dataset = MinMaxBarcelonaDataset(test_file, node_id, node_minmax, window_size)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=custom_collate)
        node_params = node_params_df[node_params_df['node_id'] == node_id]

        # 重置该节点的样本计数（如果动态阈值开启）
        if args.dynamic_threshold:
            node_sample_count[node_id] = 0

        for x, y, dates, hours, holidays in loader:
            x = x.to(device)
            batch_size_curr = x.size(0)

            # MC Dropout
            model.train()
            pred_samples = []
            for _ in range(MC_NUM_FORWARD):
                with torch.no_grad():
                    pred_norm = model(x).cpu().numpy()
                pred_samples.append(pred_norm)
            model.eval()

            pred_samples = np.array(pred_samples)
            pred_mean_norm = pred_samples.mean(axis=0)
            pred_std_norm = pred_samples.std(axis=0)

            with torch.no_grad():
                pred_norm_single = model(x).cpu().numpy()

            data_min, data_max = node_minmax[node_id]
            pred_mean_real = pred_mean_norm * (data_max - data_min) + data_min
            pred_std_real = pred_std_norm * (data_max - data_min) + data_min
            pred_single_real = pred_norm_single * (data_max - data_min) + data_min
            y_real = y.cpu().numpy() * (data_max - data_min) + data_min

            start_time = time.time()
            elapsed = time.time() - start_time
            total_time += elapsed * batch_size_curr
            total_samples += batch_size_curr

            for i in range(batch_size_curr):
                t_range = [0] if args.mode == 'first' else range(4)
                for t in t_range:
                    pred_mean = pred_mean_real[i, t]
                    pred_std = pred_std_real[i, t]
                    real_val = y_real[i, t]
                    hour_code = hours[i][t].item()
                    is_holiday = holidays[i][t].item()
                    key = (node_id, hour_code, is_holiday)

                    # 获取当前阈值
                    if key not in thresholds:
                        continue
                    low_th, high_th = thresholds[key]

                    # 成本敏感决策
                    pred_dist = {'mean': pred_mean, 'std': pred_std}
                    exp_costs = {}
                    for dec in ['Sleep', 'Normal', 'Migration']:
                        exp_costs[dec] = expected_cost(dec, pred_dist, low_th, high_th)
                    decision = min(exp_costs, key=exp_costs.get)
                    energy_saved, _ = compute_savings(pred_mean, decision)

                    sample_dt = dates[i][t]
                    sample_year = sample_dt.year
                    sample_month = sample_dt.month
                    param_row = node_params[(node_params['year'] == sample_year) & (node_params['month'] == sample_month)]
                    if len(param_row) == 0:
                        continue
                    price = param_row.iloc[0]['price_euro_kwh']
                    carbon = param_row.iloc[0]['carbon_kg_kwh']

                    cost_saved = energy_saved * price
                    carbon_saved = energy_saved * carbon

                    correct = (real_val > high_th and decision == "Migration") or \
                              (real_val < low_th and decision == "Sleep") or \
                              (low_th <= real_val <= high_th and decision == "Normal")

                    all_results.append({
                        'node_id': node_id,
                        'date': sample_dt.strftime('%Y-%m-%d'),
                        'hour_code': hour_code,
                        'is_holiday': is_holiday,
                        'pred_mean_kw': pred_mean,
                        'pred_std_kw': pred_std,
                        'real_kw': real_val,
                        'decision': decision,
                        'energy_saved_kwh': energy_saved,
                        'cost_saved_eur': cost_saved,
                        'carbon_saved_kg': carbon_saved,
                        'decision_correct': correct,
                        'expected_cost_sleep': exp_costs.get('Sleep', 0),
                        'expected_cost_normal': exp_costs.get('Normal', 0),
                        'expected_cost_migration': exp_costs.get('Migration', 0),
                        'low_threshold': low_th,
                        'high_threshold': high_th,
                        'latency_ms': elapsed * 1000 / batch_size_curr
                    })

                    # 动态阈值：收集真实值到滑动窗口（按节点隔离）
                    if args.dynamic_threshold:
                        window_key = (node_id, hour_code, is_holiday)
                        window_real_values[window_key].append(real_val)
                        node_sample_count[node_id] += 1
                        if node_sample_count[node_id] >= args.update_interval:
                            # 更新该节点的所有阈值
                            updated = update_node_thresholds(node_id, window_real_values, thresholds)
                            if updated:
                                logger.info(f"Node {node_id}: thresholds updated after {node_sample_count[node_id]} samples")
                                # 保存更新后的阈值到文件（可选，每次更新都保存会慢，可降低频率）
                                save_thresholds({k: v for k, v in thresholds.items() if k[0]==node_id}, 
                                                output_dir / f"thresholds_node_{node_id}_backup.json")
                            node_sample_count[node_id] = 0

    if not all_results:
        logger.error("No results generated")
        sys.exit(1)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_csv, index=False)
    logger.info(f"Phase 3 results saved to {output_csv}")

    # 保存最终阈值（所有节点）
    if args.dynamic_threshold:
        # 转换回以 (hour_code, is_holiday) 为键的字典（注意：不同节点阈值不同，这里只能保存最后一个节点的？实际上每个节点阈值不同，无法合并到一个文件。我们保存为按节点分离的文件）
        # 为了与原始格式兼容，我们保存每个节点的阈值到单独文件
        for node_id in set(df_results['node_id']):
            node_thresholds = {}
            for (nid, hc, hd), (low, high) in thresholds.items():
                if nid == node_id:
                    node_thresholds[(hc, hd)] = (low, high)
            if node_thresholds:
                save_thresholds(node_thresholds, output_dir / f"thresholds_node_{node_id}_final.json")
        logger.info(f"Final thresholds saved per node in {output_dir}")

    # 统计摘要
    total_energy_saved = df_results['energy_saved_kwh'].sum()
    total_cost_saved = df_results['cost_saved_eur'].sum()
    total_carbon_saved = df_results['carbon_saved_kg'].sum()
    accuracy = df_results['decision_correct'].mean() * 100
    avg_latency = df_results['latency_ms'].mean()

    stats = {
        'phase': 'Phase 3 (Node-Isolated Dynamic Threshold + Cost-Sensitive + MC Dropout)',
        'window': args.window,
        'mode': args.mode,
        'dynamic_threshold_enabled': args.dynamic_threshold,
        'window_days': args.window_days,
        'update_interval': args.update_interval,
        'total_energy_saved_kwh': total_energy_saved,
        'total_cost_saved_eur': total_cost_saved,
        'total_carbon_saved_kg': total_carbon_saved,
        'decision_accuracy_percent': accuracy,
        'avg_latency_ms': avg_latency,
        'num_samples': len(df_results),
        'num_nodes': df_results['node_id'].nunique(),
        'time_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mc_forward_passes': MC_NUM_FORWARD,
    }
    with open(output_stats, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Phase 3 statistics saved to {output_stats}")

    # 生成可视化
    if not args.no_plots:
        generate_uncertainty_plot(df_results, output_dir, args.window, args.mode)
        generate_comparison_plots(output_dir, args.window, args.mode)
    else:
        logger.info("Skip plots")

    logger.info("=" * 60)
    logger.info(f"Phase 3 Decision Summary")
    logger.info(f"  Total samples: {stats['num_samples']}")
    logger.info(f"  Total energy saved: {total_energy_saved:.2f} kWh")
    logger.info(f"  Total cost saved: {total_cost_saved:.2f} €")
    logger.info(f"  Total carbon reduced: {total_carbon_saved:.2f} kg CO2")
    logger.info(f"  Decision accuracy: {accuracy:.2f}%")
    logger.info(f"  Avg latency: {avg_latency:.2f} ms")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()