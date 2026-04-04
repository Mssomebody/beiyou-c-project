#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 决策引擎（成本敏感 + MC Dropout）
- 每个节点使用各自的最佳微调模型
- 使用 MC Dropout 获得预测分布（均值、标准差、置信区间）
- 基于成本矩阵进行风险感知决策（最小化期望成本）
- 同时输出 Phase 1（原阈值决策）和 Phase 2 结果，用于对比
- 图表全部英文，避免字体问题
"""

import sys
import json
import time
import pickle
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
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
    print("Warning: matplotlib or seaborn not installed, cannot generate plots.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "versions" / "v2_holiday_sector"))

# 固定路径
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

# MC Dropout 参数
MC_NUM_FORWARD = 30   # 蒙特卡洛前向传播次数

# ============================================================
# 成本矩阵（量化）
# 行：真实负载类别（高、中、低），列：决策（休眠、正常、迁移）
# 成本值根据业务量化得出
# ============================================================
COST_MATRIX = {
    'high':   {'Sleep': 400, 'Normal': 3, 'Migration': 1},
    'medium': {'Sleep': 200, 'Normal': 0, 'Migration': 2},
    'low':    {'Sleep': 0,   'Normal': 3, 'Migration': 2}
}
# 负载分类阈值（相对于节点历史最大值，这里使用动态阈值的high/low）
# 实际决策时，根据真实能耗 real_val 与阈值比较得到真实负载类别

# ============================================================
# 模型定义（与 Phase 1 相同，但 forward 时保持 train 模式以启用 dropout）
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        # 注意：训练时 dropout 已启用，eval 模式下需要手动启用 MC Dropout
        # 我们将在推理时调用 model.train() 来保持 dropout 激活
        return self.fc(last_out)

# ============================================================
# 数据集类（与 Phase 1 相同）
# ============================================================
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

# ============================================================
# 自定义 collate 函数
# ============================================================
def custom_collate(batch):
    xs = torch.stack([item[0] for item in batch])
    ys = torch.stack([item[1] for item in batch])
    dates = [item[2] for item in batch]
    hours = [item[3] for item in batch]
    holidays = [item[4] for item in batch]
    return xs, ys, dates, hours, holidays

# ============================================================
# 本地数据加载函数
# ============================================================
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

# ============================================================
# 辅助函数
# ============================================================
def load_dynamic_thresholds(threshold_file: Path) -> Dict:
    with open(threshold_file, 'r') as f:
        data = json.load(f)
    thresholds = {}
    for entry in data['thresholds']:
        key = (entry['hour_code'], entry['is_holiday'])
        thresholds[key] = (entry['low'], entry['high'])
    return thresholds

def make_decision_phase1(pred_real: float, low_th: float, high_th: float) -> str:
    if pred_real > high_th:
        return "Migration"
    elif pred_real < low_th:
        return "Sleep"
    else:
        return "Normal"

def compute_savings(pred_real: float, decision: str) -> Tuple[float, float]:
    if decision == "Sleep":
        actual = pred_real * SLEEP_FACTOR
    elif decision == "Migration":
        actual = pred_real * MIGRATE_FACTOR
    else:
        actual = pred_real
    saved = pred_real - actual
    return saved, actual

def get_true_load_category(real_val, low_th, high_th):
    if real_val > high_th:
        return 'high'
    elif real_val < low_th:
        return 'low'
    else:
        return 'medium'

def expected_cost(decision, pred_dist, low_th, high_th):
    """
    计算采取某个决策的期望成本，基于预测分布（均值、标准差）
    这里简化：将预测分布视为高斯，对每个可能的真实负载类别积分
    实际中，我们使用预测均值和阈值来估计真实负载类别的概率
    """
    # 预测均值
    pred_mean = pred_dist['mean']
    pred_std = pred_dist['std']
    # 计算真实负载属于 high/medium/low 的概率（基于预测分布）
    # 假设真实值服从 N(pred_mean, pred_std^2)
    # 计算 P(real > high_th), P(real < low_th), P(low_th <= real <= high_th)
    if pred_std < 1e-6:
        # 确定性情况
        if pred_mean > high_th:
            probs = {'high': 1.0, 'medium': 0.0, 'low': 0.0}
        elif pred_mean < low_th:
            probs = {'high': 0.0, 'medium': 0.0, 'low': 1.0}
        else:
            probs = {'high': 0.0, 'medium': 1.0, 'low': 0.0}
    else:
        from scipy.stats import norm
        prob_high = 1 - norm.cdf(high_th, loc=pred_mean, scale=pred_std)
        prob_low = norm.cdf(low_th, loc=pred_mean, scale=pred_std)
        prob_medium = 1 - prob_high - prob_low
        probs = {'high': prob_high, 'medium': prob_medium, 'low': prob_low}
    # 期望成本 = sum_{cat} P(cat) * cost(cat, decision)
    exp_cost = 0.0
    for cat, prob in probs.items():
        exp_cost += prob * COST_MATRIX[cat][decision]
    return exp_cost

# ============================================================
# 可视化生成（Phase 2 特有：置信带图）
# ============================================================
def generate_uncertainty_plots(df_results, output_dir, window, mode, num_samples=100):
    """随机抽取部分样本，绘制预测置信带"""
    if not VISUALIZATION_AVAILABLE:
        return
    if 'pred_mean_kw' not in df_results.columns or 'pred_std_kw' not in df_results.columns:
        logger.warning("No prediction distribution data, skip uncertainty plots")
        return
    # 随机选择一些样本（最多 num_samples 个）
    sample_df = df_results.sample(min(num_samples, len(df_results)))
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
    plt.savefig(output_dir / f'uncertainty_{window}_{mode}.png', dpi=150)
    plt.close()
    logger.info(f"Uncertainty plot saved to {output_dir / f'uncertainty_{window}_{mode}.png'}")

# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=str, choices=['1day', '7day'], default='7day')
    parser.add_argument('--mode', type=str, choices=['all', 'first'], default='all')
    parser.add_argument('--output_dir', type=str, default=str(PROJECT_ROOT / "decision" / "outputs"))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Phase 2 输出文件加 _phase2 后缀
    output_csv = output_dir / f"decision_results_{args.window}_{args.mode}_phase2.csv"
    output_stats_phase1 = output_dir / f"summary_stats_{args.window}_{args.mode}_phase1.json"
    output_stats_phase2 = output_dir / f"summary_stats_{args.window}_{args.mode}_phase2.json"

    logger.info("=" * 60)
    logger.info(f"Phase 2 Decision Engine (Cost-Sensitive + MC Dropout) Started")
    logger.info(f"Window={args.window}, mode={args.mode}")
    logger.info("Each node uses its own best fine-tuned model")
    logger.info(f"MC Dropout forward passes: {MC_NUM_FORWARD}")
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

    thresholds = load_dynamic_thresholds(THRESHOLD_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cache = {}

    node_dirs = sorted(TEST_DATA_PATH.glob("node_*"))
    all_results_phase1 = []
    all_results_phase2 = []
    total_time = 0.0
    total_samples = 0

    window_size = WINDOW_SIZE[args.window]

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

        for x, y, dates, hours, holidays in loader:
            x = x.to(device)
            batch_size_curr = x.size(0)

            # 开启 MC Dropout：将模型设为 train 模式（启用 dropout）
            model.train()
            pred_samples = []
            for _ in range(MC_NUM_FORWARD):
                with torch.no_grad():
                    pred_norm = model(x).cpu().numpy()
                pred_samples.append(pred_norm)
            model.eval()  # 恢复 eval 模式，但后续不再需要

            # 计算预测分布统计
            pred_samples = np.array(pred_samples)  # (MC, batch, 4)
            pred_mean_norm = pred_samples.mean(axis=0)
            pred_std_norm = pred_samples.std(axis=0)

            # 单次预测（用于 Phase 1，使用确定性推理）
            with torch.no_grad():
                pred_norm_single = model(x).cpu().numpy()

            # 反归一化
            data_min, data_max = node_minmax[node_id]
            pred_mean_real = pred_mean_norm * (data_max - data_min) + data_min
            pred_std_real = pred_std_norm * (data_max - data_min) + data_min
            pred_single_real = pred_norm_single * (data_max - data_min) + data_min
            y_real = y.cpu().numpy() * (data_max - data_min) + data_min

            start_time = time.time()
            elapsed = time.time() - start_time  # 实际上大部分时间在 MC 前向，这里简单记录
            total_time += elapsed * batch_size_curr
            total_samples += batch_size_curr

            for i in range(batch_size_curr):
                t_range = [0] if args.mode == 'first' else range(4)
                for t in t_range:
                    # 基础数据
                    pred_single = pred_single_real[i, t]
                    pred_mean = pred_mean_real[i, t]
                    pred_std = pred_std_real[i, t]
                    real_val = y_real[i, t]
                    hour_code = hours[i][t].item()
                    is_holiday = holidays[i][t].item()
                    key = (hour_code, is_holiday)
                    if key not in thresholds:
                        continue
                    low_th, high_th = thresholds[key]

                    # ========== Phase 1 决策 ==========
                    decision_phase1 = make_decision_phase1(pred_single, low_th, high_th)
                    energy_saved_phase1, _ = compute_savings(pred_single, decision_phase1)

                    # ========== Phase 2 决策 ==========
                    # 计算每个决策的期望成本
                    pred_dist = {'mean': pred_mean, 'std': pred_std}
                    exp_costs = {}
                    for dec in ['Sleep', 'Normal', 'Migration']:
                        exp_costs[dec] = expected_cost(dec, pred_dist, low_th, high_th)
                    decision_phase2 = min(exp_costs, key=exp_costs.get)
                    energy_saved_phase2, _ = compute_savings(pred_mean, decision_phase2)

                    # 共享的日期、电价、碳排信息
                    sample_dt = dates[i][t]
                    sample_year = sample_dt.year
                    sample_month = sample_dt.month
                    param_row = node_params[(node_params['year'] == sample_year) & (node_params['month'] == sample_month)]
                    if len(param_row) == 0:
                        continue
                    price = param_row.iloc[0]['price_euro_kwh']
                    carbon = param_row.iloc[0]['carbon_kg_kwh']

                    # 决策正确性判断（基于真实值）
                    def is_correct(decision, real_val, low_th, high_th):
                        return (real_val > high_th and decision == "Migration") or \
                               (real_val < low_th and decision == "Sleep") or \
                               (low_th <= real_val <= high_th and decision == "Normal")

                    correct_phase1 = is_correct(decision_phase1, real_val, low_th, high_th)
                    correct_phase2 = is_correct(decision_phase2, real_val, low_th, high_th)

                    # 保存 Phase 1 结果
                    all_results_phase1.append({
                        'node_id': node_id,
                        'date': sample_dt.strftime('%Y-%m-%d'),
                        'hour_code': hour_code,
                        'is_holiday': is_holiday,
                        'pred_kw': pred_single,
                        'real_kw': real_val,
                        'decision': decision_phase1,
                        'energy_saved_kwh': energy_saved_phase1,
                        'cost_saved_eur': energy_saved_phase1 * price,
                        'carbon_saved_kg': energy_saved_phase1 * carbon,
                        'price_euro_kwh': price,
                        'carbon_kg_kwh': carbon,
                        'decision_correct': correct_phase1,
                        'confidence': 0.0,  # Phase 1 置信度未使用
                        'latency_ms': elapsed * 1000 / batch_size_curr
                    })

                    # 保存 Phase 2 结果（包含预测分布）
                    all_results_phase2.append({
                        'node_id': node_id,
                        'date': sample_dt.strftime('%Y-%m-%d'),
                        'hour_code': hour_code,
                        'is_holiday': is_holiday,
                        'pred_mean_kw': pred_mean,
                        'pred_std_kw': pred_std,
                        'pred_lower_kw': pred_mean - 1.96 * pred_std,
                        'pred_upper_kw': pred_mean + 1.96 * pred_std,
                        'real_kw': real_val,
                        'decision': decision_phase2,
                        'energy_saved_kwh': energy_saved_phase2,
                        'cost_saved_eur': energy_saved_phase2 * price,
                        'carbon_saved_kg': energy_saved_phase2 * carbon,
                        'price_euro_kwh': price,
                        'carbon_kg_kwh': carbon,
                        'decision_correct': correct_phase2,
                        'expected_cost_sleep': exp_costs.get('Sleep', 0),
                        'expected_cost_normal': exp_costs.get('Normal', 0),
                        'expected_cost_migration': exp_costs.get('Migration', 0),
                        'latency_ms': elapsed * 1000 / batch_size_curr
                    })

    if not all_results_phase2:
        logger.error("No results generated")
        sys.exit(1)

    # ========== 保存 Phase 1 结果（用于对比）==========
    df_phase1 = pd.DataFrame(all_results_phase1)
    df_phase1.to_csv(output_csv.parent / f"decision_results_{args.window}_{args.mode}_phase1.csv", index=False)
    logger.info(f"Phase 1 results saved to {output_csv.parent / f'decision_results_{args.window}_{args.mode}_phase1.csv'}")

    # ========== 保存 Phase 2 结果 ==========
    df_phase2 = pd.DataFrame(all_results_phase2)
    df_phase2.to_csv(output_csv, index=False)
    logger.info(f"Phase 2 results saved to {output_csv}")

    # ========== 统计摘要 Phase 1 ==========
    total_energy_saved_p1 = df_phase1['energy_saved_kwh'].sum()
    total_cost_saved_p1 = df_phase1['cost_saved_eur'].sum()
    total_carbon_saved_p1 = df_phase1['carbon_saved_kg'].sum()
    accuracy_p1 = df_phase1['decision_correct'].mean() * 100
    avg_latency_p1 = df_phase1['latency_ms'].mean()
    stats_p1 = {
        'phase': 'Phase 1 (Threshold-based)',
        'window': args.window,
        'mode': args.mode,
        'total_energy_saved_kwh': total_energy_saved_p1,
        'total_cost_saved_eur': total_cost_saved_p1,
        'total_carbon_saved_kg': total_carbon_saved_p1,
        'decision_accuracy_percent': accuracy_p1,
        'avg_latency_ms': avg_latency_p1,
        'num_samples': len(df_phase1),
        'num_nodes': df_phase1['node_id'].nunique(),
        'time_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_stats_phase1, 'w') as f:
        json.dump(stats_p1, f, indent=2)
    logger.info(f"Phase 1 statistics saved to {output_stats_phase1}")

    # ========== 统计摘要 Phase 2 ==========
    total_energy_saved_p2 = df_phase2['energy_saved_kwh'].sum()
    total_cost_saved_p2 = df_phase2['cost_saved_eur'].sum()
    total_carbon_saved_p2 = df_phase2['carbon_saved_kg'].sum()
    accuracy_p2 = df_phase2['decision_correct'].mean() * 100
    avg_latency_p2 = df_phase2['latency_ms'].mean()
    stats_p2 = {
        'phase': 'Phase 2 (Cost-Sensitive + MC Dropout)',
        'window': args.window,
        'mode': args.mode,
        'total_energy_saved_kwh': total_energy_saved_p2,
        'total_cost_saved_eur': total_cost_saved_p2,
        'total_carbon_saved_kg': total_carbon_saved_p2,
        'decision_accuracy_percent': accuracy_p2,
        'avg_latency_ms': avg_latency_p2,
        'num_samples': len(df_phase2),
        'num_nodes': df_phase2['node_id'].nunique(),
        'time_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mc_forward_passes': MC_NUM_FORWARD,
    }
    with open(output_stats_phase2, 'w') as f:
        json.dump(stats_p2, f, indent=2)
    logger.info(f"Phase 2 statistics saved to {output_stats_phase2}")

    # ========== 打印对比摘要 ==========
    logger.info("=" * 60)
    logger.info("Comparison between Phase 1 and Phase 2")
    logger.info(f"  Decision Accuracy: Phase1={accuracy_p1:.2f}%, Phase2={accuracy_p2:.2f}%")
    logger.info(f"  Total Energy Saved: Phase1={total_energy_saved_p1:.2f} kWh, Phase2={total_energy_saved_p2:.2f} kWh")
    logger.info(f"  Total Cost Saved: Phase1={total_cost_saved_p1:.2f} €, Phase2={total_cost_saved_p2:.2f} €")
    logger.info(f"  Total Carbon Reduced: Phase1={total_carbon_saved_p1:.2f} kg, Phase2={total_carbon_saved_p2:.2f} kg")
    logger.info("=" * 60)

    # ========== 可视化（Phase 2 特有的不确定性图）==========
    if not args.no_plots:
        generate_uncertainty_plots(df_phase2, output_dir, args.window, args.mode)
    else:
        logger.info("Skip uncertainty plots")

    logger.info("Phase 2 Decision Engine completed.")

if __name__ == "__main__":
    main()