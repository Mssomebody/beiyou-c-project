#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 决策引擎（统一版）- 支持一天/七天窗口，支持输出模式选择，含可视化

功能：
- 加载微调后的模型（一天窗口或七天窗口）
- 对新口径测试集进行预测，反归一化
- 基于动态阈值（按时段/节假日）做出决策（休眠/迁移/正常）
- 节能效果量化（能耗、成本、碳排）
- 决策准确率、时效性统计
- **自动生成节能曲线、成本柱状图、碳减排图、准确率饼图等可视化图表**
- 输出 CSV 和 JSON 摘要，文件名区分窗口和模式

使用示例：
    # 七天窗口，预测全部四个时段，生成图表
    python run_decision.py --window 7day --mode all
    # 一天窗口，只预测第一个时段（未来6小时），不生成图表
    python run_decision.py --window 1day --mode first --no-plots

作者: FedGreen-C 项目组
版本: 3.1 (完整版)
日期: 2026-03-31
"""

import sys
import os
import json
import time
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 可视化库（如果缺失则给出警告，但不中断）
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("警告: matplotlib 或 seaborn 未安装，无法生成图表。请运行 'pip install matplotlib seaborn' 安装。")

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 固定路径
MINMAX_PATH = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
NODE_PARAMS_PATH = PROJECT_ROOT / "decision" / "config" / "node_weighted_params_monthly.csv"
THRESHOLD_PATH = PROJECT_ROOT / "decision" / "config" / "thresholds_dynamic.json"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_2023_2025"

# 模型路径
MODEL_7DAY = PROJECT_ROOT / "decision" / "models" / "model_fed_finetune.pth"
MODEL_1DAY = PROJECT_ROOT / "decision" / "models" / "model_fed_finetune_1day.pth"

# 节能系数
SLEEP_FACTOR = 0.05
MIGRATE_FACTOR = 0.8

# 模型参数
INPUT_DIM = 7
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 4
DROPOUT = 0.2

# 窗口参数
WINDOW_SIZE = {
    "1day": 4,   # 1天 = 4个时段
    "7day": 28   # 7天 = 28个时段
}
PREDICT_SIZE = 4  # 输出未来4个时段

# ============================================================
# 模型定义（与训练一致）
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 数据集类（支持不同窗口大小）
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
        indices = []
        for i in range(total - self.window_size - self.predict_size + 1):
            indices.append(i)
        return indices

    def _extract_sample_metadata(self):
        """为每个样本提取每个预测时段的日期、hour_code、is_holiday"""
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

        # 输入特征
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

        # 目标
        y = self.energy[start+self.window_size:start+self.window_size+self.predict_size]
        y = (y - self.data_min) / (self.data_max - self.data_min + 1e-8)
        y = torch.FloatTensor(y)

        # 附加元数据
        dates = self.sample_dates[idx]
        hours = self.sample_hour_codes[idx]
        holidays = self.sample_holidays[idx]
        return x, y, dates, hours, holidays


# ============================================================
# 辅助函数
# ============================================================
def load_dynamic_thresholds(threshold_file: Path) -> Dict:
    """加载动态阈值 {(hour_code, is_holiday): (low, high)}"""
    with open(threshold_file, 'r') as f:
        data = json.load(f)
    thresholds = {}
    for entry in data['thresholds']:
        key = (entry['hour_code'], entry['is_holiday'])
        thresholds[key] = (entry['low'], entry['high'])
    return thresholds


def make_decision(pred_real: float, low_th: float, high_th: float) -> str:
    if pred_real > high_th:
        return "迁移"
    elif pred_real < low_th:
        return "休眠"
    else:
        return "正常"


def compute_savings(pred_real: float, decision: str) -> Tuple[float, float]:
    """返回 (energy_saved, actual_energy)"""
    if decision == "休眠":
        actual = pred_real * SLEEP_FACTOR
    elif decision == "迁移":
        actual = pred_real * MIGRATE_FACTOR
    else:
        actual = pred_real
    saved = pred_real - actual
    return saved, actual


# ============================================================
# 可视化生成（与原版一致）
# ============================================================
def generate_visualizations(df_results, stats, output_dir: Path, window: str, mode: str):
    """生成决策效果仿真图表"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("跳过可视化生成：matplotlib/seaborn 未安装")
        return
    sns.set_style("whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 累计节省能耗曲线
    df_sorted = df_results.sort_values('date')
    df_sorted['cumulative_energy'] = df_sorted['energy_saved_kwh'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted['date'], df_sorted['cumulative_energy'], marker='o', linestyle='-', markersize=2)
    plt.title(f'累计节省能耗 (窗口={window}, 模式={mode})')
    plt.xlabel('日期')
    plt.ylabel('累计节省能耗 (kWh)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f'cumulative_energy_{window}_{mode}.png', dpi=150)
    plt.close()

    # 2. 各节点节省成本柱状图
    node_cost = df_results.groupby('node_id')['cost_saved_eur'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    node_cost.plot(kind='bar')
    plt.title(f'各节点节省成本 (窗口={window}, 模式={mode})')
    plt.xlabel('节点 ID')
    plt.ylabel('节省成本 (€)')
    plt.tight_layout()
    plt.savefig(output_dir / f'cost_saved_by_node_{window}_{mode}.png', dpi=150)
    plt.close()

    # 3. 各节点碳减排柱状图
    node_carbon = df_results.groupby('node_id')['carbon_saved_kg'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    node_carbon.plot(kind='bar', color='green')
    plt.title(f'各节点碳减排 (窗口={window}, 模式={mode})')
    plt.xlabel('节点 ID')
    plt.ylabel('碳减排 (kg CO₂)')
    plt.tight_layout()
    plt.savefig(output_dir / f'carbon_saved_by_node_{window}_{mode}.png', dpi=150)
    plt.close()

    # 4. 决策准确率饼图
    correct = stats['decision_accuracy_percent']
    incorrect = 100 - correct
    plt.figure(figsize=(6, 6))
    plt.pie([correct, incorrect], labels=['正确', '错误'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    plt.title(f'决策准确率 (窗口={window}, 模式={mode})')
    plt.savefig(output_dir / f'decision_accuracy_{window}_{mode}.png', dpi=150)
    plt.close()

    # 5. 决策类型分布
    decision_counts = df_results['decision'].value_counts()
    plt.figure(figsize=(8, 5))
    decision_counts.plot(kind='bar', color=['#3498db', '#e67e22', '#2ecc71'])
    plt.title(f'决策类型分布 (窗口={window}, 模式={mode})')
    plt.xlabel('决策')
    plt.ylabel('次数')
    plt.tight_layout()
    plt.savefig(output_dir / f'decision_distribution_{window}_{mode}.png', dpi=150)
    plt.close()

    # 6. 按时段的平均节省能耗（仅当 mode='all' 且包含 hour_code）
    if mode == 'all' and 'hour_code' in df_results.columns:
        hour_savings = df_results.groupby('hour_code')['energy_saved_kwh'].mean()
        plt.figure(figsize=(8, 5))
        hour_savings.plot(kind='bar')
        plt.title(f'各时段平均节省能耗 (窗口={window}, 模式={mode})')
        plt.xlabel('时段')
        plt.ylabel('平均节省能耗 (kWh)')
        plt.tight_layout()
        plt.savefig(output_dir / f'savings_by_hour_{window}_{mode}.png', dpi=150)
        plt.close()

    logger.info(f"可视化图表已保存至 {output_dir}")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 1 决策引擎（统一版，含可视化）")
    parser.add_argument('--window', type=str, choices=['1day', '7day'], default='7day',
                        help='窗口长度：1day（输入4步）或 7day（输入28步）')
    parser.add_argument('--mode', type=str, choices=['all', 'first'], default='all',
                        help='输出模式：all（四个时段全部决策），first（仅第一个时段，用于实时决策）')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径（覆盖默认）')
    parser.add_argument('--output_dir', type=str, default=str(PROJECT_ROOT / "decision" / "outputs"),
                        help='输出目录')
    parser.add_argument('--no-plots', action='store_true', help='不生成可视化图表')
    args = parser.parse_args()

    # 确定模型路径
    if args.model:
        model_path = Path(args.model)
    else:
        if args.window == '7day':
            model_path = MODEL_7DAY
        else:
            model_path = MODEL_1DAY

    # 输出文件名
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"decision_results_{args.window}_{args.mode}.csv"
    output_stats = output_dir / f"summary_stats_{args.window}_{args.mode}.json"

    logger.info("=" * 60)
    logger.info(f"Phase 1 决策引擎启动（窗口={args.window}, 模式={args.mode}）")
    logger.info(f"模型路径: {model_path}")
    logger.info("=" * 60)

    # 1. 检查文件
    for fpath, name in [(model_path, "模型"), (NODE_PARAMS_PATH, "节点参数"),
                        (THRESHOLD_PATH, "动态阈值"), (MINMAX_PATH, "MinMax参数"),
                        (TEST_DATA_PATH, "测试数据目录")]:
        if not fpath.exists():
            logger.error(f"{name}文件不存在: {fpath}")
            sys.exit(1)

    # 2. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"模型加载成功，使用设备: {device}")

    # 3. 加载节点 MinMax 参数
    with open(MINMAX_PATH, 'rb') as f:
        node_minmax = pickle.load(f)

    # 4. 加载节点月度参数
    node_params_df = pd.read_csv(NODE_PARAMS_PATH)
    node_params_df['date'] = pd.to_datetime(node_params_df[['year', 'month']].assign(day=1))

    # 5. 加载动态阈值
    thresholds = load_dynamic_thresholds(THRESHOLD_PATH)

    # 6. 遍历所有测试节点
    node_dirs = sorted(TEST_DATA_PATH.glob("node_*"))
    all_results = []
    total_time = 0.0
    total_samples = 0

    window_size = WINDOW_SIZE[args.window]

    for node_dir in node_dirs:
        node_id = int(node_dir.name.split('_')[1])
        if node_id == 8025:
            continue
        logger.info(f"处理节点 {node_id}...")
        test_file = node_dir / "test.pkl"
        if not test_file.exists():
            logger.warning(f"节点 {node_id} 的测试文件不存在，跳过")
            continue

        dataset = MinMaxBarcelonaDataset(test_file, node_id, node_minmax, window_size)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        node_params = node_params_df[node_params_df['node_id'] == node_id]

        for x, y, dates, hours, holidays in loader:
            x = x.to(device)
            start_time = time.time()
            with torch.no_grad():
                pred_norm = model(x).cpu().numpy()  # (batch, 4)
            elapsed = time.time() - start_time
            total_time += elapsed * len(x)
            total_samples += len(x)

            data_min, data_max = node_minmax[node_id]
            pred_real = pred_norm * (data_max - data_min) + data_min
            y_real = y.cpu().numpy() * (data_max - data_min) + data_min

            for i in range(len(x)):
                if args.mode == 'first':
                    t_range = [0]
                else:
                    t_range = range(4)

                for t in t_range:
                    pred_val = pred_real[i, t]
                    real_val = y_real[i, t]
                    hour_code = hours[i][t].item()
                    is_holiday = holidays[i][t].item()
                    key = (hour_code, is_holiday)
                    if key not in thresholds:
                        logger.warning(f"节点 {node_id} 缺少时段 ({hour_code},{is_holiday}) 阈值，跳过")
                        continue
                    low_th, high_th = thresholds[key]

                    decision = make_decision(pred_val, low_th, high_th)
                    energy_saved, actual_energy = compute_savings(pred_val, decision)

                    sample_dt = dates[i][t]
                    sample_year = sample_dt.year
                    sample_month = sample_dt.month
                    param_row = node_params[(node_params['year'] == sample_year) &
                                            (node_params['month'] == sample_month)]
                    if len(param_row) == 0:
                        logger.warning(f"节点 {node_id} 缺失 {sample_year}-{sample_month} 参数，跳过")
                        continue
                    price = param_row.iloc[0]['price_euro_kwh']
                    carbon = param_row.iloc[0]['carbon_kg_kwh']

                    cost_saved = energy_saved * price
                    carbon_saved = energy_saved * carbon

                    correct = (real_val > high_th and decision == "迁移") or \
                              (real_val < low_th and decision == "休眠") or \
                              (low_th <= real_val <= high_th and decision == "正常")

                    all_results.append({
                        'node_id': node_id,
                        'date': sample_dt.strftime('%Y-%m-%d'),
                        'hour_code': hour_code,
                        'is_holiday': is_holiday,
                        'pred_kw': pred_val,
                        'real_kw': real_val,
                        'decision': decision,
                        'energy_saved_kwh': energy_saved,
                        'cost_saved_eur': cost_saved,
                        'carbon_saved_kg': carbon_saved,
                        'price_euro_kwh': price,
                        'carbon_kg_kwh': carbon,
                        'decision_correct': correct,
                        'latency_ms': elapsed * 1000 / len(x)
                    })

    if not all_results:
        logger.error("没有生成任何决策结果")
        sys.exit(1)

    # 7. 保存详细结果
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_csv, index=False)
    logger.info(f"决策结果已保存至 {output_csv}")

    # 8. 统计摘要
    total_energy_saved = df_results['energy_saved_kwh'].sum()
    total_cost_saved = df_results['cost_saved_eur'].sum()
    total_carbon_saved = df_results['carbon_saved_kg'].sum()
    accuracy = df_results['decision_correct'].mean() * 100
    avg_latency = df_results['latency_ms'].mean()

    stats = {
        'window': args.window,
        'mode': args.mode,
        'total_energy_saved_kwh': total_energy_saved,
        'total_cost_saved_eur': total_cost_saved,
        'total_carbon_saved_kg': total_carbon_saved,
        'decision_accuracy_percent': accuracy,
        'avg_latency_ms': avg_latency,
        'num_samples': len(df_results),
        'num_nodes': df_results['node_id'].nunique(),
        'time_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sleep_factor': SLEEP_FACTOR,
        'migrate_factor': MIGRATE_FACTOR
    }

    with open(output_stats, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"统计摘要已保存至 {output_stats}")

    # 9. 生成可视化图表（除非指定 --no-plots）
    if not args.no_plots:
        generate_visualizations(df_results, stats, output_dir, args.window, args.mode)
    else:
        logger.info("跳过可视化图表生成（--no-plots）")

    # 打印摘要
    logger.info("=" * 60)
    logger.info(f"决策摘要（窗口={args.window}, 模式={args.mode}）")
    logger.info(f"  总样本数: {stats['num_samples']}")
    logger.info(f"  总节省能耗: {total_energy_saved:.2f} kWh")
    logger.info(f"  总节省成本: {total_cost_saved:.2f} €")
    logger.info(f"  总碳减排: {total_carbon_saved:.2f} kg CO₂")
    logger.info(f"  决策准确率: {accuracy:.2f}%")
    logger.info(f"  平均决策延迟: {avg_latency:.2f} ms")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()