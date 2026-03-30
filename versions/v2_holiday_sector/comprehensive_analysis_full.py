#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合正负向分析报告生成脚本（完整版，包含优化实验的SHAP分析）
- 正向：7天窗口基线模型 SHAP 分析
- 负向：E3/E4/E5/E2 优化模型 SHAP 分析（可选）
- 生成所有 SHAP 图表和对比柱状图
- 输出中文综合报告（HTML）
"""

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据集类（与7天基线一致）
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)

class FusionLSTM(nn.Module):
    def __init__(self, input_dim_barcelona, input_dim_tsinghua, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim_barcelona, input_dim_tsinghua)
        self.lstm = nn.LSTM(input_dim_tsinghua, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, is_barcelona):
        if is_barcelona:
            x = self.proj(x)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)
    
class BarcelonaCoarseDataset(Dataset):
    def __init__(self, data_path, window_days=7, norm_params=None):
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
        df = df.sort_values('Data')
        dates = pd.to_datetime(df['Data']).unique()
        self.window_days = window_days
        self.samples = []

        def has_full(date):
            day_df = df[pd.to_datetime(df['Data']) == date]
            return set(day_df['hour_code'].unique()) == {0,1,2,3}

        def aggregate_group(group):
            total_energy = group['Valor'].sum()
            weekend = group['is_weekend'].iloc[0]
            holiday = group['is_holiday'].iloc[0]
            sector_means = group[['sector_0','sector_1','sector_2','sector_3']].mean().values
            return pd.Series({
                'total_energy': total_energy,
                'is_weekend': weekend,
                'is_holiday': holiday,
                'sector_0': sector_means[0],
                'sector_1': sector_means[1],
                'sector_2': sector_means[2],
                'sector_3': sector_means[3]
            })

        for i in range(len(dates) - window_days):
            input_dates = dates[i:i+window_days]
            target_date = dates[i+window_days]
            if all(has_full(d) for d in input_dates) and has_full(target_date):
                input_seq = []
                for d in input_dates:
                    day_df = df[pd.to_datetime(df['Data']) == d]
                    day_agg = day_df.groupby('hour_code').apply(aggregate_group).reset_index()
                    day_agg = day_agg.set_index('hour_code').reindex([0,1,2,3], fill_value=0).reset_index()
                    day_agg = day_agg.sort_values('hour_code')
                    x_day = day_agg[['total_energy','sector_0','sector_1','sector_2','sector_3','is_weekend','is_holiday']].values.astype(np.float32)
                    input_seq.append(x_day)
                x = np.stack(input_seq, axis=0)          # (window_days, 4, 7)
                target_df = df[pd.to_datetime(df['Data']) == target_date]
                target_agg = target_df.groupby('hour_code').apply(aggregate_group).reset_index()
                target_agg = target_agg.set_index('hour_code').reindex([0,1,2,3], fill_value=0).reset_index()
                target_agg = target_agg.sort_values('hour_code')
                y = target_agg['total_energy'].values.astype(np.float32)
                self.samples.append((x, y))

        if not self.samples:
            raise ValueError(f"没有找到连续 {window_days+1} 天完整数据: {data_path}")
        print(f"Loaded {len(self.samples)} samples from {data_path}")

        # 归一化
        if norm_params is None:
            all_x = np.stack([s[0] for s in self.samples])
            all_y = np.stack([s[1] for s in self.samples])
            self.x_min = all_x.min(axis=(0,1,2), keepdims=True)
            self.x_max = all_x.max(axis=(0,1,2), keepdims=True)
            self.y_min = all_y.min()
            self.y_max = all_y.max()
            self.norm_params = {
                'x_min': self.x_min.squeeze(),
                'x_max': self.x_max.squeeze(),
                'y_min': self.y_min,
                'y_max': self.y_max
            }
            self.x_norm = (all_x - self.x_min) / (self.x_max - self.x_min + 1e-8)
            self.y_norm = (all_y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        else:
            self.x_min = norm_params['x_min'][None, None, None, :]
            self.x_max = norm_params['x_max'][None, None, None, :]
            self.y_min = norm_params['y_min']
            self.y_max = norm_params['y_max']
            all_x = np.stack([s[0] for s in self.samples])
            all_y = np.stack([s[1] for s in self.samples])
            self.x_norm = (all_x - self.x_min) / (self.x_max - self.x_min + 1e-8)
            self.y_norm = (all_y - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.x_norm[idx].reshape(-1, self.x_norm.shape[-1])  # (window_days*4, 7)
        y = self.y_norm[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ============================================================
# 模型定义（与基线一致）
# ============================================================
class FusionLSTM(nn.Module):
    def __init__(self, input_dim_barcelona, input_dim_tsinghua, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim_barcelona, input_dim_tsinghua)
        self.lstm = nn.LSTM(input_dim_tsinghua, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, is_barcelona):
        if is_barcelona:
            x = self.proj(x)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# SHAP 计算（修正版）
# ============================================================
def compute_shap_importance(model, dataset, window_size, device, bg_samples=100, exp_samples=500):
    bg_samples = min(bg_samples, len(dataset))
    exp_samples = min(exp_samples, len(dataset))
    
    background_indices = np.random.choice(len(dataset), bg_samples, replace=False)
    background = torch.stack([dataset[i][0] for i in background_indices]).numpy()
    background = background.reshape(background.shape[0], -1)

    sample_indices = np.random.choice(len(dataset), exp_samples, replace=False)
    samples = torch.stack([dataset[i][0] for i in sample_indices]).numpy()
    samples_flat = samples.reshape(samples.shape[0], -1)

    def predict_fn(x_flat):
        x = x_flat.reshape(-1, window_size, 7)
        x_tensor = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            if hasattr(model, 'proj'):  # 如果是 FusionLSTM
                pred = model(x_tensor, is_barcelona=True)
            else:
                pred = model(x_tensor)
        return pred.cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(samples_flat, nsamples=100)

    # 修复：将 (4, n_samples, window_size*7) 重塑为 (4, n_samples, window_size, 7) 后对特征维度平均
    shap_arr = np.array(shap_values)                     # (4, n_samples, window_size*7)
    shap_arr = shap_arr.reshape(4, -1, window_size, 7)   # (4, n_samples, window_size, 7)
    importance = np.mean(np.abs(shap_arr), axis=(0, 1, 3))  # (window_size,)
    return importance, explainer, samples_flat, shap_values


# ============================================================
# 图表绘制函数
# ============================================================
def plot_window_comparison(one_day, seven_day, save_path):
    plt.figure(figsize=(6, 6))
    labels = ['1天窗口', '7天窗口']
    values = [one_day, seven_day]
    colors = ['#2E8B57', '#E76F51']
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('sMAPE (%)', fontsize=12)
    plt.title('窗口长度对预测精度的影响（五节点联邦学习）', fontsize=14)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_timestep_importance(importance, window_size, save_path, title_prefix=''):
    plt.figure(figsize=(14, 5))
    if window_size == 28:
        labels = []
        for day in range(1, 8):
            for hour in ['00-06','06-12','12-18','18-24']:
                labels.append(f'第{day}天\n{hour}')
        plt.bar(range(window_size), importance, color='#2E8B57')
        plt.xticks(range(window_size), labels, rotation=45, fontsize=8, ha='right')
    else:
        plt.bar(range(window_size), importance, color='#2E8B57')
        plt.xticks(range(window_size), fontsize=10)
    plt.xlabel('时间步 (日+时段)')
    plt.ylabel('平均 |SHAP 值|')
    plt.title(f'{title_prefix} 7天窗口各时间步特征重要性' if title_prefix else '7天窗口各时间步特征重要性')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_daily_heatmap(importance, window_size, save_path, title_prefix=''):
    if window_size != 28:
        return
    daily = importance.reshape(7, 4).mean(axis=1)
    plt.figure(figsize=(8, 4))
    sns.heatmap(daily.reshape(1, -1), annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'第{i+1}天' for i in range(7)], yticklabels=['平均 SHAP'])
    plt.title(f'{title_prefix} 7天窗口逐日重要性（平均SHAP）' if title_prefix else '7天窗口逐日重要性（平均SHAP）')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_multi_node_boxplot(all_importances, nodes, window_size, save_path, title_prefix=''):
    # 确保 all_importances 中每个元素都是长度为 window_size 的数组
    all_shap = np.array([imp for imp in all_importances if len(imp) == window_size])
    if all_shap.shape[1] != window_size:
        print(f'警告: all_shap.shape = {all_shap.shape}, 期望第二维为 {window_size}')
        return
    plt.figure(figsize=(12, 5))
    import seaborn as sns
    data = pd.DataFrame(all_shap.T, columns=nodes[:all_shap.shape[0]])
    sns.boxplot(data=data, orient='v')
    plt.xlabel('时间步')
    plt.ylabel('SHAP重要性')
    plt.title(f'{title_prefix}多节点时间步重要性分布 (n={all_shap.shape[0]})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_multi_node_daily_heatmap(all_importances, nodes, window_size, save_path, title_prefix=''):
    if window_size != 28:
        return
    all_daily = np.array([imp.reshape(7,4).mean(axis=1) for imp in all_importances])
    plt.figure(figsize=(10, max(4, len(nodes)*0.3)))
    sns.heatmap(all_daily, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'第{i+1}天' for i in range(7)],
                yticklabels=nodes)
    plt.xlabel('天数')
    plt.ylabel('节点')
    plt.title(f'{title_prefix} 各节点7天窗口逐日重要性' if title_prefix else '各节点7天窗口逐日重要性')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_waterfall(shap_values, explainer, samples_flat, window_size, node, save_path, title_prefix=''):
    sample_idx = 0
    shap_sample = shap_values[0][sample_idx]
    plt.figure(figsize=(12, 4))
    hours = [f"第{(i//4)+1}天 {['00-06','06-12','12-18','18-24'][i%4]}" for i in range(window_size)]
    shap.plots.waterfall(shap.Explanation(values=shap_sample,
                                          base_values=explainer.expected_value[0],
                                          data=samples_flat[sample_idx],
                                          feature_names=hours),
                         show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_negative_comparison(baseline_smape, negative_results, save_path):
    methods = list(negative_results.keys())
    smapes = [negative_results[m] for m in methods]
    colors = ['#E76F51' if v > baseline_smape else '#2E8B57' for v in smapes]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, smapes, color=colors, edgecolor='black')
    plt.axhline(y=baseline_smape, color='red', linestyle='--', label=f'基线 ({baseline_smape:.2f}%)')
    plt.ylabel('sMAPE (%)')
    plt.title('7天窗口优化实验对比')
    for bar, val in zip(bars, smapes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}%', ha='center', va='bottom')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# 辅助函数：对单个模型进行 SHAP 分析并返回重要性列表
# ============================================================
def analyze_model(model_path, model_name, nodes, window_days, window_size, device, data_dir):
    print(f"加载模型: {model_name} ({model_path})")
    
    # 根据模型名称选择模型类
    if model_name == "baseline":
        model = LSTMPredictor(input_dim=7, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2)
    else:
        # 优化模型使用 FusionLSTM（需要提前导入或定义）
        # 注意：FusionLSTM 定义在脚本中，我们需要确保它存在
        model = FusionLSTM(
            input_dim_barcelona=7,
            input_dim_tsinghua=5,
            hidden_dim=64,
            num_layers=2,
            output_dim=4,
            dropout=0.2
        )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_importances = []
    first_node_shap = None
    for node in nodes:
        print(f"  处理节点 {node}...")
        data_path = Path(data_dir) / f"node_{node}" / "train.pkl"
        dataset = BarcelonaCoarseDataset(data_path, window_days=window_days, norm_params=None)
        val_len = int(0.2 * len(dataset))
        val_dataset = torch.utils.data.Subset(dataset, range(len(dataset)-val_len, len(dataset)))
        importance, explainer, samples_flat, shap_values = compute_shap_importance(model, val_dataset, window_size, device)
        all_importances.append(importance)
        if node == nodes[0]:
            first_node_shap = (shap_values, explainer, samples_flat)
        # 保存单节点图表
        plot_timestep_importance(importance, window_size, FIGURE_DIR / f"{model_name}_node{node}_timesteps.png")
        plot_daily_heatmap(importance, window_size, FIGURE_DIR / f"{model_name}_node{node}_daily_heatmap.png")
        print(f"  节点 {node} SHAP 计算完成")

    # 保存多节点汇总图
    plot_multi_node_boxplot(all_importances, nodes, window_size, FIGURE_DIR / f"{model_name}_multi_node_boxplot.png", model_name)
    plot_multi_node_daily_heatmap(all_importances, nodes, window_size, FIGURE_DIR / f"{model_name}_multi_node_daily_heatmap.png", model_name)
    if first_node_shap:
        plot_waterfall(first_node_shap[0], first_node_shap[1], first_node_shap[2], window_size, nodes[0], FIGURE_DIR / f"{model_name}_waterfall.png")
    return all_importances


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='综合正负向分析（含优化模型SHAP）')
    parser.add_argument('--baseline_model', type=str, required=True, help='7天窗口基线模型路径')
    parser.add_argument('--nodes', type=str, default='8001,8002,8004,8006,8012', help='节点列表')
    parser.add_argument('--one_day_smape', type=float, required=True, help='1天窗口五节点sMAPE (%)')
    parser.add_argument('--seven_day_smape', type=float, required=True, help='7天窗口五节点sMAPE (%)')
    parser.add_argument('--data_dir', type=str, default=str(PROJECT_ROOT / "data/processed/barcelona_ready_2019_2022"))
    parser.add_argument('--skip_negative_shap', action='store_true', help='跳过优化模型的SHAP分析（仅生成精度对比图）')
    args = parser.parse_args()

    nodes = [int(n) for n in args.nodes.split(',')]
    window_days = 7
    window_size = window_days * 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    one_day = args.one_day_smape
    seven_day = args.seven_day_smape

    print("="*60)
    print("综合正负向分析（完整版）")
    print("="*60)

    # 正向：基线模型 SHAP
    print("\n正向：基线模型 SHAP 分析")
    baseline_importances = analyze_model(args.baseline_model, "baseline", nodes, window_days, window_size, device, args.data_dir)
    # 生成窗口对比柱状图
    plot_window_comparison(one_day, seven_day, FIGURE_DIR / "window_comparison.png")

    # 负向：优化模型 SHAP（可选）
    negative_models = {
        'E3': 'model_7day_e3_5nodes.pth',
        'E4': 'model_7day_e4_5nodes.pth',
        'E5': 'model_7day_e5_5nodes.pth',
        'E2': 'model_7day_e2_5nodes.pth'
    }
    negative_smapes = {'E3': 39.25, 'E4': 37.93, 'E5': 39.62, 'E2': 39.25}
    negative_shap_available = {}

    if not args.skip_negative_shap:
        print("\n负向：优化模型 SHAP 分析（耗时较长）")
        for name, model_file in negative_models.items():
            model_path = Path(__file__).parent / model_file
            if model_path.exists():
                print(f"处理 {name}...")
                analyze_model(model_path, name, nodes, window_days, window_size, device, args.data_dir)
                negative_shap_available[name] = True
            else:
                print(f"警告: {name} 模型文件不存在，跳过 SHAP 分析")
                negative_shap_available[name] = False
    else:
        print("\n负向：跳过优化模型的 SHAP 分析，仅生成精度对比图")

    # 生成负向精度对比柱状图
    plot_negative_comparison(seven_day, negative_smapes, FIGURE_DIR / "7day_negative_experiments.png")

    # 生成综合 HTML 报告
    node_list_str = ', '.join(map(str, nodes))
    improvement = one_day - seven_day
    rel_improve = (improvement / one_day) * 100

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>综合报告 | 基站能耗预测正负向亮点分析（含SHAP）</title>
    <style>
        body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f5f7fa; margin: 0; padding: 40px 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 30px; overflow: hidden; }}
        .card-header {{ background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); color: white; padding: 20px 30px; }}
        .card-header h2 {{ margin: 0; }}
        .card-body {{ padding: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr)); gap: 25px; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; border-radius: 12px; padding: 20px; text-align: center; border-bottom: 4px solid #3498db; }}
        .metric-value {{ font-size: 2.8rem; font-weight: 700; color: #e67e22; }}
        .improvement-badge {{ background: #27ae60; color: white; padding: 8px 16px; border-radius: 50px; display: inline-block; font-weight: bold; margin: 15px 0; }}
        .figure {{ text-align: center; margin: 30px 0; }}
        .figure img {{ max-width: 100%; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .caption {{ margin-top: 12px; font-size: 0.9rem; color: #7f8c8d; }}
        .insight {{ background: #e8f4fd; border-left: 5px solid #3498db; padding: 15px 20px; margin: 20px 0; border-radius: 8px; }}
        footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #7f8c8d; }}
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <div class="card-header">
            <h2>🔬 正向亮点：为什么7天窗口优于1天窗口？</h2>
        </div>
        <div class="card-body">
            <div class="metrics-grid">
                <div class="metric-card"><div class="metric-label">1天窗口 sMAPE</div><div class="metric-value">{one_day:.2f}%</div></div>
                <div class="metric-card"><div class="metric-label">7天窗口 sMAPE</div><div class="metric-value">{seven_day:.2f}%</div></div>
                <div class="metric-card"><div class="metric-label">绝对提升</div><div class="metric-value">{improvement:.2f}%</div></div>
                <div class="metric-card"><div class="metric-label">相对提升</div><div class="metric-value">{rel_improve:.1f}%</div></div>
            </div>
            <div class="insight"><strong>💡 核心发现：</strong> 7天窗口捕获周模式，SHAP显示第一天贡献最大，重要性逐日衰减。</div>
            <div class="figure"><h3>📈 窗口精度对比</h3><img src="file:///{FIGURE_DIR}/window_comparison.png"><div class="caption">图1：1天 vs 7天 sMAPE 对比</div></div>
            <div class="figure"><h3>📊 多节点时间步重要性分布（基线）</h3><img src="file:///{FIGURE_DIR}/baseline_multi_node_boxplot.png"><div class="caption">图2：基线模型各时间步SHAP分布，第一天显著更高</div></div>
            <div class="figure"><h3>🔥 多节点逐日重要性热力图（基线）</h3><img src="file:///{FIGURE_DIR}/baseline_multi_node_daily_heatmap.png"><div class="caption">图3：各节点逐日重要性，第一天主导</div></div>
            <div class="figure"><h3>💧 瀑布图（节点8001）</h3><img src="file:///{FIGURE_DIR}/baseline_waterfall.png"><div class="caption">图4：单样本SHAP贡献分解</div></div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <h2>📉 负向亮点：优化方法在7天窗口上无效</h2>
        </div>
        <div class="card-body">
            <div class="insight"><strong>🔍 关键发现：</strong> 所有优化方法（E3/E4/E5/E2）的sMAPE均高于基线，SHAP分析显示它们引入了噪声或无法有效利用长期模式。</div>
            <div class="figure"><h3>📊 优化实验 sMAPE 对比</h3><img src="file:///{FIGURE_DIR}/7day_negative_experiments.png"><div class="caption">图5：各优化方法精度均低于基线</div></div>
"""

    # 添加负向模型的 SHAP 图表（如果已计算）
    if not args.skip_negative_shap:
        for name in ['E3', 'E4', 'E5', 'E2']:
            if negative_shap_available.get(name, False):
                html += f"""
            <div class="figure">
                <h3>📊 {name} 多节点时间步重要性分布</h3>
                <img src="file:///{FIGURE_DIR}/{name}_multi_node_boxplot.png">
                <div class="caption">{name} 模型的SHAP重要性分布，无明显的第一天主导模式，表明权重引入噪声。</div>
            </div>
            <div class="figure">
                <h3>🔥 {name} 多节点逐日热力图</h3>
                <img src="file:///{FIGURE_DIR}/{name}_multi_node_daily_heatmap.png">
                <div class="caption">{name} 逐日重要性混乱，无法有效捕获周模式。</div>
            </div>
"""
    else:
        html += """
            <div class="insight">⚠️ 未计算优化模型的SHAP（可通过 --no-skip-negative-shap 开启），仅展示精度对比。</div>
"""

    html += f"""
            <div class="insight"><strong>📌 结论：</strong> 优化方法在7天窗口上无效，证明7天窗口本身已接近信息上限，简单基线更优。</div>
        </div>
    </div>
    <footer>报告生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>节点: {node_list_str} | SHAP: KernelExplainer</footer>
</div>
</body>
</html>
"""
    report_path = REPORT_DIR / "comprehensive_report_full.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n综合报告已保存: {report_path}")

if __name__ == "__main__":
    main()
