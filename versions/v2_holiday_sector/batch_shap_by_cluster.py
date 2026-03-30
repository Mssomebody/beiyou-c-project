#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级批量 SHAP 分析：基于节点聚类结果分层抽样，比较 1天 vs 7天窗口
自动路径定位，顺序划分数据集，支持单节点测试模式
"""

import sys
import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ============================================================
# 自动路径定位（基于脚本所在目录）
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # versions/v2_holiday_sector -> 项目根目录
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_2019_2022"
CLUSTER_FILE = PROJECT_ROOT / "results" / "barcelona_clustering" / "node_classification.csv"
OUTPUT_DIR = SCRIPT_DIR / "results" / "shap_multi_nodes"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 配置类
# ============================================================
class Config:
    def __init__(self, node=None):
        self.node = node
        self.data_dir = DATA_DIR
        self.batch_size = 64
        self.epochs = 10
        self.learning_rate = 0.001
        self.hidden_dim = 64
        self.num_layers = 2
        self.dropout = 0.2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.shap_background = 100
        self.shap_samples = 500
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

# ============================================================
# 数据集类
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, window_size=4, predict_size=4):
        df = pd.read_pickle(data_path)
        self.energy = df['Valor_norm'].values.astype(np.float32)
        self.window_size = window_size
        self.predict_size = predict_size
        self.indices = self._build_indices()

    def _build_indices(self):
        indices = []
        total_len = len(self.energy)
        for i in range(total_len - self.window_size - self.predict_size + 1):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.energy[start:start + self.window_size]
        y = self.energy[start + self.window_size:start + self.window_size + self.predict_size]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y)

# ============================================================
# LSTM 模型
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)

# ============================================================
# 训练函数（顺序划分，与单节点分析一致）
# ============================================================
def train_model(dataset, config, model_name):
    # 时间顺序划分：前80%训练，后20%验证
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = LSTMPredictor(
        input_dim=1,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_dim=4,
        dropout=config.dropout
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"temp_model_{model_name}.pth")

    model.load_state_dict(torch.load(f"temp_model_{model_name}.pth"))

    # 计算验证集 sMAPE
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(config.device)
            out = model(x).cpu().numpy()
            all_preds.append(out)
            all_trues.append(y.numpy())
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    smape = np.mean(2 * np.abs(preds - trues) / (np.abs(preds) + np.abs(trues) + 1e-8)) * 100
    return model, best_val_loss, smape

# ============================================================
# SHAP 分析（自适应版）
# ============================================================
def shap_analysis(model, dataset, window_size, config, title_prefix):
    background_indices = np.random.choice(len(dataset), config.shap_background, replace=False)
    background = torch.stack([dataset[i][0] for i in background_indices]).numpy()
    background = background.reshape(background.shape[0], -1)

    sample_indices = np.random.choice(len(dataset), config.shap_samples, replace=False)
    samples = torch.stack([dataset[i][0] for i in sample_indices]).numpy()
    samples_flat = samples.reshape(samples.shape[0], -1)

    def predict_fn(x_flat):
        x = x_flat.reshape(-1, window_size, 1)
        x_tensor = torch.FloatTensor(x).to(config.device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()
        return pred

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(samples_flat, nsamples=100)

    # 自适应处理 shap_values
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values)          # (n_outputs, n_samples, n_features)
    else:
        if shap_values.ndim == 3 and shap_values.shape[-1] == 4:
            shap_arr = np.moveaxis(shap_values, -1, 0)
        else:
            shap_arr = shap_values

    if shap_arr.shape[0] != 4:
        for perm in [(1,0,2), (2,1,0)]:
            test = shap_arr.transpose(perm)
            if test.shape[0] == 4:
                shap_arr = test
                break
        else:
            raise ValueError(f"无法调整 SHAP 数组形状: {shap_arr.shape}")

    shap_timestep_importance = np.mean(np.abs(shap_arr), axis=(0, 1))
    return shap_timestep_importance

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='批量 SHAP 分析（基于聚类）')
    parser.add_argument('--samples_per_cluster', type=int, default=3,
                        help='每个聚类类别抽取的节点数（测试模式下忽略）')
    parser.add_argument('--test_node', type=int, default=None,
                        help='仅测试单个节点（不进行分层抽样）')
    args = parser.parse_args()

    # 设置日志
    log_file = OUTPUT_DIR / f"batch_shap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("批量 SHAP 分析（分层抽样）")
    logger.info("="*60)
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"聚类文件: {CLUSTER_FILE}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    if args.test_node is not None:
        logger.info(f"测试模式：仅处理节点 {args.test_node}")
    else:
        logger.info(f"每类抽样数: {args.samples_per_cluster}")

    # 1. 检查聚类文件
    if not CLUSTER_FILE.exists():
        logger.error(f"聚类文件不存在: {CLUSTER_FILE}")
        sys.exit(1)

    df = pd.read_csv(CLUSTER_FILE)
    logger.info(f"聚类文件列: {list(df.columns)}")
    logger.info(f"总节点数: {len(df)}")

    node_col = 'node' if 'node' in df.columns else 'node_id'
    cluster_col = 'cluster' if 'cluster' in df.columns else 'label'
    if cluster_col not in df.columns:
        logger.error(f"未找到聚类列，可用列: {list(df.columns)}")
        sys.exit(1)

    # 2. 确定节点列表（测试模式或分层抽样）
    if args.test_node is not None:
        selected_nodes = [args.test_node]
        # 检查该节点是否在聚类文件中
        if args.test_node not in df[node_col].values:
            logger.warning(f"测试节点 {args.test_node} 不在聚类文件中，仍将尝试处理（可能数据存在）")
    else:
        clusters = df[cluster_col].unique()
        selected_nodes = []
        for cid in clusters:
            nodes_in_cluster = df[df[cluster_col] == cid][node_col].tolist()
            n_sample = min(args.samples_per_cluster, len(nodes_in_cluster))
            sampled = np.random.choice(nodes_in_cluster, n_sample, replace=False).tolist()
            selected_nodes.extend(sampled)
            logger.info(f"类别 {cid}: {len(nodes_in_cluster)} 个节点，抽样 {n_sample} 个: {sampled}")

    logger.info(f"共选中 {len(selected_nodes)} 个节点: {selected_nodes}")

    # 保存选中节点列表
    with open(OUTPUT_DIR / "selected_nodes.txt", 'w') as f:
        for node in selected_nodes:
            f.write(f"{node}\n")

    # 3. 对每个节点进行分析
    results = []
    for node in tqdm(selected_nodes, desc="处理节点"):
        logger.info(f"\n{'='*40}\n处理节点 {node}\n{'='*40}")
        config = Config(node=node)
        train_path = config.data_dir / f"node_{node}" / "train.pkl"
        if not train_path.exists():
            logger.warning(f"节点 {node} 数据不存在，跳过")
            continue

        try:
            # 1天窗口
            dataset_1d = TimeSeriesDataset(train_path, window_size=4, predict_size=4)
            model_1d, val_loss_1d, smape_1d = train_model(dataset_1d, config, f"{node}_1d")
            shap_1d = shap_analysis(model_1d, dataset_1d, 4, config, f"{node}_1d")

            # 7天窗口
            dataset_7d = TimeSeriesDataset(train_path, window_size=28, predict_size=4)
            model_7d, val_loss_7d, smape_7d = train_model(dataset_7d, config, f"{node}_7d")
            shap_7d = shap_analysis(model_7d, dataset_7d, 28, config, f"{node}_7d")

            # 验证 shap_7d 长度是否为 28
            if len(shap_7d) != 28:
                logger.warning(f"节点 {node} 7天窗口 SHAP 长度异常: {len(shap_7d)}，预期 28，跳过")
                continue

            # 保存 SHAP 数组
            np.save(OUTPUT_DIR / f"node_{node}_shap_1d.npy", shap_1d)
            np.save(OUTPUT_DIR / f"node_{node}_shap_7d.npy", shap_7d)

            results.append({
                'node': node,
                'val_loss_1d': val_loss_1d,
                'smape_1d': smape_1d,
                'val_loss_7d': val_loss_7d,
                'smape_7d': smape_7d,
                'shap_1d': shap_1d,
                'shap_7d': shap_7d
            })

            # 清理临时模型
            for f in [f"temp_model_{node}_1d.pth", f"temp_model_{node}_7d.pth"]:
                if Path(f).exists():
                    Path(f).unlink()

        except Exception as e:
            logger.error(f"节点 {node} 分析失败: {e}", exc_info=True)

    if not results:
        logger.error("没有成功分析任何节点，退出")
        sys.exit(1)

    # 4. 汇总与可视化
    logger.info("\n生成汇总图表...")
    nodes = [r['node'] for r in results]
    smape_1d = [r['smape_1d'] for r in results]
    smape_7d = [r['smape_7d'] for r in results]

    # 箱线图
    plt.figure(figsize=(6, 5))
    data_to_plot = [smape_1d, smape_7d]
    plt.boxplot(data_to_plot, labels=['1天窗口', '7天窗口'])
    plt.ylabel('sMAPE (%)')
    plt.title('各节点预测精度对比')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'smape_comparison_boxplot.png', dpi=150)
    plt.close()

    # 7天窗口逐日重要性热力图
    daily_importance = []
    for r in results:
        shap_7d = r['shap_7d']
        if len(shap_7d) == 28:
            daily = shap_7d.reshape(7, 4).mean(axis=1)
            daily_importance.append(daily)
        else:
            logger.warning(f"节点 {r['node']} SHAP 长度异常，不参与热力图")
    if daily_importance:
        daily_importance = np.array(daily_importance)
        plt.figure(figsize=(10, max(4, len(daily_importance)*0.3)))
        sns.heatmap(daily_importance, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[f'Day{i+1}' for i in range(7)],
                    yticklabels=[f'Node {n}' for n in nodes[:len(daily_importance)]])
        plt.xlabel('天数')
        plt.ylabel('节点')
        plt.title('各节点 7 天窗口逐日特征重要性 (平均 SHAP)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'daily_importance_heatmap.png', dpi=150)
        plt.close()
    else:
        logger.warning("没有有效的 SHAP 数据生成热力图")

    # 保存汇总表格
    df_summary = pd.DataFrame({
        'node': nodes,
        'smape_1d': smape_1d,
        'smape_7d': smape_7d,
        'val_loss_1d': [r['val_loss_1d'] for r in results],
        'val_loss_7d': [r['val_loss_7d'] for r in results]
    })
    df_summary.to_csv(OUTPUT_DIR / 'summary.csv', index=False)

    # 统计
    better_1d = sum(1 for s1, s7 in zip(smape_1d, smape_7d) if s1 < s7)
    logger.info(f"成功分析 {len(results)} 个节点，其中 {better_1d} 个节点 1 天窗口 sMAPE 更低（更好），占比 {better_1d/len(results)*100:.1f}%")
    logger.info(f"所有结果已保存至: {OUTPUT_DIR}")
    logger.info("="*60)

if __name__ == "__main__":
    main()