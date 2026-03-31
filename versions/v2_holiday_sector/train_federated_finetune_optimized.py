#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦微调 - 新口径41节点（五星级优化版）

功能：
- 加载预训练模型（旧口径联邦预训练）
- 在新口径41个节点上运行联邦微调（FedAvg + FedProx）
- 早停机制（基于验证集真实sMAPE）
- 学习率调度（ReduceLROnPlateau）
- 梯度裁剪
- 保存验证集最佳模型
- 最终输出测试集真实空间sMAPE
- 详细日志记录
- 可复现性（固定随机种子）

输入：
- 预训练模型: results/two_stage/model_fed_pretrain.pth
- 节点MinMax参数: versions/v2_holiday_sector/node_minmax.pkl
- 新口径数据: data/processed/barcelona_ready_2023_2025/node_*/train.pkl, val.pkl, test.pkl

输出：
- 最佳微调模型: decision/models/model_fed_finetune.pth
- 日志和中间结果: results/two_stage/finetune_log.txt
"""

import sys
import os
import pickle
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 输入文件
PRETRAIN_MODEL = PROJECT_ROOT / "results" / "two_stage" / "model_fed_pretrain.pth"
MINMAX_FILE = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
NEW_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_2023_2025"

# 输出
OUTPUT_MODEL = PROJECT_ROOT / "decision" / "models" / "model_fed_finetune.pth"
LOG_FILE = PROJECT_ROOT / "results" / "two_stage" / "finetune_log.txt"

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 固定随机种子（可复现性）
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# 模型定义（与预训练一致）
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


# ============================================================
# 数据集：使用原始 Valor 和节点 MinMax 参数归一化
# ============================================================
class MinMaxBarcelonaDataset(Dataset):
    def __init__(self, data_path: Path, node_id: int, node_minmax: Dict[int, Tuple[float, float]],
                 window_size=28, predict_size=4):
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.data_min, self.data_max = node_minmax[node_id]
        # 原始能耗
        self.energy = self.df['Valor'].values
        # 部门、周末、节假日特征
        sector_codes = self.df['sector_code'].values
        self.sector_onehot = self._one_hot_sector(sector_codes)
        self.holiday = self.df['is_holiday'].values
        self.weekend = self.df['is_weekend'].values
        self.indices = self._build_indices()

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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        # 输入特征：能耗（归一化）+ 部门 + 周末 + 节假日
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

        # 目标：未来四个时段的能耗（归一化）
        y = self.energy[start+self.window_size:start+self.window_size+self.predict_size]
        y = (y - self.data_min) / (self.data_max - self.data_min + 1e-8)
        y = torch.FloatTensor(y)

        return x, y


# ============================================================
# 联邦训练器
# ============================================================
class FederatedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.best_val_smape = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def create_model(self):
        return LSTMPredictor(
            input_dim=7,
            hidden_dim=64,
            num_layers=2,
            output_dim=4,
            dropout=0.2
        )

    def train_round(self, model, client_loaders, mu):
        """执行一轮联邦通信"""
        client_weights = []
        client_sizes = []
        client_losses = []

        for client_id, loader in client_loaders.items():
            local_model = self.create_model().to(self.device)
            local_model.load_state_dict(model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=self.config.lr)
            criterion = nn.MSELoss()

            local_loss = 0.0
            for _ in range(self.config.local_epochs):
                epoch_loss = 0.0
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(x)
                    loss = criterion(output, y)
                    if mu > 0:
                        prox_loss = 0.0
                        for param, global_param in zip(local_model.parameters(), model.parameters()):
                            prox_loss += torch.norm(param - global_param) ** 2
                        loss += (mu / 2) * prox_loss
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                local_loss += epoch_loss / len(loader) if len(loader) > 0 else 0.0
            client_losses.append(local_loss / self.config.local_epochs)
            client_weights.append(local_model.state_dict())
            client_sizes.append(len(loader.dataset))

        # 加权平均聚合
        total = sum(client_sizes)
        global_weights = {}
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key])
            for w, size in zip(client_weights, client_sizes):
                global_weights[key] += w[key] * (size / total)
        model.load_state_dict(global_weights)

        avg_loss = np.mean(client_losses)
        return model, avg_loss

    def evaluate_smape(self, model, loaders, node_minmax, real_space=True):
        """
        评估 sMAPE
        real_space: True 表示反归一化后计算真实空间 sMAPE，False 表示归一化空间
        """
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for node_id, loader in loaders.items():
                data_min, data_max = node_minmax[node_id]
                for x, y in loader:
                    x = x.to(self.device)
                    pred_norm = model(x).cpu().numpy()
                    target_norm = y.cpu().numpy()
                    if real_space:
                        pred = pred_norm * (data_max - data_min) + data_min
                        target = target_norm * (data_max - data_min) + data_min
                    else:
                        pred = pred_norm
                        target = target_norm
                    all_preds.append(pred)
                    all_targets.append(target)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        denominator = (np.abs(all_targets) + np.abs(all_preds)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(np.abs(all_targets - all_preds) / denominator) * 100
        return smape

    def train(self, client_train_loaders, client_val_loaders, client_test_loaders, node_minmax, rounds, mu):
        """完整训练流程，含早停和学习率调度"""
        model = self.create_model().to(self.device)

        # 加载预训练模型
        if self.config.pretrain_path and Path(self.config.pretrain_path).exists():
            state_dict = torch.load(self.config.pretrain_path, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"加载预训练模型: {self.config.pretrain_path}")
        else:
            logger.warning("未找到预训练模型，从头开始训练")

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 记录训练过程
        train_losses = []
        val_smapes = []

        for round_num in range(1, self.config.rounds + 1):
            model, avg_loss = self.train_round(model, client_train_loaders, mu)
            train_losses.append(avg_loss)

            # 验证集评估（真实空间）
            val_smape = self.evaluate_smape(model, client_val_loaders, node_minmax, real_space=True)
            val_smapes.append(val_smape)

            # 学习率调度
            scheduler.step(val_smape)

            logger.info(f"Round {round_num:3d}: train_loss = {avg_loss:.6f}, val_smape = {val_smape:.2f}%")

            # 早停与最佳模型保存
            if val_smape < self.best_val_smape:
                self.best_val_smape = val_smape
                self.patience_counter = 0
                self.best_model_state = model.state_dict().copy()
                logger.info(f"  -> 新的最佳模型 (val_smape={val_smape:.2f}%)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"早停触发，停止于第 {round_num} 轮")
                    break

        # 加载最佳模型
        model.load_state_dict(self.best_model_state)
        logger.info(f"训练结束，最佳验证 sMAPE = {self.best_val_smape:.2f}%")

        # 测试集评估
        test_smape = self.evaluate_smape(model, client_test_loaders, node_minmax, real_space=True)
        logger.info(f"测试集真实 sMAPE = {test_smape:.2f}%")

        # 保存最终模型
        torch.save(model.state_dict(), self.config.output_model)
        logger.info(f"最佳模型已保存至 {self.config.output_model}")

        return model, test_smape


# ============================================================
# 数据加载函数
# ============================================================
def load_node_loaders(node_ids: List[int], data_dir: Path, node_minmax: Dict,
                      split: str, batch_size: int = 64, shuffle: bool = True) -> Dict[int, DataLoader]:
    """为指定节点列表加载指定 split 的数据加载器"""
    loaders = {}
    for node_id in node_ids:
        node_dir = data_dir / f"node_{node_id}"
        pkl_file = node_dir / f"{split}.pkl"
        if not pkl_file.exists():
            logger.warning(f"节点 {node_id} 的 {split}.pkl 不存在，跳过")
            continue
        dataset = MinMaxBarcelonaDataset(pkl_file, node_id, node_minmax)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)
        loaders[node_id] = loader
    return loaders


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="联邦微调（新口径）")
    parser.add_argument('--rounds', type=int, default=20, help='通信轮数')
    parser.add_argument('--local_epochs', type=int, default=3, help='本地训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx 系数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--pretrain_path', type=str, default=str(PRETRAIN_MODEL), help='预训练模型路径')
    parser.add_argument('--output_model', type=str, default=str(OUTPUT_MODEL), help='输出模型路径')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("联邦微调启动")
    logger.info(f"设备: {args.device}")
    logger.info(f"通信轮数: {args.rounds}, 本地轮数: {args.local_epochs}")
    logger.info(f"学习率: {args.lr}, FedProx μ: {args.mu}")
    logger.info("=" * 60)

    # 1. 加载节点 MinMax 参数
    if not MINMAX_FILE.exists():
        logger.error(f"MinMax 参数文件不存在: {MINMAX_FILE}")
        logger.error("请先运行 generate_minmax_map.py 生成该文件")
        sys.exit(1)
    with open(MINMAX_FILE, 'rb') as f:
        node_minmax = pickle.load(f)
    node_ids = list(node_minmax.keys())  # 41个节点（剔除8025）
    logger.info(f"加载 {len(node_ids)} 个节点的 MinMax 参数")

    # 2. 检查新口径数据目录
    if not NEW_DATA_DIR.exists():
        logger.error(f"新口径数据目录不存在: {NEW_DATA_DIR}")
        sys.exit(1)

    # 3. 加载训练、验证、测试数据加载器
    logger.info("加载训练数据...")
    train_loaders = load_node_loaders(node_ids, NEW_DATA_DIR, node_minmax, 'train', args.batch_size, shuffle=True)
    logger.info("加载验证数据...")
    val_loaders = load_node_loaders(node_ids, NEW_DATA_DIR, node_minmax, 'val', args.batch_size, shuffle=False)
    logger.info("加载测试数据...")
    test_loaders = load_node_loaders(node_ids, NEW_DATA_DIR, node_minmax, 'test', args.batch_size, shuffle=False)

    if not train_loaders or not val_loaders or not test_loaders:
        logger.error("数据加载失败，请检查数据完整性")
        sys.exit(1)

    # 4. 创建训练器并开始训练
    trainer = FederatedTrainer(args)
    model, test_smape = trainer.train(
        train_loaders, val_loaders, test_loaders, node_minmax,
        args.rounds, args.mu
    )

    logger.info("=" * 60)
    logger.info(f"微调完成！测试集真实 sMAPE = {test_smape:.2f}%")
    logger.info(f"最佳模型保存至: {args.output_model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()