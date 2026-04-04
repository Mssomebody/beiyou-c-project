#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滑动窗口可学习时段训练（五星级）

- 使用原始6小时粒度数据（28步输入，4步输出）
- 模型在输入特征前加入可学习时段权重（每个时间步的 hour_code 对应的可学习参数）
- 联邦学习（FedAvg + FedProx），41节点
- 支持早停、学习率调度、梯度裁剪
- 输出最终模型到 decision/models/model_learnable_hour.pth
- 可选择性加载预训练权重（可选），但为了公平对比，建议从头训练

作者: FedGreen-C 项目组
版本: 2.0
日期: 2026-03-31
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
MINMAX_FILE = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
OLD_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"

# 输出
OUTPUT_MODEL = PROJECT_ROOT / "decision" / "models" / "model_learnable_hour.pth"
LOG_FILE = PROJECT_ROOT / "results" / "two_stage" / "learnable_hour_log.txt"

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
# 数据集：返回原始特征 + hour_code（用于可学习时段）
# ============================================================
class MinMaxBarcelonaDataset(Dataset):
    """
    滑动窗口数据集，返回输入特征（不包含时段编码）和 hour_code 序列
    """
    def __init__(self, data_path: Path, node_id: int, node_minmax: Dict[int, Tuple[float, float]],
                 window_size=28, predict_size=4):
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

        # 原始特征（7维）
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

        x = torch.cat([x_energy, x_sector, x_holiday, x_weekend], dim=1)  # (window_size, 7)

        # hour_code 序列 (window_size,)
        hour_seq = self.hour_code[start:start+self.window_size]
        hour_seq = torch.LongTensor(hour_seq)

        # 目标
        y = self.energy[start+self.window_size:start+self.window_size+self.predict_size]
        y = (y - self.data_min) / (self.data_max - self.data_min + 1e-8)
        y = torch.FloatTensor(y)

        return x, y, hour_seq


# ============================================================
# 可学习时段模型：在 LSTM 前对输入特征按时段加权
# ============================================================
class LearnableHourLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        # 时段权重参数：每个小时一个可学习标量（共4个时段）
        self.hour_weights = nn.Parameter(torch.ones(4))
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hour_seq):
        """
        x: (batch, seq_len, input_dim)
        hour_seq: (batch, seq_len) 每个时间步的 hour_code (0~3)
        """
        # 获取每个时间步的权重
        # hour_weights 是 (4,)，通过索引得到 (batch, seq_len)
        weights = self.hour_weights[hour_seq]  # (batch, seq_len)
        weights = weights.unsqueeze(-1)        # (batch, seq_len, 1)
        x_weighted = x * weights
        out, _ = self.lstm(x_weighted)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 联邦训练器（支持可学习时段模型）
# ============================================================
class FederatedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.best_val_smape = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def create_model(self):
        return LearnableHourLSTM(
            input_dim=7,
            hidden_dim=64,
            num_layers=2,
            output_dim=4,
            dropout=0.2
        )

    def train_round(self, model, client_loaders, mu):
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
                for x, y, hour_seq in loader:
                    x, y, hour_seq = x.to(self.device), y.to(self.device), hour_seq.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(x, hour_seq)
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
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for node_id, loader in loaders.items():
                data_min, data_max = node_minmax[node_id]
                for x, y, hour_seq in loader:
                    x, hour_seq = x.to(self.device), hour_seq.to(self.device)
                    pred_norm = model(x, hour_seq).cpu().numpy()
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
        model = self.create_model().to(self.device)

        # 可选：加载预训练权重（用于对比时，可让可学习时段从相同初始点开始）
        if self.config.pretrain_path and Path(self.config.pretrain_path).exists():
            state_dict = torch.load(self.config.pretrain_path, map_location=self.device)
            # 由于模型结构不同，只能加载部分层（LSTM 和 fc），hour_weights 随机初始化
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'hour_weights' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"加载预训练模型部分权重: {self.config.pretrain_path}")
        else:
            logger.info("从头开始训练")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
)

        for round_num in range(1, self.config.rounds + 1):
            model, avg_loss = self.train_round(model, client_train_loaders, mu)
            val_smape = self.evaluate_smape(model, client_val_loaders, node_minmax, real_space=True)
            scheduler.step(val_smape)

            logger.info(f"Round {round_num:3d}: train_loss = {avg_loss:.6f}, val_smape = {val_smape:.2f}%")

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

        model.load_state_dict(self.best_model_state)
        logger.info(f"训练结束，最佳验证 sMAPE = {self.best_val_smape:.2f}%")

        test_smape = self.evaluate_smape(model, client_test_loaders, node_minmax, real_space=True)
        logger.info(f"测试集真实 sMAPE = {test_smape:.2f}%")

        torch.save(model.state_dict(), self.config.output_model)
        logger.info(f"最佳模型已保存至 {self.config.output_model}")

        return model, test_smape


# ============================================================
# 数据加载函数
# ============================================================
def load_node_loaders(node_ids: List[int], data_dir: Path, node_minmax: Dict,
                      split: str, batch_size: int = 64, shuffle: bool = True) -> Dict[int, DataLoader]:
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
    parser = argparse.ArgumentParser(description="滑动窗口可学习时段训练")
    parser.add_argument('--rounds', type=int, default=50, help='通信轮数')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--mu', type=float, default=0.05, help='FedProx 系数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--pretrain_path', type=str, default=None,
                        help='预训练模型路径（可选，用于加载 LSTM 部分权重）')
    parser.add_argument('--output_model', type=str, default=str(OUTPUT_MODEL), help='输出模型路径')
    args = parser.parse_args()

    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("滑动窗口可学习时段训练启动")
    logger.info(f"设备: {args.device}")
    logger.info(f"通信轮数: {args.rounds}, 本地轮数: {args.local_epochs}")
    logger.info(f"学习率: {args.lr}, FedProx μ: {args.mu}")
    logger.info("=" * 60)

    # 1. 加载节点 MinMax 参数
    if not MINMAX_FILE.exists():
        logger.error(f"MinMax 参数文件不存在: {MINMAX_FILE}")
        sys.exit(1)
    with open(MINMAX_FILE, 'rb') as f:
        node_minmax = pickle.load(f)
    node_ids = list(node_minmax.keys())  # 41个节点
    logger.info(f"加载 {len(node_ids)} 个节点的 MinMax 参数")

    # 2. 加载旧口径数据
    if not OLD_DATA_DIR.exists():
        logger.error(f"旧口径数据目录不存在: {OLD_DATA_DIR}")
        sys.exit(1)

    logger.info("加载训练数据...")
    train_loaders = load_node_loaders(node_ids, OLD_DATA_DIR, node_minmax, 'train', args.batch_size, shuffle=True)
    logger.info("加载验证数据...")
    val_loaders = load_node_loaders(node_ids, OLD_DATA_DIR, node_minmax, 'val', args.batch_size, shuffle=False)
    logger.info("加载测试数据...")
    test_loaders = load_node_loaders(node_ids, OLD_DATA_DIR, node_minmax, 'test', args.batch_size, shuffle=False)

    if not train_loaders or not val_loaders or not test_loaders:
        logger.error("数据加载失败，请检查数据完整性")
        sys.exit(1)

    # 3. 创建训练器并开始训练
    trainer = FederatedTrainer(args)
    model, test_smape = trainer.train(
        train_loaders, val_loaders, test_loaders, node_minmax,
        args.rounds, args.mu
    )

    logger.info("=" * 60)
    logger.info(f"训练完成！测试集真实 sMAPE = {test_smape:.2f}%")
    logger.info(f"最佳模型保存至: {args.output_model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()