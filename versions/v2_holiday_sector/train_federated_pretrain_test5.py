#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦预训练 - 旧口径41节点（高精度优化版，带损失记录）
- 使用原始 Valor 列，按节点 MinMax 归一化
- 评估时反归一化计算真实 sMAPE
- 支持断点续训、最佳模型保存、学习率调度、早停
- 每轮记录 train_loss 和 val_smape 到 results/two_stage/loss_history.csv
- 超参数：hidden_dim=128, lr=0.002, mu=0.01, local_epochs=10, rounds=100
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

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 输入文件
MINMAX_FILE = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
OLD_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"

# 输出
OUTPUT_MODEL = PROJECT_ROOT / "results" / "two_stage" / "model_fed_pretrain.pth"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "two_stage" / "checkpoints"
LOG_FILE = PROJECT_ROOT / "results" / "two_stage" / "pretrain_log.txt"
LOSS_HISTORY = PROJECT_ROOT / "results" / "two_stage" / "loss_history.csv"

# 模型参数
INPUT_DIM = 7
HIDDEN_DIM = 128
NUM_LAYERS = 2
OUTPUT_DIM = 4
DROPOUT = 0.2
WINDOW_SIZE = 28
PREDICT_SIZE = 4

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 固定随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# 数据集：使用原始 Valor 和节点 MinMax 参数归一化
# ============================================================
class MinMaxBarcelonaDataset(Dataset):
    def __init__(self, data_path: Path, node_id: int, node_minmax: Dict[int, Tuple[float, float]],
                 window_size=WINDOW_SIZE, predict_size=PREDICT_SIZE):
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.data_min, self.data_max = node_minmax[node_id]
        self.energy = self.df['Valor'].values
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

        # 目标（归一化）
        y = self.energy[start+self.window_size:start+self.window_size+self.predict_size]
        y = (y - self.data_min) / (self.data_max - self.data_min + 1e-8)
        y = torch.FloatTensor(y)

        return x, y


# ============================================================
# 模型定义
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 联邦训练器
# ============================================================
class FederatedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.best_val_smape = float('inf')
        self.patience_counter = 0
        self.start_round = 1

    def create_model(self):
        return LSTMPredictor()

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
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                local_loss += epoch_loss / len(loader) if len(loader) > 0 else 0.0
            client_losses.append(local_loss / self.config.local_epochs)
            client_weights.append(local_model.state_dict())
            client_sizes.append(len(loader.dataset))

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
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
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

    def train(self, client_train_loaders, client_val_loaders, node_minmax, rounds, mu):
        model = self.create_model().to(self.device)

        # 查找最新检查点
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_files = list(CHECKPOINT_DIR.glob("round_*.pth"))
        if checkpoint_files:
            latest = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[1]))
            checkpoint = torch.load(latest, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.start_round = checkpoint['round'] + 1
            logger.info(f"从检查点 {latest.name} 恢复，继续第 {self.start_round} 轮")
        else:
            self.start_round = 1
            logger.info("从头开始训练")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 初始化损失记录文件
        LOSS_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        if self.start_round == 1:
            with open(LOSS_HISTORY, 'w') as f:
                f.write("round,train_loss,val_smape\n")

        for round_num in range(self.start_round, rounds + 1):
            model, avg_loss = self.train_round(model, client_train_loaders, mu)
            val_smape = self.evaluate_smape(model, client_val_loaders, node_minmax, real_space=True)
            logger.info(f"Round {round_num:3d}: train_loss = {avg_loss:.6f}, val_smape = {val_smape:.2f}%")

            # 记录损失
            with open(LOSS_HISTORY, 'a') as f:
                f.write(f"{round_num},{avg_loss:.6f},{val_smape:.2f}\n")

            scheduler.step(val_smape)

            # 保存检查点
            torch.save({
                'round': round_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_smape': val_smape
            }, CHECKPOINT_DIR / f"round_{round_num}.pth")

            # 早停
            if val_smape < self.best_val_smape - 0.01:
                self.best_val_smape = val_smape
                self.patience_counter = 0
                torch.save(model.state_dict(), self.config.output_model)
                logger.info(f"  -> 新的最佳模型 (val_smape={val_smape:.2f}%)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= 15:
                    logger.info(f"早停触发，停止于第 {round_num} 轮")
                    break

        # 加载最佳模型
        model.load_state_dict(torch.load(self.config.output_model))
        return model, self.best_val_smape


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
    parser = argparse.ArgumentParser(description="联邦预训练（高精度优化版）")
    parser.add_argument('--rounds', type=int, default=100, help='通信轮数')
    parser.add_argument('--local_epochs', type=int, default=10, help='本地训练轮数')
    parser.add_argument('--lr', type=float, default=0.002, help='学习率')
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx 系数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_model', type=str, default=str(OUTPUT_MODEL), help='输出模型路径')
    args = parser.parse_args()

    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("联邦预训练（高精度优化版）启动")
    logger.info(f"设备: {args.device}")
    logger.info(f"通信轮数: {args.rounds}, 本地轮数: {args.local_epochs}")
    logger.info(f"学习率: {args.lr}, FedProx μ: {args.mu}")
    logger.info(f"隐藏层维度: {HIDDEN_DIM}")
    logger.info("=" * 60)

    # 1. 加载节点 MinMax 参数
    if not MINMAX_FILE.exists():
        logger.error(f"MinMax 参数文件不存在: {MINMAX_FILE}")
        sys.exit(1)
    with open(MINMAX_FILE, 'rb') as f:
        node_minmax = pickle.load(f)
    node_ids = list(node_minmax.keys())  # 41个节点
    # 测试模式：只取前5个节点
    node_ids = node_ids[:5]
    logger.info(f"加载 {len(node_ids)} 个节点的 MinMax 参数 (测试模式: 仅前5个)")

    # 2. 加载旧口径数据
    if not OLD_DATA_DIR.exists():
        logger.error(f"旧口径数据目录不存在: {OLD_DATA_DIR}")
        sys.exit(1)

    logger.info("加载训练数据...")
    train_loaders = load_node_loaders(node_ids, OLD_DATA_DIR, node_minmax, 'train', args.batch_size, shuffle=True)
    logger.info("加载验证数据...")
    val_loaders = load_node_loaders(node_ids, OLD_DATA_DIR, node_minmax, 'val', args.batch_size, shuffle=False)

    if not train_loaders or not val_loaders:
        logger.error("数据加载失败，请检查数据完整性")
        sys.exit(1)

    # 3. 创建训练器并开始训练
    trainer = FederatedTrainer(args)
    model, best_val_smape = trainer.train(train_loaders, val_loaders, node_minmax, args.rounds, args.mu)

    logger.info("=" * 60)
    logger.info(f"预训练完成！最佳验证真实 sMAPE = {best_val_smape:.2f}%")
    logger.info(f"最佳模型保存至: {args.output_model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()