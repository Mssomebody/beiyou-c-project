#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对数变换微调（针对分布漂移节点）
- 对目标值取 log1p，压缩动态范围
- 使用标准 MSE 损失，评估时还原到原始空间计算 sMAPE
- 输出模型、损失记录、汇总统计、曲线图
"""

import sys
import torch
import torch.nn as nn
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "versions" / "v2_holiday_sector"))
from train_federated_pretrain import FederatedTrainer, load_node_loaders, MinMaxBarcelonaDataset
from torch.utils.data import DataLoader, Dataset

# ========== 配置 ==========
TARGET_NODES = [8006, 8029, 8036]
LR = 5e-4                    # 对数空间下学习率可以稍大
MAX_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 64
DEVICE = 'cpu'
SOURCE_MODEL_DIR = PROJECT_ROOT / "results/finetune/models"
OUTPUT_BASE = PROJECT_ROOT / "results/finetune_log"
MODEL_DIR = OUTPUT_BASE / "models"
LOSS_DIR = OUTPUT_BASE / "losses"
SUMMARY_FILE = OUTPUT_BASE / "summary.csv"
STATS_FILE = OUTPUT_BASE / "stats.json"
LOG_FILE = OUTPUT_BASE / "finetune.log"
CURVE_PNG = OUTPUT_BASE / "curves.png"

for d in [MODEL_DIR, LOSS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LogMinMaxDataset(Dataset):
    """将原始 MinMaxBarcelonaDataset 包装，对目标值取 log1p"""
    def __init__(self, original_dataset):
        self.original = original_dataset
    def __len__(self):
        return len(self.original)
    def __getitem__(self, idx):
        x, y = self.original[idx]
        # y 已经是归一化到 [0,1] 的原始能耗值
        # 需要先反归一化到原始空间，再取 log1p，然后再归一化到 [0,1]？
        # 更简单：直接对原始能耗取 log1p，然后重新归一化。
        # 但这里我们直接在加载数据时修改原始 dataset 的 target。
        # 由于 MinMaxBarcelonaDataset 内部已经归一化，我们只能获取 y_norm。
        # 所以我们在这里将 y_norm 反归一化 -> 取 log1p -> 重新归一化。
        # 需要知道该节点的 data_min, data_max。
        # 为了方便，我们在外部循环中传入 data_min, data_max。
        pass

# 更简单的做法：不包装，而是重新实现一个 LogMinMaxBarcelonaDataset，但为了不破坏结构，
# 我们直接在训练循环中，对从原始 dataset 取出的 y 进行变换。
# 然而原始 load_node_loaders 返回的 DataLoader 中的 y 已经是归一化的 tensor。
# 我们可以在训练循环中，将 y 反归一化 -> log1p -> 重新归一化。但这样每 batch 都要计算，效率低。
# 更好的方法：预先生成对数变换后的 pickle 文件。但您要求不修改原数据。

# 考虑到时间，我提供一个更实用的方案：直接修改原始数据集类，增加一个参数 use_log。
# 但您要求不修改核心逻辑，所以我将提供一个独立的训练函数，手动构建 DataLoader。

def create_log_loaders(node_id, data_dir, node_minmax, batch_size, shuffle):
    """手动创建 DataLoader，对目标值进行对数变换"""
    from train_federated_pretrain import MinMaxBarcelonaDataset
    data_min, data_max = node_minmax[node_id]
    train_dataset = MinMaxBarcelonaDataset(data_dir / f"node_{node_id}" / "train.pkl", node_id, node_minmax)
    val_dataset = MinMaxBarcelonaDataset(data_dir / f"node_{node_id}" / "val.pkl", node_id, node_minmax)
    
    # 包装数据集，对 y 进行 log1p 变换（在原始空间操作）
    class LogDataset(Dataset):
        def __init__(self, ds, data_min, data_max):
            self.ds = ds
            self.data_min = data_min
            self.data_max = data_max
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            x, y_norm = self.ds[idx]
            # 反归一化到原始空间
            y_real = y_norm * (self.data_max - self.data_min) + self.data_min
            # 取对数
            y_log = torch.log1p(y_real)
            # 重新归一化到 [0,1]（使用对数空间下的 min/max，这里简单线性缩放到 [0,1]）
            # 为了稳定，我们使用全局对数空间下的 min/max（预先计算）
            # 这里简化：不对 y_log 做额外归一化，因为模型输出也可以直接预测 log 值。
            # 但模型输出层没有约束，可以输出任意实数。我们让模型直接预测 log(y+1)。
            # 因此不需要再归一化。评估时，将预测值 expm1 还原。
            return x, y_log
    train_log = LogDataset(train_dataset, data_min, data_max)
    val_log = LogDataset(val_dataset, data_min, data_max)
    train_loader = DataLoader(train_log, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_log, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def log_smape(model, val_loader, data_min, data_max, device):
    """计算对数空间预测的 sMAPE（还原到原始空间）"""
    model.eval()
    all_preds_log, all_targets_log = [], []
    with torch.no_grad():
        for x, y_log in val_loader:
            x = x.to(device)
            pred_log = model(x).cpu().numpy()
            target_log = y_log.cpu().numpy()
            all_preds_log.append(pred_log)
            all_targets_log.append(target_log)
    all_preds_log = np.concatenate(all_preds_log)
    all_targets_log = np.concatenate(all_targets_log)
    # 还原到原始空间
    pred_real = np.expm1(all_preds_log)
    target_real = np.expm1(all_targets_log)
    denominator = (np.abs(target_real) + np.abs(pred_real)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(np.abs(target_real - pred_real) / denominator) * 100
    return smape

def finetune_log(node_id):
    logger.info(f"开始对数变换微调节点 {node_id}")
    start_time = time.time()

    with open(PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl", 'rb') as f:
        node_minmax = pickle.load(f)
    data_min, data_max = node_minmax[node_id]

    data_dir = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"
    train_loader, val_loader = create_log_loaders(node_id, data_dir, node_minmax, BATCH_SIZE, shuffle=True)

    # 加载原始微调模型（但模型需要调整输出层？原始模型输出维度是4，预测的是原始能耗，现在要预测 log 值，但网络结构不变，只是含义变了）
    src_model = SOURCE_MODEL_DIR / f"node_{node_id}.pth"
    if not src_model.exists():
        logger.error(f"源模型不存在: {src_model}")
        return None

    # 创建模型（结构相同）
    class Config:
        def __init__(self):
            self.device = DEVICE
            self.lr = LR
            self.local_epochs = MAX_EPOCHS
            self.rounds = 20
            self.output_model = 'dummy.pth'
    cfg = Config()
    trainer = FederatedTrainer(cfg)
    model = trainer.create_model()
    model.load_state_dict(torch.load(src_model, map_location='cpu'))
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_smape = float('inf')
    best_state = None
    patience_counter = 0
    loss_records = []

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for x, y_log in train_loader:
            x, y_log = x.to(DEVICE), y_log.to(DEVICE)
            optimizer.zero_grad()
            pred_log = model(x)
            loss = criterion(pred_log, y_log)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches

        # 评估 sMAPE（原始空间）
        val_smape = log_smape(model, val_loader, data_min, data_max, DEVICE)
        loss_records.append((epoch, avg_loss, val_smape))
        logger.info(f"节点 {node_id} Epoch {epoch:2d}: train_loss={avg_loss:.6f}, val_smape={val_smape:.2f}%")

        if val_smape < best_smape - 1e-4:
            best_smape = val_smape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"节点 {node_id} 早停于第 {epoch} 轮，最佳 sMAPE={best_smape:.2f}%")
                break

    if best_state is None:
        best_state = model.state_dict()
        best_smape = loss_records[-1][2]

    torch.save(best_state, MODEL_DIR / f"node_{node_id}.pth")
    loss_df = pd.DataFrame(loss_records, columns=["epoch", "train_loss", "val_smape"])
    loss_df.to_csv(LOSS_DIR / f"node_{node_id}_loss.csv", index=False)

    elapsed = time.time() - start_time
    logger.info(f"节点 {node_id} 完成，最佳 sMAPE={best_smape:.2f}%，耗时 {elapsed:.1f}s")
    return {'node_id': node_id, 'best_smape': best_smape, 'loss_records': loss_records}

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("对数变换微调（针对分布漂移节点）")
    logger.info(f"节点: {TARGET_NODES}")
    logger.info(f"学习率: {LR}, 最大轮数: {MAX_EPOCHS}, 早停耐心: {PATIENCE}")
    logger.info(f"输出目录: {OUTPUT_BASE}")
    logger.info("="*60)

    results = []
    for node_id in TARGET_NODES:
        res = finetune_log(node_id)
        if res:
            results.append(res)

    if not results:
        logger.error("没有成功微调任何节点")
        sys.exit(1)

    summary_df = pd.DataFrame([(r['node_id'], r['best_smape']) for r in results], columns=['node_id', 'best_val_smape'])
    summary_df.to_csv(SUMMARY_FILE, index=False)

    all_smapes = summary_df['best_val_smape'].tolist()
    stats = {
        "mean": float(np.mean(all_smapes)),
        "median": float(np.median(all_smapes)),
        "std": float(np.std(all_smapes)),
        "min": float(np.min(all_smapes)),
        "max": float(np.max(all_smapes)),
        "num_nodes": len(all_smapes),
        "timestamp": datetime.now().isoformat()
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"统计结果: 平均 sMAPE = {stats['mean']:.2f}%")

    plt.figure(figsize=(12, 6))
    for r in results:
        loss_records = r['loss_records']
        if loss_records:
            epochs = [e for e, _, _ in loss_records]
            losses = [l for _, l, _ in loss_records]
            plt.plot(epochs, losses, label=f"Node {r['node_id']}")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Log-scale Fine-tuning Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(CURVE_PNG, dpi=150)
    plt.close()

    logger.info("="*60)
    logger.info("对数变换微调完成！")
    logger.info(f"汇总表: {SUMMARY_FILE}")
    logger.info(f"统计: {STATS_FILE}")
    logger.info(f"模型: {MODEL_DIR}")
    logger.info(f"损失记录: {LOSS_DIR}")
    logger.info("="*60)