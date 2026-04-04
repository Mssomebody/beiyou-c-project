#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加权损失微调（针对分布漂移节点 8006, 8029, 8036）
- 使用加权 MSELoss，对高能耗样本赋予更高权重
- 输出每个节点的最佳模型、损失记录、汇总统计、曲线图、日志
- 不修改原有数据，不破坏项目结构
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
from train_federated_pretrain import FederatedTrainer, load_node_loaders

# ========== 配置 ==========
TARGET_NODES = [8006, 8029, 8036]          # 需要加权微调的节点
LR = 5e-5                                   # 学习率
MAX_EPOCHS = 30                             # 最大轮数
PATIENCE = 10                               # 早停耐心
BATCH_SIZE = 64
DEVICE = 'cpu'
WEIGHT_THRESHOLD = 180000                   # 阈值：原始能耗 > 此值则加权
WEIGHT_VALUE = 5.0                          # 权重倍数
SOURCE_MODEL_DIR = PROJECT_ROOT / "results/finetune/models"   # 原始微调模型
OUTPUT_BASE = PROJECT_ROOT / "results/finetune_weighted"
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

# 自定义加权损失（在原始空间计算）
class WeightedMSELoss(nn.Module):
    def __init__(self, data_min, data_max, threshold=WEIGHT_THRESHOLD, weight=WEIGHT_VALUE):
        super().__init__()
        self.data_min = data_min
        self.data_max = data_max
        self.threshold = threshold
        self.weight = weight

    def forward(self, pred_norm, target_norm):
        # 反归一化到原始能耗空间
        pred = pred_norm * (self.data_max - self.data_min) + self.data_min
        target = target_norm * (self.data_max - self.data_min) + self.data_min
        mse = (pred - target) ** 2
        weights = torch.where(target > self.threshold, self.weight, 1.0)
        return (weights * mse).mean()

class Config:
    def __init__(self):
        self.device = DEVICE
        self.lr = LR
        self.local_epochs = MAX_EPOCHS
        self.rounds = 20
        self.output_model = 'dummy.pth'

def weighted_finetune(node_id):
    logger.info(f"开始加权微调节点 {node_id}")
    start_time = time.time()

    # 加载归一化参数
    with open(PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl", 'rb') as f:
        node_minmax = pickle.load(f)
    data_min, data_max = node_minmax[node_id]

    # 加载数据
    data_dir = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"
    train_loaders = load_node_loaders([node_id], data_dir, node_minmax, 'train', batch_size=BATCH_SIZE, shuffle=True)
    val_loaders = load_node_loaders([node_id], data_dir, node_minmax, 'val', batch_size=BATCH_SIZE, shuffle=False)
    if node_id not in train_loaders or node_id not in val_loaders:
        logger.error(f"节点 {node_id} 数据缺失")
        return None

    # 加载原始微调模型
    src_model = SOURCE_MODEL_DIR / f"node_{node_id}.pth"
    if not src_model.exists():
        logger.error(f"源模型不存在: {src_model}")
        return None

    cfg = Config()
    trainer = FederatedTrainer(cfg)
    model = trainer.create_model()
    model.load_state_dict(torch.load(src_model, map_location='cpu'))
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = WeightedMSELoss(data_min, data_max, threshold=WEIGHT_THRESHOLD, weight=WEIGHT_VALUE)

    best_smape = float('inf')
    best_state = None
    patience_counter = 0
    loss_records = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for x, y in train_loaders[node_id]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches

        model.eval()
        val_smape = trainer.evaluate_smape(model, {node_id: val_loaders[node_id]}, node_minmax, real_space=True)
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

    # 保存
    torch.save(best_state, MODEL_DIR / f"node_{node_id}.pth")
    loss_df = pd.DataFrame(loss_records, columns=["epoch", "train_loss", "val_smape"])
    loss_df.to_csv(LOSS_DIR / f"node_{node_id}_loss.csv", index=False)

    elapsed = time.time() - start_time
    logger.info(f"节点 {node_id} 完成，最佳 sMAPE={best_smape:.2f}%，耗时 {elapsed:.1f}s")
    return {'node_id': node_id, 'best_smape': best_smape, 'loss_records': loss_records}

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("加权损失微调（针对分布漂移节点）")
    logger.info(f"节点: {TARGET_NODES}")
    logger.info(f"阈值: {WEIGHT_THRESHOLD}, 权重: {WEIGHT_VALUE}")
    logger.info(f"学习率: {LR}, 最大轮数: {MAX_EPOCHS}, 早停耐心: {PATIENCE}")
    logger.info(f"输出目录: {OUTPUT_BASE}")
    logger.info("="*60)

    results = []
    for node_id in TARGET_NODES:
        res = weighted_finetune(node_id)
        if res:
            results.append(res)

    if not results:
        logger.error("没有成功微调任何节点")
        sys.exit(1)

    # 汇总表
    summary_df = pd.DataFrame([(r['node_id'], r['best_smape']) for r in results], columns=['node_id', 'best_val_smape'])
    summary_df.to_csv(SUMMARY_FILE, index=False)

    # 统计
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

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    for r in results:
        loss_records = r['loss_records']
        if loss_records:
            epochs = [e for e, _, _ in loss_records]
            losses = [l for _, l, _ in loss_records]
            plt.plot(epochs, losses, label=f"Node {r['node_id']}")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Weighted Fine-tuning Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(CURVE_PNG, dpi=150)
    plt.close()

    logger.info("="*60)
    logger.info("加权微调完成！")
    logger.info(f"汇总表: {SUMMARY_FILE}")
    logger.info(f"统计: {STATS_FILE}")
    logger.info(f"模型: {MODEL_DIR}")
    logger.info(f"损失记录: {LOSS_DIR}")
    logger.info(f"曲线图: {CURVE_PNG}")
    logger.info("="*60)