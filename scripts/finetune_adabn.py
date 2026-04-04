#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaBN 微调（针对分布漂移节点 8006, 8029, 8036）
- 在 LSTM 输出后添加 BatchNorm1d 层
- 每个 epoch 后用验证集更新 BN 统计量（AdaBN）
- 输出完整支撑材料
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
from train_federated_pretrain import FederatedTrainer, load_node_loaders, LSTMPredictor

# ========== 配置 ==========
TARGET_NODES = [8006, 8029, 8036]
LR = 5e-5                     # 学习率
MAX_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 64
DEVICE = 'cpu'
WEIGHT_DECAY = 1e-5           # 轻微正则化

# 路径
SOURCE_MODEL_DIR = PROJECT_ROOT / "results/finetune/models"   # 第一次微调模型
OUTPUT_BASE = PROJECT_ROOT / "results/finetune_adabn"
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

# ========== 带 BN 的模型（继承原模型，添加 BN 层）==========
class LSTMPredictorWithBN(LSTMPredictor):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)
        # 在 fc 之前添加 BN 层
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]          # (batch, hidden_dim)
        last_out = self.bn(last_out)      # AdaBN 会更新统计量
        return self.fc(last_out)

def load_pretrained_weights(model, src_path, device):
    """加载原始微调模型权重（忽略 BN 层，因为原模型没有）"""
    src_state = torch.load(src_path, map_location=device)
    model_state = model.state_dict()
    # 只加载匹配的键（排除 bn 层）
    pretrained_dict = {k: v for k, v in src_state.items() if k in model_state and 'bn' not in k}
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    logger.info(f"加载了 {len(pretrained_dict)} 个预训练参数，跳过 BN 层")
    return model

def adabn_update_bn(model, val_loader, device):
    """用验证集数据更新 BN 统计量（不计算梯度）"""
    model.eval()
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            _ = model(x)   # 前向传播，自动更新 BN 的 running_mean/var

def finetune_adabn(node_id):
    logger.info(f"开始 AdaBN 微调节点 {node_id}")
    start_time = time.time()

    # 加载节点归一化参数
    with open(PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl", 'rb') as f:
        node_minmax = pickle.load(f)

    # 加载数据
    data_dir = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"
    train_loaders = load_node_loaders([node_id], data_dir, node_minmax, 'train', batch_size=BATCH_SIZE, shuffle=True)
    val_loaders = load_node_loaders([node_id], data_dir, node_minmax, 'val', batch_size=BATCH_SIZE, shuffle=False)
    if node_id not in train_loaders or node_id not in val_loaders:
        logger.error(f"节点 {node_id} 数据缺失")
        return None

    # 创建模型并加载预训练权重
    model = LSTMPredictorWithBN().to(DEVICE)
    src_model_path = SOURCE_MODEL_DIR / f"node_{node_id}.pth"
    if not src_model_path.exists():
        logger.error(f"源模型不存在: {src_model_path}")
        return None
    model = load_pretrained_weights(model, src_model_path, DEVICE)

    # 优化器（BN 层参数也参与训练）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_smape = float('inf')
    best_state = None
    patience_counter = 0
    loss_records = []

    # 获取验证集 DataLoader（用于 AdaBN 更新）
    val_loader = val_loaders[node_id]

    for epoch in range(1, MAX_EPOCHS + 1):
        # 训练一个 epoch
        model.train()
        total_loss = 0.0
        num_batches = 0
        for x, y in train_loaders[node_id]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_train_loss = total_loss / num_batches

        # AdaBN：用验证集更新 BN 统计量
        adabn_update_bn(model, val_loader, DEVICE)

        # 评估验证 sMAPE（使用原始评估函数）
        # 注意：需要临时恢复 model 的 eval 模式，但 BN 已经用验证集统计量，无需额外操作
        model.eval()
        # 由于 evaluate_smape 内部会调用 model.eval() 并前向传播，BN 会使用已经更新的 running stats
        # 我们直接使用 trainer 的评估函数
        class DummyConfig:
            def __init__(self):
                self.device = DEVICE
        trainer = FederatedTrainer(DummyConfig())
        val_smape = trainer.evaluate_smape(model, {node_id: val_loader}, node_minmax, real_space=True)
        loss_records.append((epoch, avg_train_loss, val_smape))
        logger.info(f"节点 {node_id} Epoch {epoch:2d}: train_loss={avg_train_loss:.6f}, val_smape={val_smape:.2f}%")

        # 早停
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

    # 保存最佳模型
    torch.save(best_state, MODEL_DIR / f"node_{node_id}.pth")
    loss_df = pd.DataFrame(loss_records, columns=["epoch", "train_loss", "val_smape"])
    loss_df.to_csv(LOSS_DIR / f"node_{node_id}_loss.csv", index=False)

    elapsed = time.time() - start_time
    logger.info(f"节点 {node_id} 完成，最佳 sMAPE={best_smape:.2f}%，耗时 {elapsed:.1f}s")
    return {'node_id': node_id, 'best_smape': best_smape, 'loss_records': loss_records}

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("AdaBN 微调（针对分布漂移节点）")
    logger.info(f"节点: {TARGET_NODES}")
    logger.info(f"学习率: {LR}, 最大轮数: {MAX_EPOCHS}, 早停耐心: {PATIENCE}")
    logger.info(f"输出目录: {OUTPUT_BASE}")
    logger.info("="*60)

    results = []
    for node_id in TARGET_NODES:
        res = finetune_adabn(node_id)
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
    plt.title("AdaBN Fine-tuning Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(CURVE_PNG, dpi=150)
    plt.close()

    logger.info("="*60)
    logger.info("AdaBN 微调完成！")
    logger.info(f"汇总表: {SUMMARY_FILE}")
    logger.info(f"统计: {STATS_FILE}")
    logger.info(f"模型: {MODEL_DIR}")
    logger.info(f"损失记录: {LOSS_DIR}")
    logger.info("="*60)