#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二次微调（全功能版，支持并行、早停、完整输出）
- 可指定节点列表或按 sMAPE 阈值筛选
- 可调整学习率、最大轮数、早停耐心、并行进程数
- 输出每个节点的最佳模型、损失记录、汇总统计（CSV/JSON/曲线图）
- 不覆盖原始微调结果，输出到独立目录
"""

import sys
import torch
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime
import matplotlib.pyplot as plt
import json
import time

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "versions" / "v2_holiday_sector"))
from train_federated_pretrain import FederatedTrainer, load_node_loaders

# ========== 默认配置（可通过命令行参数覆盖） ==========
DEFAULT_MAX_EPOCHS = 20
DEFAULT_LR = 5e-5
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE = 'cpu'
DEFAULT_PATIENCE = 5
DEFAULT_NUM_PROCESSES = min(8, cpu_count())
DEFAULT_THRESHOLD = 50.0          # 只对 sMAPE > 此值的节点二次微调
DEFAULT_SOURCE_DIR = "results/finetune/models"
DEFAULT_OUTPUT_BASE = "results/finetune_secondary"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="二次微调（全功能版）")
    parser.add_argument('--nodes', type=int, nargs='+', help='指定节点ID列表，若不指定则按阈值筛选')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'sMAPE阈值，只处理大于该值的节点（默认{DEFAULT_THRESHOLD}）')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help=f'学习率（默认{DEFAULT_LR}）')
    parser.add_argument('--epochs', type=int, default=DEFAULT_MAX_EPOCHS, help=f'最大轮数（默认{DEFAULT_MAX_EPOCHS}）')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, help=f'早停耐心（默认{DEFAULT_PATIENCE}）')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help=f'批次大小（默认{DEFAULT_BATCH_SIZE}）')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='设备（cpu/cuda）')
    parser.add_argument('--processes', type=int, default=DEFAULT_NUM_PROCESSES, help=f'并行进程数（默认{DEFAULT_NUM_PROCESSES}）')
    parser.add_argument('--source_dir', type=str, default=DEFAULT_SOURCE_DIR, help='原始微调模型目录')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUTPUT_BASE, help='输出根目录')
    return parser.parse_args()

args = parse_args()

# 输出目录
OUTPUT_BASE = Path(args.out_dir)
MODEL_DIR = OUTPUT_BASE / "models"
LOSS_DIR = OUTPUT_BASE / "losses"
SUMMARY_FILE = OUTPUT_BASE / "summary.csv"
STATS_FILE = OUTPUT_BASE / "stats.json"
LOG_FILE = OUTPUT_BASE / "finetune.log"
CURVE_PNG = OUTPUT_BASE / "curves.png"

for d in [MODEL_DIR, LOSS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.device = args.device
        self.lr = args.lr
        self.local_epochs = args.epochs
        self.rounds = 20
        self.output_model = str(OUTPUT_BASE / "dummy.pth")

def finetune_node(node_id):
    """对单个节点进行二次微调（在每个子进程中独立运行）"""
    proc_logger = logging.getLogger(f"node_{node_id}")
    proc_logger.setLevel(logging.INFO)
    if not proc_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        proc_logger.addHandler(ch)

    try:
        proc_logger.info(f"开始二次微调")
        start_time = time.time()

        # 加载资源
        with open(PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl", 'rb') as f:
            node_minmax = pickle.load(f)
        source_model_path = Path(args.source_dir) / f"node_{node_id}.pth"
        if not source_model_path.exists():
            proc_logger.warning(f"源模型 {source_model_path} 不存在，跳过")
            return node_id, None, None, None

        # 创建训练器
        cfg = Config()
        trainer = FederatedTrainer(cfg)

        # 加载数据
        data_dir = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"
        train_loaders = load_node_loaders([node_id], data_dir, node_minmax, 'train', batch_size=args.batch_size, shuffle=True)
        val_loaders = load_node_loaders([node_id], data_dir, node_minmax, 'val', batch_size=args.batch_size, shuffle=False)
        if node_id not in train_loaders or node_id not in val_loaders:
            proc_logger.warning(f"节点 {node_id} 数据缺失，跳过")
            return node_id, None, None, None

        # 加载原始模型
        model = trainer.create_model()
        model.load_state_dict(torch.load(source_model_path, map_location='cpu'))
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        best_smape = float('inf')
        best_state = None
        patience_counter = 0
        loss_records = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0
            for x, y in train_loaders[node_id]:
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            model.eval()
            val_smape = trainer.evaluate_smape(model, {node_id: val_loaders[node_id]}, node_minmax, real_space=True)
            loss_records.append((epoch, avg_loss, val_smape))
            proc_logger.debug(f"Epoch {epoch:2d}: train_loss={avg_loss:.6f}, val_smape={val_smape:.2f}%")

            if val_smape < best_smape - 1e-4:
                best_smape = val_smape
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    proc_logger.info(f"早停于第 {epoch} 轮，最佳 val_smape = {best_smape:.2f}%")
                    break

        if best_state is None:
            best_state = model.state_dict()
            best_smape = loss_records[-1][2]

        # 保存模型和损失记录
        torch.save(best_state, MODEL_DIR / f"node_{node_id}.pth")
        loss_df = pd.DataFrame(loss_records, columns=["epoch", "train_loss", "val_smape"])
        loss_df.to_csv(LOSS_DIR / f"node_{node_id}_loss.csv", index=False)

        elapsed = time.time() - start_time
        proc_logger.info(f"完成，最佳 val_smape = {best_smape:.2f}%，耗时 {elapsed:.1f} 秒")
        return node_id, best_smape, loss_records, elapsed

    except Exception as e:
        proc_logger.error(f"二次微调失败: {e}", exc_info=True)
        return node_id, None, None, None

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("二次微调（全功能版）")
    logger.info(f"配置: epochs={args.epochs}, lr={args.lr}, patience={args.patience}, batch_size={args.batch_size}")
    logger.info(f"设备: {args.device}, 并行进程数: {args.processes}")
    logger.info(f"源模型目录: {args.source_dir}")
    logger.info(f"输出目录: {OUTPUT_BASE}")
    logger.info("="*60)

    # 读取原始汇总表，获取节点列表
    orig_summary = Path("results/finetune/summary.csv")
    if not orig_summary.exists():
        logger.error(f"原始汇总文件不存在: {orig_summary}")
        sys.exit(1)
    df_orig = pd.read_csv(orig_summary)

    if args.nodes:
        node_ids = args.nodes
        logger.info(f"指定节点: {node_ids}")
    else:
        node_ids = df_orig[df_orig['best_val_smape'] > args.threshold]['node_id'].tolist()
        logger.info(f"按阈值 {args.threshold} 筛选出 {len(node_ids)} 个节点: {node_ids[:10]}...")

    if not node_ids:
        logger.info("没有需要二次微调的节点，退出")
        sys.exit(0)

    # 并行执行
    logger.info(f"启动 {args.processes} 个进程并行微调...")
    start_total = time.time()
    with Pool(processes=args.processes) as pool:
        results = pool.map(finetune_node, node_ids)

    # 汇总结果
    summary = []
    all_smapes = []
    for node_id, smape, _, _ in results:
        if smape is not None:
            summary.append({"node_id": node_id, "best_val_smape": smape})
            all_smapes.append(smape)

    if not all_smapes:
        logger.error("没有成功微调任何节点，退出")
        sys.exit(1)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    logger.info(f"汇总结果已保存至 {SUMMARY_FILE}")

    # 统计信息
    stats = {
        "mean": float(np.mean(all_smapes)),
        "median": float(np.median(all_smapes)),
        "std": float(np.std(all_smapes)),
        "min": float(np.min(all_smapes)),
        "max": float(np.max(all_smapes)),
        "num_nodes": len(all_smapes),
        "total_time_sec": time.time() - start_total,
        "timestamp": datetime.now().isoformat()
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"统计结果: 平均 sMAPE = {stats['mean']:.2f}%, 中位数 = {stats['median']:.2f}%")
    logger.info(f"总耗时: {stats['total_time_sec']:.1f} 秒")

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    for node_id, _, loss_records, _ in results:
        if loss_records:
            epochs = [r[0] for r in loss_records]
            losses = [r[1] for r in loss_records]
            plt.plot(epochs, losses, alpha=0.3, linewidth=0.8, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Secondary Fine-tuning Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.savefig(CURVE_PNG, dpi=150)
    plt.close()
    logger.info(f"损失曲线图已保存至 {CURVE_PNG}")

    logger.info("="*60)
    logger.info("二次微调完成！")
    logger.info(f"结果汇总: {SUMMARY_FILE}")
    logger.info(f"统计信息: {STATS_FILE}")
    logger.info(f"各节点模型: {MODEL_DIR}")
    logger.info(f"各节点损失记录: {LOSS_DIR}")
    logger.info("="*60)