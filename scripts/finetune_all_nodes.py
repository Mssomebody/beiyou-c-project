#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移学习 + 本地微调（并行，自适应早停，保存最佳模型）
- 使用五节点成功模型作为初始权重
- 每个节点独立微调，早停轮数 patience=2，恢复最佳模型
- 自动检测GPU，并行进程数动态适配
- 保存每个节点的最佳模型、损失记录、汇总统计、全局曲线
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

# ========== 配置参数 ==========
MAX_EPOCHS = 20             # 最大微调轮数（早停会提前终止）
LR = 0.0001                 # 微调学习率（与两阶段成功经验一致）
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATIENCE = 2                # 验证sMAPE连续多少轮不下降则停止
NUM_PROCESSES = min(8, cpu_count())   # 并行进程数（不超过CPU核心数）
PRETRAINED_MODEL = PROJECT_ROOT / "backup_models" / "medium_test_v2_model.pth"
MINMAX_FILE = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"

# 输出目录
OUTPUT_BASE = PROJECT_ROOT / "results" / "finetune"
MODEL_DIR = OUTPUT_BASE / "models"
LOSS_DIR = OUTPUT_BASE / "losses"
SUMMARY_FILE = OUTPUT_BASE / "summary.csv"
STATS_FILE = OUTPUT_BASE / "stats.json"
LOG_FILE = OUTPUT_BASE / "finetune.log"
CURVE_PNG = OUTPUT_BASE / "curves.png"

# 创建输出目录
for d in [MODEL_DIR, LOSS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 配置根日志（主进程）
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
    """模拟参数配置，用于初始化 FederatedTrainer"""
    def __init__(self):
        self.device = DEVICE
        self.lr = LR
        self.local_epochs = MAX_EPOCHS
        self.rounds = 20
        self.output_model = str(OUTPUT_BASE / "dummy.pth")

def finetune_node(node_id):
    """对单个节点进行微调（在每个子进程中独立运行）"""
    # 每个子进程独立配置日志（避免多进程写同一文件）
    proc_logger = logging.getLogger(f"node_{node_id}")
    proc_logger.setLevel(logging.INFO)
    # 只输出到控制台，不写文件（主日志已足够）
    if not proc_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        proc_logger.addHandler(ch)
    
    try:
        proc_logger.info(f"开始微调")
        start_time = time.time()
        
        # 子进程独立加载必要资源
        with open(MINMAX_FILE, 'rb') as f:
            node_minmax = pickle.load(f)
        pretrained_state = torch.load(PRETRAINED_MODEL, map_location='cpu')
        
        # 创建训练器实例
        args = Config()
        trainer = FederatedTrainer(args)
        
        # 加载该节点的训练和验证数据
        train_loaders = load_node_loaders([node_id], DATA_DIR, node_minmax, 'train', batch_size=BATCH_SIZE, shuffle=True)
        val_loaders = load_node_loaders([node_id], DATA_DIR, node_minmax, 'val', batch_size=BATCH_SIZE, shuffle=False)
        if node_id not in train_loaders or node_id not in val_loaders:
            proc_logger.warning(f"数据缺失，跳过")
            return node_id, None, None, None
        
        # 创建模型并加载预训练权重
        model = trainer.create_model()
        model.load_state_dict(pretrained_state)
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.MSELoss()
        
        best_smape = float('inf')
        best_model_state = None
        patience_counter = 0
        loss_records = []  # 每轮 (epoch, train_loss, val_smape)
        
        for epoch in range(1, MAX_EPOCHS + 1):
            # 训练一个epoch
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
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # 验证
            model.eval()
            val_smape = trainer.evaluate_smape(model, {node_id: val_loaders[node_id]}, node_minmax, real_space=True)
            loss_records.append((epoch, avg_train_loss, val_smape))
            
            proc_logger.debug(f"Epoch {epoch:2d}: train_loss={avg_train_loss:.6f}, val_smape={val_smape:.2f}%")
            
            # 早停判断
            if val_smape < best_smape - 1e-4:
                best_smape = val_smape
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    proc_logger.info(f"早停于第 {epoch} 轮，最佳 val_smape = {best_smape:.2f}%")
                    break
        
        # 如果从未更新最佳模型（例如只跑了一轮且未下降），则保存最后一轮
        if best_model_state is None:
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_smape = loss_records[-1][2] if loss_records else float('inf')
        
        # 保存最佳模型
        torch.save(best_model_state, MODEL_DIR / f"node_{node_id}.pth")
        
        # 保存损失记录到CSV
        loss_df = pd.DataFrame(loss_records, columns=["epoch", "train_loss", "val_smape"])
        loss_df.to_csv(LOSS_DIR / f"node_{node_id}_loss.csv", index=False)
        
        elapsed = time.time() - start_time
        proc_logger.info(f"完成，最佳 val_smape = {best_smape:.2f}%，耗时 {elapsed:.1f} 秒")
        return node_id, best_smape, loss_records, elapsed
    
    except Exception as e:
        proc_logger.error(f"微调失败: {e}", exc_info=True)
        return node_id, None, None, None

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("全节点微调（迁移学习 + 本地微调）")
    logger.info(f"配置: MAX_EPOCHS={MAX_EPOCHS}, LR={LR}, BATCH_SIZE={BATCH_SIZE}, PATIENCE={PATIENCE}")
    logger.info(f"设备: {DEVICE}, 并行进程数: {NUM_PROCESSES}")
    logger.info(f"输出目录: {OUTPUT_BASE}")
    logger.info("="*60)
    
    # 预检查必要文件
    if not PRETRAINED_MODEL.exists():
        logger.error(f"预训练模型不存在: {PRETRAINED_MODEL}")
        sys.exit(1)
    if not MINMAX_FILE.exists():
        logger.error(f"MinMax文件不存在: {MINMAX_FILE}")
        sys.exit(1)
    
    # 获取所有节点ID（仅用于分配任务）
    with open(MINMAX_FILE, 'rb') as f:
        node_minmax = pickle.load(f)
    node_ids = list(node_minmax.keys())
    logger.info(f"共 {len(node_ids)} 个节点: {node_ids[:5]}...")
    
    # 并行执行
    logger.info(f"启动 {NUM_PROCESSES} 个进程并行微调...")
    start_total = time.time()
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(finetune_node, node_ids)
    
    # 汇总结果
    summary = []
    all_smapes = []
    for node_id, smape, loss_records, elapsed in results:
        if smape is not None:
            summary.append({"node_id": node_id, "best_val_smape": smape, "elapsed_sec": elapsed})
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
    
    # 绘制所有节点的训练损失曲线（叠加）
    plt.figure(figsize=(12, 6))
    for node_id, _, loss_records, _ in results:
        if loss_records:
            epochs = [r[0] for r in loss_records]
            losses = [r[1] for r in loss_records]
            plt.plot(epochs, losses, alpha=0.3, linewidth=0.8, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("All Nodes Fine-tuning Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.savefig(CURVE_PNG, dpi=150)
    plt.close()
    logger.info(f"损失曲线图已保存至 {CURVE_PNG}")
    
    logger.info("="*60)
    logger.info("全节点微调完成！")
    logger.info(f"结果汇总: {SUMMARY_FILE}")
    logger.info(f"统计信息: {STATS_FILE}")
    logger.info(f"各节点模型: {MODEL_DIR}")
    logger.info(f"各节点损失记录: {LOSS_DIR}")
    logger.info("="*60)