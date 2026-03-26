#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
两阶段训练 - 口径修复 (五星级专业版)

阶段1: 旧口径预训练
阶段2: 新口径微调
"""

import sys
import os
import pickle
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor, train_epoch, evaluate

# ============================================================
# 配置类
# ============================================================

class Config:
    """集中配置管理"""
    
    # 路径配置
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    OLD_PATH = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_2019_2022"
    NEW_PATH = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_2023_2025"
    RESULTS_DIR = PROJECT_ROOT / "results" / "two_stage"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # 模型参数
    NODE_ID = 8001
    BATCH_SIZE = 64
    HIDDEN = 96
    LAYERS = 4
    DROPOUT = 0.2298
    
    # 训练参数
    EPOCHS_STAGE1 = 50
    EPOCHS_STAGE2 = 20
    LR_STAGE1 = 0.001003
    LR_STAGE2 = 0.0001
    
    # 随机种子
    RANDOM_SEED = 42


# ============================================================
# 日志系统
# ============================================================

def setup_logger() -> logging.Logger:
    """配置日志器"""
    logger = logging.getLogger("two_stage")
    logger.setLevel(logging.INFO)
    
    # 创建目录
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 控制台处理器
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    
    # 文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file = logging.FileHandler(
        Config.LOG_DIR / f"two_stage_{timestamp}.log",
        encoding='utf-8'
    )
    file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file)
    
    return logger


logger = setup_logger()


# ============================================================
# 数据加载
# ============================================================

def get_loaders(data_path: Path, node_id: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取数据加载器
    
    Args:
        data_path: 数据路径
        node_id: 节点ID
        batch_size: 批大小
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = data_path / f"node_{node_id}"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    train_dataset = BarcelonaDataset(str(data_dir / "train.pkl"))
    val_dataset = BarcelonaDataset(str(data_dir / "val.pkl"))
    test_dataset = BarcelonaDataset(str(data_dir / "test.pkl"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader


# ============================================================
# 评估指标
# ============================================================

def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 sMAPE"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def evaluate_smape(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """评估 sMAPE"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return compute_smape(all_targets, all_preds)


# ============================================================
# 训练函数
# ============================================================

def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    stage_name: str
) -> Dict:
    """训练单个阶段"""
    logger.info(f"\n{'='*50}")
    logger.info(f"{stage_name}")
    logger.info(f"{'='*50}")
    logger.info(f"学习率: {lr}, 轮数: {epochs}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1:2d}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    return history


def save_results(results: Dict, filepath: Path):
    """保存结果"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"结果保存: {filepath}")


# ============================================================
# 主流程
# ============================================================

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("两阶段训练 - 口径修复")
    logger.info("="*60)
    
    # 设置随机种子
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    # 加载数据
    logger.info("\n加载数据...")
    train_loader_old, val_loader_old, test_loader_old = get_loaders(
        Config.OLD_PATH, Config.NODE_ID, Config.BATCH_SIZE
    )
    train_loader_new, val_loader_new, test_loader_new = get_loaders(
        Config.NEW_PATH, Config.NODE_ID, Config.BATCH_SIZE
    )
    
    # 获取输入维度
    sample_x, _ = train_loader_old.dataset[0]
    input_dim = sample_x.shape[1]
    logger.info(f"输入维度: {input_dim}")
    
    # 创建模型
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN,
        num_layers=Config.LAYERS,
        output_dim=4,
        dropout=Config.DROPOUT
    )
    model.to(device)
    
    # ========== 阶段1：旧口径预训练 ==========
    history_stage1 = train_stage(
        model, train_loader_old, val_loader_old,
        Config.EPOCHS_STAGE1, Config.LR_STAGE1, device, "阶段1：旧口径预训练"
    )
    smape_old = evaluate_smape(model, test_loader_old, device)
    logger.info(f"\n阶段1完成 - 旧口径测试 sMAPE: {smape_old:.2f}%")
    
    # ========== 阶段2：新口径微调 ==========
    history_stage2 = train_stage(
        model, train_loader_new, val_loader_new,
        Config.EPOCHS_STAGE2, Config.LR_STAGE2, device, "阶段2：新口径微调"
    )
    smape_new = evaluate_smape(model, test_loader_new, device)
    logger.info(f"\n阶段2完成 - 新口径测试 sMAPE: {smape_new:.2f}%")
    
    # ========== 保存模型 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Config.RESULTS_DIR / f"two_stage_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型保存: {model_path}")
    
    # ========== 结果汇总 ==========
    results = {
        'timestamp': timestamp,
        'config': {
            'node_id': Config.NODE_ID,
            'hidden': Config.HIDDEN,
            'layers': Config.LAYERS,
            'dropout': Config.DROPOUT,
            'lr_stage1': Config.LR_STAGE1,
            'lr_stage2': Config.LR_STAGE2,
            'epochs_stage1': Config.EPOCHS_STAGE1,
            'epochs_stage2': Config.EPOCHS_STAGE2,
        },
        'metrics': {
            'old_smape': smape_old,
            'new_smape': smape_new,
        },
        'baselines': {
            'mixed': 58.12,
            'old_only': 65.53,
            'new_only': 70.78,
        },
        'history_stage1': history_stage1,
        'history_stage2': history_stage2,
    }
    
    # 保存结果
    results_path = Config.RESULTS_DIR / f"two_stage_results_{timestamp}.json"
    save_results(results, results_path)
    
    # 打印汇总
    logger.info("\n" + "="*60)
    logger.info("两阶段训练完成")
    logger.info("="*60)
    logger.info(f"混合口径基线: 58.12%")
    logger.info(f"旧口径单独: 65.53%")
    logger.info(f"新口径单独: 70.78%")
    logger.info(f"两阶段训练结果: {smape_new:.2f}%")
    
    if smape_new < 70.78:
        logger.info("✅ 两阶段训练有效！比新口径单独训练更好")
    else:
        logger.info("⚠️ 两阶段训练效果不明显，需要调整参数")
    
    return results


if __name__ == "__main__":
    results = main()
