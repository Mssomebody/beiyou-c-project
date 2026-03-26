#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4G+5G代际协同 - 对比实验
实验1: 4G单独训练
实验2: 5G单独训练
实验3: 4G+5G联邦学习

对比目标：证明联邦学习能让5G从4G学到知识
"""

import os
import sys
import pickle
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed" / "tsinghua"
    RESULTS_DIR = PROJECT_ROOT / "results" / "4g_5g_comparison"
    
    # 模型配置
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BATCH_SIZE = 64
    SEQ_LEN = 24
    PRED_LEN = 1
    EPOCHS = 50
    LR = 0.001
    
    # 联邦配置
    LOCAL_EPOCHS = 5
    NUM_ROUNDS = 10
    
    SEED = 42


# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据集
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, target: np.ndarray, seq_len=24, pred_len=1):
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.pred_len = pred_len
    
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.target[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y.flatten())


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ============================================================
# 数据加载
# ============================================================

def load_station_data(data_dir: Path, max_stations: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """加载基站数据"""
    station_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
    station_dirs = station_dirs[:max_stations]
    
    all_features = []
    all_targets = []
    
    for station_dir in tqdm(station_dirs, desc="加载数据"):
        data_path = station_dir / 'data.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        all_features.append(data['features'])
        all_targets.append(data['target'])
    
    features = np.concatenate(all_features, axis=0)
    target = np.concatenate(all_targets, axis=0)
    
    return features, target


def create_dataloader(features, target, batch_size, seq_len, pred_len, train_ratio=0.8):
    """创建数据加载器"""
    dataset = TimeSeriesDataset(features, target, seq_len, pred_len)
    n = len(dataset)
    train_size = int(n * train_ratio)
    test_size = n - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============================================================
# 训练函数
# ============================================================

def train_single(model, train_loader, test_loader, epochs, lr, device):
    """单独训练"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
    
    model.load_state_dict(best_model)
    return model, best_loss


def federated_train(models, client_loaders, rounds, local_epochs, lr, device):
    """联邦训练"""
    for client_models in models.values():
        for m in client_models:
            m.to(device)
    
    criterion = nn.MSELoss()
    
    for round_num in range(1, rounds + 1):
        client_updates = []
        client_losses = []
        
        for client_name, client_models_list in models.items():
            for model in client_models_list:
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                for _ in range(local_epochs):
                    model.train()
                    for x, y in client_loaders[client_name]:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        output = model(x)
                        loss = criterion(output, y)
                        loss.backward()
                        optimizer.step()
                
                client_updates.append(model.state_dict())
                client_losses.append(loss.item())
        
        # 聚合（简单平均）
        avg_state_dict = {}
        for key in client_updates[0].keys():
            avg_state_dict[key] = torch.stack([u[key].float() for u in client_updates]).mean(dim=0)
        
        for client_name in models:
            for model in models[client_name]:
                model.load_state_dict(avg_state_dict)
        
        if round_num % 5 == 0:
            logger.info(f"  Round {round_num}: avg_loss={np.mean(client_losses):.6f}")
    
    return models


# ============================================================
# 评估
# ============================================================

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


# ============================================================
# 主函数
# ============================================================

def main():
    logger.info("="*60)
    logger.info("4G+5G代际协同对比实验")
    logger.info("="*60)
    
    # 创建结果目录
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    # 加载数据
    logger.info("\n加载数据...")
    
    # 4G数据
    data_4g = Config.DATA_DIR / '4g'
    features_4g, target_4g = load_station_data(data_4g, max_stations=500)
    logger.info(f"4G: {len(features_4g)} 样本")
    
    # 5G数据
    data_5g = Config.DATA_DIR / '5g'
    features_5g, target_5g = load_station_data(data_5g, max_stations=500)
    logger.info(f"5G: {len(features_5g)} 样本")
    
    # 创建数据加载器
    train_loader_4g, test_loader_4g = create_dataloader(
        features_4g, target_4g, Config.BATCH_SIZE, Config.SEQ_LEN, Config.PRED_LEN
    )
    train_loader_5g, test_loader_5g = create_dataloader(
        features_5g, target_5g, Config.BATCH_SIZE, Config.SEQ_LEN, Config.PRED_LEN
    )
    
    results = {}
    
    # ========== 实验1: 4G单独训练 ==========
    logger.info("\n" + "="*60)
    logger.info("实验1: 4G单独训练")
    logger.info("="*60)
    
    model_4g = LSTMPredictor(
        input_dim=features_4g.shape[1],
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    )
    model_4g, loss_4g = train_single(
        model_4g, train_loader_4g, test_loader_4g,
        Config.EPOCHS, Config.LR, device
    )
    results['4g_alone'] = loss_4g
    logger.info(f"4G单独训练测试损失: {loss_4g:.6f}")
    
    # ========== 实验2: 5G单独训练 ==========
    logger.info("\n" + "="*60)
    logger.info("实验2: 5G单独训练")
    logger.info("="*60)
    
    model_5g = LSTMPredictor(
        input_dim=features_5g.shape[1],
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    )
    model_5g, loss_5g = train_single(
        model_5g, train_loader_5g, test_loader_5g,
        Config.EPOCHS, Config.LR, device
    )
    results['5g_alone'] = loss_5g
    logger.info(f"5G单独训练测试损失: {loss_5g:.6f}")
    
    # ========== 实验3: 4G+5G联邦学习 ==========
    logger.info("\n" + "="*60)
    logger.info("实验3: 4G+5G联邦学习")
    logger.info("="*60)
    
    # 创建客户端模型
    models = {
        '4g': [LSTMPredictor(
            input_dim=features_4g.shape[1],
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        )],
        '5g': [LSTMPredictor(
            input_dim=features_5g.shape[1],
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        )]
    }
    
    client_loaders = {
        '4g': train_loader_4g,
        '5g': train_loader_5g
    }
    
    models = federated_train(
        models, client_loaders,
        Config.NUM_ROUNDS, Config.LOCAL_EPOCHS, Config.LR, device
    )
    
    # 评估联邦模型
    loss_4g_fed = evaluate_model(models['4g'][0], test_loader_4g, device)
    loss_5g_fed = evaluate_model(models['5g'][0], test_loader_5g, device)
    
    results['4g_federated'] = loss_4g_fed
    results['5g_federated'] = loss_5g_fed
    
    logger.info(f"4G联邦学习测试损失: {loss_4g_fed:.6f}")
    logger.info(f"5G联邦学习测试损失: {loss_5g_fed:.6f}")
    
    # ========== 结果汇总 ==========
    logger.info("\n" + "="*60)
    logger.info("结果汇总")
    logger.info("="*60)
    logger.info(f"4G单独训练:     {results['4g_alone']:.6f}")
    logger.info(f"4G联邦学习:     {results['4g_federated']:.6f}")
    logger.info(f"4G提升:         {(results['4g_alone'] - results['4g_federated']):.6f}")
    logger.info(f"")
    logger.info(f"5G单独训练:     {results['5g_alone']:.6f}")
    logger.info(f"5G联邦学习:     {results['5g_federated']:.6f}")
    logger.info(f"5G提升:         {(results['5g_alone'] - results['5g_federated']):.6f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = Config.RESULTS_DIR / f"results_{timestamp}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✅ 结果保存: {result_path}")
    
    return results


if __name__ == "__main__":
    main()
