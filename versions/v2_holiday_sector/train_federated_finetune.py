#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦微调 - 新口径41节点
加载预训练模型，使用 MinMax 参数对新口径原始能耗归一化
"""

import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from versions.v2_holiday_sector.train_federated_pretrain import LSTMPredictor, FederatedTrainer

# 自定义数据集：读取原始 Valor，用节点的 MinMax 参数归一化
class MinMaxBarcelonaDataset(Dataset):
    def __init__(self, data_path, node_id, node_minmax, window_size=28, predict_size=4):
        self.df = pd.read_pickle(data_path)
        self.window_size = window_size
        self.predict_size = predict_size
        self.data_min, self.data_max = node_minmax[node_id]
        # 原始能耗
        self.energy = self.df['Valor'].values
        # 部门、周末、节假日特征（与旧口径一致）
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
        indices = []
        total = len(self.energy)
        for i in range(total - self.window_size - self.predict_size + 1):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        # 能耗归一化
        x_energy = self.energy[start:start+self.window_size]
        x_energy = (x_energy - self.data_min) / (self.data_max - self.data_min + 1e-8)
        x_energy = torch.FloatTensor(x_energy).unsqueeze(-1)  # (window,1)

        # 部门特征（取窗口最后一个时间点的 sector，并重复 window 次）
        sector_idx = start + self.window_size - 1
        x_sector = self.sector_onehot[sector_idx]
        x_sector = torch.FloatTensor(x_sector).unsqueeze(0).repeat(self.window_size, 1)

        # 节假日特征
        x_holiday = self.holiday[start:start+self.window_size]
        x_holiday = torch.FloatTensor(x_holiday).unsqueeze(-1)

        # 周末特征
        x_weekend = self.weekend[start:start+self.window_size]
        x_weekend = torch.FloatTensor(x_weekend).unsqueeze(-1)

        x = torch.cat([x_energy, x_sector, x_holiday, x_weekend], dim=1)

        y = self.energy[start+self.window_size:start+self.window_size+self.predict_size]
        y = (y - self.data_min) / (self.data_max - self.data_min + 1e-8)
        y = torch.FloatTensor(y)

        return x, y

def load_node_data_new(node_ids, data_path, node_minmax, split='train'):
    loaders = {}
    for node_id in node_ids:
        node_dir = data_path / f"node_{node_id}"
        pkl_file = node_dir / f"{split}.pkl"
        if not pkl_file.exists():
            print(f"警告: {pkl_file} 不存在，跳过节点 {node_id}")
            continue
        dataset = MinMaxBarcelonaDataset(str(pkl_file), node_id, node_minmax)
        loader = DataLoader(dataset, batch_size=64, shuffle=(split=='train'), drop_last=False)
        loaders[node_id] = loader
    return loaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # 加载节点 MinMax 映射
    with open(PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl", 'rb') as f:
        node_minmax = pickle.load(f)
    new_nodes = list(node_minmax.keys())  # 与旧口径相同节点列表
    new_nodes.sort()
    print(f"使用 {len(new_nodes)} 个新口径节点进行微调")

    data_path = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_2023_2025"
    train_loaders = load_node_data_new(new_nodes, data_path, node_minmax, split='train')
    val_loaders = load_node_data_new(new_nodes, data_path, node_minmax, split='val')

    trainer = FederatedTrainer(args)
    # 创建模型并加载预训练权重
    model = trainer.create_model().to(args.device)
    pretrain_path = PROJECT_ROOT / "results" / "two_stage" / "model_fed_pretrain.pth"
    if pretrain_path.exists():
        model.load_state_dict(torch.load(pretrain_path, map_location=args.device))
        print(f"加载预训练模型: {pretrain_path}")
    else:
        print("警告: 未找到预训练模型，从头开始训练")

    # 微调
    model = trainer.train(train_loaders, val_loaders, args.rounds, args.mu)

    # 测试集评估（反归一化后真实 sMAPE）
    test_loaders = load_node_data_new(new_nodes, data_path, node_minmax, split='test')
    model.eval()
    all_preds_real, all_targets_real = [], []
    with torch.no_grad():
        for node_id, loader in test_loaders.items():
            data_min, data_max = node_minmax[node_id]
            for x, y in loader:
                x = x.to(args.device)
                pred_norm = model(x).cpu().numpy()
                pred_real = pred_norm * (data_max - data_min) + data_min
                target_real = y.numpy() * (data_max - data_min) + data_min
                all_preds_real.append(pred_real)
                all_targets_real.append(target_real)
    all_preds_real = np.concatenate(all_preds_real)
    all_targets_real = np.concatenate(all_targets_real)
    denominator = (np.abs(all_targets_real) + np.abs(all_preds_real)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(np.abs(all_targets_real - all_preds_real) / denominator) * 100
    print(f"最终测试集真实 sMAPE: {smape:.2f}%")

    # 保存微调模型
    model_path = PROJECT_ROOT / "results" / "two_stage" / "model_fed_finetune.pth"
    torch.save(model.state_dict(), model_path)
    print(f"微调模型保存至 {model_path}")

if __name__ == "__main__":
    main()