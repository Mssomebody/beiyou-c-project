#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦预训练 - 旧口径41节点
使用 MinMax 归一化，保存模型和每个节点的归一化参数
"""

import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from versions.v2_holiday_sector.src.data_loader.barcelona_dataset_v1 import BarcelonaDataset

# ============================================================
# 模型定义
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
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

    def create_model(self):
        return LSTMPredictor(
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

    def evaluate_smape(self, model, loaders):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for loader in loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x)
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        denominator = (np.abs(all_targets) + np.abs(all_preds)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(np.abs(all_targets - all_preds) / denominator) * 100
        return smape

    def train(self, client_loaders, val_loaders, rounds, mu):
        model = self.create_model().to(self.device)
        for r in range(1, rounds+1):
            model, avg_loss = self.train_round(model, client_loaders, mu)
            val_smape = self.evaluate_smape(model, val_loaders)
            print(f"Round {r:3d}: train_loss = {avg_loss:.6f}, val_smape = {val_smape:.2f}%")
        return model

def load_node_data(node_ids, data_path, split='train', window_size=28, predict_size=4):
    loaders = {}
    for node_id in node_ids:
        node_dir = data_path / f"node_{node_id}"
        pkl_file = node_dir / f"{split}.pkl"
        if not pkl_file.exists():
            print(f"警告: {pkl_file} 不存在，跳过节点 {node_id}")
            continue
        dataset = BarcelonaDataset(str(pkl_file), window_size, predict_size)
        loader = DataLoader(dataset, batch_size=64, shuffle=(split=='train'), drop_last=False)
        loaders[node_id] = loader
    return loaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mu', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # 节点列表（剔除8025）
    old_nodes = []
    for node_dir in (PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1").glob("node_*"):
        node_id = int(node_dir.name.split('_')[1])
        if node_id != 8025:
            old_nodes.append(node_id)
    old_nodes.sort()
    print(f"使用 {len(old_nodes)} 个旧口径节点进行预训练")

    data_path = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"
    train_loaders = load_node_data(old_nodes, data_path, split='train')
    val_loaders = load_node_data(old_nodes, data_path, split='val')

    trainer = FederatedTrainer(args)
    model = trainer.train(train_loaders, val_loaders, args.rounds, args.mu)

    # 保存模型
    model_path = PROJECT_ROOT / "results" / "two_stage" / "model_fed_pretrain.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"预训练模型保存至 {model_path}")

if __name__ == "__main__":
    main()