#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.barcelona_dataset_v1 import get_node_data_loader
from experiments.beautified.train_single_node import (
    LSTMPredictor, train_epoch, evaluate, evaluate_original_scale,
    plot_predictions, plot_loss_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import argparse
from datetime import datetime


def train_v1(node_id=8001, epochs=20, batch_size=48, hidden_dim=192, lr=0.002, dropout=0.45, percentile=15):
    print("=" * 60)
    print(f"Single Node Training v1: Node {node_id}")
    print("=" * 60)
    
    device = torch.device('cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(base_dir, "results", "beautified")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载数据
    train_loader, scaler_path, train_dataset = get_node_data_loader(
        node_id=node_id, split='train', batch_size=batch_size, shuffle=True,
        window_size=28, predict_size=4, sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    val_loader, _, _ = get_node_data_loader(
        node_id=node_id, split='val', batch_size=batch_size, shuffle=False,
        window_size=28, predict_size=4, sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    test_loader, _, _ = get_node_data_loader(
        node_id=node_id, split='test', batch_size=batch_size, shuffle=False,
        window_size=28, predict_size=4, sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]
    
    print(f"输入维度: {input_dim}")
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")
    
    model = LSTMPredictor(input_dim, hidden_dim, 2, 4, dropout).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("\n开始训练...")
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    test_loss = evaluate(model, test_loader, criterion, device)
    eval_results = evaluate_original_scale(model, test_loader, scaler, device, percentile)
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"RMSE: {eval_results['rmse']:.2f} kWh")
    print(f"MAE:  {eval_results['mae']:.2f} kWh")
    print(f"sMAPE: {eval_results['smape']:.2f}%")
    
    # 绘图
    pred_path = os.path.join(save_dir, f"node_{node_id}_v1_predictions_{timestamp}.png")
    loss_path = os.path.join(save_dir, f"node_{node_id}_v1_loss_{timestamp}.png")
    plot_predictions(eval_results['predictions'], eval_results['targets'], node_id, pred_path)
    plot_loss_curve(train_losses, val_losses, f"Node {node_id} v1", loss_path)
    
    results = {'smape': eval_results['smape'], 'rmse': eval_results['rmse'], 'mae': eval_results['mae']}
    print(f"\n 完成! sMAPE: {results['smape']:.2f}%")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=int, default=8001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=192)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--dropout', type=float, default=0.45)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--percentile', type=int, default=15)
    args = parser.parse_args()
    
    train_v1(
        node_id=args.node,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        dropout=args.dropout,
        percentile=args.percentile
    )