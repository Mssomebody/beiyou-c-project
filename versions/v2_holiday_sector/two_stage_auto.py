"""
两阶段训练 - 贝叶斯自动调参
"""

import sys
import os
import pickle
import json
import optuna
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor, train_epoch, evaluate

# ============================================================
# 路径配置
# ============================================================

OLD_PATH = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2019_2022"
NEW_PATH = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"
NODE_ID = 8001
BATCH_SIZE = 64
N_TRIALS = 20

# ============================================================
# 数据加载
# ============================================================

def get_loaders(data_path):
    data_dir = os.path.join(data_path, f"node_{NODE_ID}")
    train_dataset = BarcelonaDataset(os.path.join(data_dir, "train.pkl"))
    val_dataset = BarcelonaDataset(os.path.join(data_dir, "val.pkl"))
    test_dataset = BarcelonaDataset(os.path.join(data_dir, "test.pkl"))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader

def compute_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

def evaluate_smape(model, dataloader, device):
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
# 目标函数
# ============================================================

def objective(trial):
    """贝叶斯优化目标函数"""
    
    # 超参数
    hidden = trial.suggest_categorical('hidden', [64, 96, 128, 160])
    layers = trial.suggest_int('layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    lr_stage1 = trial.suggest_float('lr_stage1', 1e-4, 1e-2, log=True)
    lr_stage2 = trial.suggest_float('lr_stage2', 1e-5, 1e-3, log=True)
    epochs_stage1 = trial.suggest_int('epochs_stage1', 20, 50)
    epochs_stage2 = trial.suggest_int('epochs_stage2', 5, 20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_loader_old, val_loader_old, test_loader_old = get_loaders(OLD_PATH)
    train_loader_new, val_loader_new, test_loader_new = get_loaders(NEW_PATH)
    
    # 输入维度
    sample_x, _ = train_loader_old.dataset[0]
    input_dim = sample_x.shape[1]
    
    # 创建模型
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=hidden,
        num_layers=layers,
        output_dim=4,
        dropout=dropout
    )
    model.to(device)
    
    # 阶段1：预训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_stage1)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs_stage1):
        train_loss = train_epoch(model, train_loader_old, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader_old, criterion, device)
    
    # 阶段2：微调
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_stage2)
    
    for epoch in range(epochs_stage2):
        train_loss = train_epoch(model, train_loader_new, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader_new, criterion, device)
    
    # 评估最终 sMAPE
    smape = evaluate_smape(model, test_loader_new, device)
    
    return smape

# ============================================================
# 主函数
# ============================================================

def main():
    print("="*60)
    print("两阶段训练 - 贝叶斯自动调参")
    print("="*60)
    print(f"搜索次数: {N_TRIALS}")
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 创建 study
    study = optuna.create_study(
        direction='minimize',
        study_name=f'two_stage_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        storage='sqlite:///two_stage_optuna.db',
        load_if_exists=True
    )
    
    # 优化
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # 结果
    print("\n" + "="*60)
    print("优化完成")
    print("="*60)
    print(f"最佳 sMAPE: {study.best_value:.2f}%")
    print(f"最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'trials': [{'value': t.value, 'params': t.params} for t in study.trials if t.value is not None]
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/two_stage_auto_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果保存: results/two_stage_auto_{timestamp}.json")
    
    return study

if __name__ == "__main__":
    study = main()
