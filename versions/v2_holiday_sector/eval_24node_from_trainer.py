#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接使用 train_federated_pro_full.py 中的评估函数评估24节点模型"""

import sys
sys.path.insert(0, 'D:/Desk/desk/beiyou_c_project')

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import yaml

# 加载配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
with open(PROJECT_ROOT / "versions" / "v2_holiday_sector" / "configs" / "paths.yaml", 'r') as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

DATA_ROOT = Path(GLOBAL_CONFIG['data_root'])
DATA_PATH = DATA_ROOT / GLOBAL_CONFIG['barcelona'][GLOBAL_CONFIG['current']['barcelona']]

from src.raspberry.model import LSTMPredictor
from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset

def get_loader(node_id, split, batch_size):
    data_dir = DATA_PATH / f"node_{node_id}"
    file_path = data_dir / f"{split}.pkl"
    if not file_path.exists():
        return None
    dataset = BarcelonaDataset(str(file_path), window_size=28, predict_size=4,
                                sector_feature=True, holiday_feature=True, weekend_feature=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

def evaluate_smape(model, test_loaders, device):
    """与 train_federated_pro_full.py 完全一致"""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for loader in test_loaders.values():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    denominator = (np.abs(all_targets) + np.abs(all_preds)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(all_targets - all_preds) / denominator) * 100

# 配置
model_path = 'D:/Desk/desk/beiyou_c_project/results/beautified/federated_nodes24_rounds30_mu0.1_model_20260323_063955.pth'
node_ids = list(range(8001, 8025))
batch_size = 64
device = torch.device('cpu')

print("="*50)
print("评估24节点联邦模型 (使用训练脚本评估函数)")
print("="*50)
print(f"数据路径: {DATA_PATH}")
print(f"节点数: {len(node_ids)}")

# 加载模型
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

# 从state_dict推断参数
input_size = state_dict['lstm.weight_ih_l0'].shape[1]
hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
num_layers = max([int(k.split('_ih_l')[1].split('.')[0]) for k in state_dict if 'lstm.weight_ih_l' in k]) + 1
output_size = state_dict['fc.weight'].shape[0]

print(f"模型参数: input={input_size}, hidden={hidden_size}, layers={num_layers}, output={output_size}")

model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 加载测试数据
test_loaders = {}
for node_id in node_ids:
    loader = get_loader(node_id, 'test', batch_size)
    if loader:
        test_loaders[node_id] = loader

print(f"有效测试节点: {len(test_loaders)}/{len(node_ids)}")

# 评估
smape = evaluate_smape(model, test_loaders, device)
print(f"\n{'='*50}")
print(f"24节点联邦模型 sMAPE: {smape:.2f}%")
print(f"{'='*50}")
