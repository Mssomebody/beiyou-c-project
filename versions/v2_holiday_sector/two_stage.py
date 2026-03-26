"""
两阶段训练 - 口径修复
阶段1: 旧口径预训练
阶段2: 新口径微调
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor, train_epoch, evaluate

# ============================================================
# 配置
# ============================================================

OLD_PATH = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2019_2022"
NEW_PATH = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"
NODE_ID = 8001
BATCH_SIZE = 64
EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 10
LR_STAGE1 = 0.00572      # 混合口径最佳学习率
LR_STAGE2 = 0.0005       # 微调用小学习率
HIDDEN = 160
LAYERS = 4
DROPOUT = 0.157

# ============================================================
# 数据加载
# ============================================================

def get_loaders(data_path, node_id, batch_size):
    data_dir = os.path.join(data_path, f"node_{node_id}")
    train_dataset = BarcelonaDataset(os.path.join(data_dir, "train.pkl"))
    val_dataset = BarcelonaDataset(os.path.join(data_dir, "val.pkl"))
    test_dataset = BarcelonaDataset(os.path.join(data_dir, "test.pkl"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader

# ============================================================
# 评估 sMAPE
# ============================================================

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
# 主流程
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 加载数据
print("\n" + "="*50)
print("加载数据")
print("="*50)
train_loader_old, val_loader_old, test_loader_old = get_loaders(OLD_PATH, NODE_ID, BATCH_SIZE)
train_loader_new, val_loader_new, test_loader_new = get_loaders(NEW_PATH, NODE_ID, BATCH_SIZE)

# 获取输入维度
sample_x, _ = train_loader_old.dataset[0]
input_dim = sample_x.shape[1]
print(f"输入维度: {input_dim}")

# ============================================================
# 阶段1：旧口径预训练
# ============================================================
print("\n" + "="*50)
print("阶段1：旧口径预训练")
print("="*50)

model = LSTMPredictor(
    input_dim=input_dim,
    hidden_dim=HIDDEN,
    num_layers=LAYERS,
    output_dim=4,
    dropout=DROPOUT
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_STAGE1)
criterion = nn.MSELoss()

for epoch in range(EPOCHS_STAGE1):
    train_loss = train_epoch(model, train_loader_old, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader_old, criterion, device)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

# 阶段1评估（旧口径测试集）
smape_stage1 = evaluate_smape(model, test_loader_old, device)
print(f"\n阶段1完成 - 旧口径测试 sMAPE: {smape_stage1:.2f}%")

# ============================================================
# 阶段2：新口径微调
# ============================================================
print("\n" + "="*50)
print("阶段2：新口径微调")
print("="*50)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_STAGE2)

for epoch in range(EPOCHS_STAGE2):
    train_loss = train_epoch(model, train_loader_new, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader_new, criterion, device)
    print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

# 阶段2评估（新口径测试集）
smape_stage2 = evaluate_smape(model, test_loader_new, device)
print(f"\n阶段2完成 - 新口径测试 sMAPE: {smape_stage2:.2f}%")

# ============================================================
# 结果汇总
# ============================================================
print("\n" + "="*50)
print("两阶段训练完成")
print("="*50)
print(f"混合口径基线: 58.12%")
print(f"旧口径单独: 65.53%")
print(f"新口径单独: 70.78%")
print(f"两阶段训练结果: {smape_stage2:.2f}%")

if smape_stage2 < 70.78:
    print("✅ 两阶段训练有效！比新口径单独训练更好")
else:
    print("⚠️ 两阶段训练效果不明显，需要调整参数")
