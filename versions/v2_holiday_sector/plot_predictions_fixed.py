"""
预测曲线画图 - 中文字体修复版
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor
from torch.utils.data import DataLoader

# 配置
MODEL_PATH = r"D:\Desk\desk\beiyou_c_project\results\two_stage\two_stage_model_20260323_183730.pth"
DATA_PATH = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"
NODE_ID = 8001
BATCH_SIZE = 32

# 加载数据
data_dir = os.path.join(DATA_PATH, f"node_{NODE_ID}")
test_dataset = BarcelonaDataset(os.path.join(data_dir, "test.pkl"))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 加载模型
sample_x, _ = test_dataset[0]
input_dim = sample_x.shape[1]
model = LSTMPredictor(input_dim=input_dim, hidden_dim=96, num_layers=4, output_dim=4, dropout=0.2298)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# 预测
all_preds = []
all_targets = []
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        all_preds.append(output.numpy())
        all_targets.append(y.numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# 画图
n_samples = 50
plt.figure(figsize=(14, 6))
plt.plot(all_targets[:n_samples, 0], label='True', color='blue', linewidth=1.5)
plt.plot(all_preds[:n_samples, 0], label='Predicted', color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Sample Index')
plt.ylabel('Normalized Energy')
plt.title(f'Node {NODE_ID} Prediction Results (Two-Stage Model)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/predictions_fixed.png', dpi=150)
print("✅ 图片保存: results/predictions_fixed.png")

# 误差统计
errors = np.abs(all_targets - all_preds)
print(f"\nError Statistics:")
print(f"  MAE: {np.mean(errors):.4f}")
print(f"  Max Error: {np.max(errors):.4f}")
print(f"  Std: {np.std(errors):.4f}")
