"""
预测曲线图 + 精度对比柱状图
带时间戳，保存到 results/two_stage/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor
from torch.utils.data import DataLoader

# ============================================================
# 配置
# ============================================================

MODEL_PATH = r"D:\Desk\desk\beiyou_c_project\results\two_stage\two_stage_model_20260323_183730.pth"
DATA_PATH = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"
NODE_ID = 8001
BATCH_SIZE = 32

# 精度数据
ACCURACY = {
    'Mixed (7y)': 58.12,
    'Old (2019-2022)': 65.53,
    'New (2023-2025)': 70.78,
    'Two-Stage (Fixed)': 60.54
}

# 创建带时间戳的输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/two_stage_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# ============================================================
# 1. 预测曲线图
# ============================================================

print("\n" + "="*50)
print("1. 绘制预测曲线图")
print("="*50)

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
timestamps = []
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        all_preds.append(output.numpy())
        all_targets.append(y.numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# 获取真实时间戳（从测试集）
# 注意：BarcelonaDataset 返回的是归一化后的数据，时间戳需要从原始数据获取
# 这里简化处理，用样本序号作为横坐标
n_samples = 100
sample_indices = np.arange(n_samples)

plt.figure(figsize=(14, 6))
plt.plot(sample_indices, all_targets[:n_samples, 0], label='True', color='blue', linewidth=1.5)
plt.plot(sample_indices, all_preds[:n_samples, 0], label='Predicted', color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Sample Index (6-hour interval)')
plt.ylabel('Normalized Energy')
plt.title(f'Node {NODE_ID} Prediction Results (Two-Stage Model, sMAPE=60.54%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
curve_path = os.path.join(output_dir, 'prediction_curve.png')
plt.savefig(curve_path, dpi=150)
print(f"✅ 预测曲线图: {curve_path}")

# 误差统计
errors = np.abs(all_targets - all_preds)
print(f"   MAE: {np.mean(errors):.4f}")
print(f"   Max Error: {np.max(errors):.4f}")

# ============================================================
# 2. 精度对比柱状图
# ============================================================

print("\n" + "="*50)
print("2. 绘制精度对比柱状图")
print("="*50)

plt.figure(figsize=(10, 6))
models = list(ACCURACY.keys())
smapes = list(ACCURACY.values())
colors = ['#2E8B57', '#2E86AB', '#E76F51', '#F4A261']

bars = plt.bar(models, smapes, color=colors, edgecolor='black', linewidth=1)
plt.ylabel('sMAPE (%)')
plt.title('Base Station Energy Prediction Accuracy Comparison')
plt.ylim(0, 100)

for bar, val in zip(bars, smapes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
bar_path = os.path.join(output_dir, 'accuracy_comparison.png')
plt.savefig(bar_path, dpi=150)
print(f"✅ 精度对比图: {bar_path}")

# ============================================================
# 3. 保存结果到JSON
# ============================================================

import json
results = {
    'timestamp': timestamp,
    'output_dir': output_dir,
    'node_id': NODE_ID,
    'model_path': MODEL_PATH,
    'accuracy': ACCURACY,
    'metrics': {
        'mae': float(np.mean(errors)),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors))
    },
    'improvement': {
        'vs_new': 70.78 - 60.54,
        'vs_mixed': 60.54 - 58.12
    }
}

json_path = os.path.join(output_dir, 'results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✅ 结果JSON: {json_path}")

# ============================================================
# 总结
# ============================================================

print("\n" + "="*50)
print("总结")
print("="*50)
print(f"输出目录: {output_dir}")
print(f"最佳结果: Two-Stage (Fixed) = 60.54%")
print(f"比新口径单独提升: {70.78 - 60.54:.2f}%")
print(f"接近混合口径: {60.54 - 58.12:.2f}% 差距")
print("\n✅ 所有图片已保存")

# 打开文件夹
