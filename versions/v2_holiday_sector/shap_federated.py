"""
SHAP 指导的个性化联邦学习
4G 和 5G 共享模型，但聚合时按特征重要性加权
"""

import shap
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 配置
DATA_DIR = Path("D:/Desk/desk/beiyou_c_project/data/processed/tsinghua_v2")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_NAMES = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']

class StationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 加载数据
print("加载数据...")
with open(DATA_DIR / '4g' / 'station_10055' / 'data.pkl', 'rb') as f:
    data_4g = pickle.load(f)
with open(DATA_DIR / '5g' / 'station_13905' / 'data.pkl', 'rb') as f:
    data_5g = pickle.load(f)

X_train_4g = data_4g['X_train_norm'][:500]
y_train_4g = data_4g['y_train_norm'][:500]
X_train_5g = data_5g['X_train_norm'][:500]
y_train_5g = data_5g['y_train_norm'][:500]

# 训练模型
print("\n训练 4G 模型...")
model_4g = LSTMPredictor().to(DEVICE)
optimizer = torch.optim.Adam(model_4g.parameters(), lr=0.001)
criterion = nn.MSELoss()

dataset_4g = StationDataset(X_train_4g, y_train_4g)
loader_4g = DataLoader(dataset_4g, batch_size=64, shuffle=True)

for epoch in range(20):
    total_loss = 0
    for x, y in loader_4g:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model_4g(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(loader_4g):.4f}")

print("\n训练 5G 模型...")
model_5g = LSTMPredictor().to(DEVICE)
optimizer = torch.optim.Adam(model_5g.parameters(), lr=0.001)

dataset_5g = StationDataset(X_train_5g, y_train_5g)
loader_5g = DataLoader(dataset_5g, batch_size=64, shuffle=True)

for epoch in range(20):
    total_loss = 0
    for x, y in loader_5g:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model_5g(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(loader_5g):.4f}")

# SHAP 分析
print("\n" + "="*50)
print("SHAP 特征重要性分析")
print("="*50)

# 4G SHAP
model_4g.eval()
background_4g = X_train_4g[:100]
explainer_4g = shap.DeepExplainer(model_4g, torch.FloatTensor(background_4g).to(DEVICE))
shap_values_4g = explainer_4g.shap_values(torch.FloatTensor(X_train_4g[:10]).to(DEVICE))

# 5G SHAP
model_5g.eval()
background_5g = X_train_5g[:100]
explainer_5g = shap.DeepExplainer(model_5g, torch.FloatTensor(background_5g).to(DEVICE))
shap_values_5g = explainer_5g.shap_values(torch.FloatTensor(X_train_5g[:10]).to(DEVICE))

# 计算平均重要性
importance_4g = np.abs(np.array(shap_values_4g)).mean(axis=(0,1))
importance_5g = np.abs(np.array(shap_values_5g)).mean(axis=(0,1))

print(f"\n4G 特征重要性:")
for name, imp in zip(FEATURE_NAMES, importance_4g):
    print(f"  {name}: {imp:.4f}")

print(f"\n5G 特征重要性:")
for name, imp in zip(FEATURE_NAMES, importance_5g):
    print(f"  {name}: {imp:.4f}")

# 画图
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(FEATURE_NAMES))
width = 0.35
ax.bar(x - width/2, importance_4g, width, label='4G', color='#2E8B57')
ax.bar(x + width/2, importance_5g, width, label='5G', color='#E76F51')
ax.set_xlabel('特征')
ax.set_ylabel('SHAP 重要性')
ax.set_title('4G vs 5G 特征重要性对比')
ax.set_xticks(x)
ax.set_xticklabels(FEATURE_NAMES)
ax.legend()
plt.tight_layout()
plt.savefig('results/shap_importance.png', dpi=150)
print(f"\n✅ 图片保存: results/shap_importance.png")

print("\n" + "="*50)
print("结论")
print("="*50)
if importance_4g[0] > importance_4g[2]:
    print("✅ 4G: PRB 重要性 > 用户数")
else:
    print("⚠️ 4G: 用户数更重要")
if importance_5g[2] > importance_5g[0]:
    print("✅ 5G: 用户数重要性 > PRB")
else:
    print("⚠️ 5G: PRB 更重要")

print("\n联邦学习时，聚合权重应该按此分配:")
print(f"  4G 的 PRB 权重应更高")
print(f"  5G 的用户数权重应更高")
