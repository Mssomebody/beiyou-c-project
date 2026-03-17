# day1_mlp.py
# Week 1 Day 1: 第一个神经网络 - 多层感知机(MLP)分类器
# 目标：理解PyTorch基础，能训练、预测、评估

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

print("=" * 50)
print("Week 1 Day 1: PyTorch基础 - 多层感知机(MLP)")
print("=" * 50)

# ==================== 1. 生成模拟数据 ====================
print("\n[1/5] 生成模拟数据...")

# 模拟10个基站的特征：业务量、温度、时间(小时)
# 目标：预测基站是否高能耗（二分类）

n_samples = 1000  # 1000个样本

# 特征：业务量(GB), 温度(°C), 时间(0-23小时)
traffic = np.random.normal(100, 30, n_samples)  # 均值100，标准差30
temperature = np.random.normal(25, 5, n_samples)  # 均值25°C
hour = np.random.randint(0, 24, n_samples)  # 0-23点

# 合成特征矩阵
X = np.column_stack([traffic, temperature, hour])

# 标签：高能耗规则（业务量>120 且 温度>28 且 时间在10-22点）
y = ((traffic > 120) & (temperature > 28) & (hour >= 10) & (hour <= 22)).astype(int)

# 划分训练集和测试集（80%训练，20%测试）
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"  总样本: {n_samples}")
print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
print(f"  高能耗样本比例: {y.mean():.2%}")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 增加维度，变成[n, 1]
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==================== 2. 定义神经网络 ====================
print("\n[2/5] 定义神经网络...")

class MLP(nn.Module):
    """
    多层感知机：3输入 -> 64隐藏 -> 32隐藏 -> 1输出
    """
    def __init__(self, input_size=3, hidden1=64, hidden2=32, output_size=1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden2, output_size)
        self.sigmoid = nn.Sigmoid()  # 二分类输出概率
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

# 实例化模型
model = MLP()
print(f"  模型结构:\n{model}")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"  总参数量: {total_params}")

# ==================== 3. 训练配置 ====================
print("\n[3/5] 配置训练...")

criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

n_epochs = 50  # 训练50轮
train_losses = []  # 记录损失
train_accs = []  # 记录准确率

print(f"  优化器: Adam, 学习率: 0.001")
print(f"  损失函数: BCELoss")
print(f"  训练轮数: {n_epochs}")

# ==================== 4. 训练循环 ====================
print("\n[4/5] 开始训练...")

for epoch in range(n_epochs):
    model.train()  # 训练模式
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        epoch_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    # 计算epoch平均损失和准确率
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total
    
    train_losses.append(avg_loss)
    train_accs.append(accuracy)
    
    # 每10轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

print(f"  最终训练准确率: {train_accs[-1]:.4f}")

# ==================== 5. 测试评估 ====================
print("\n[5/5] 测试评估...")

model.eval()  # 评估模式
with torch.no_grad():  # 不计算梯度，节省内存
    test_outputs = model(X_test_tensor)
    test_predicted = (test_outputs > 0.5).float()
    test_accuracy = (test_predicted == y_test_tensor).float().mean()
    
    print(f"  测试准确率: {test_accuracy:.4f}")
    
    # 计算混淆矩阵
    tp = ((test_predicted == 1) & (y_test_tensor == 1)).sum().item()
    tn = ((test_predicted == 0) & (y_test_tensor == 0)).sum().item()
    fp = ((test_predicted == 1) & (y_test_tensor == 0)).sum().item()
    fn = ((test_predicted == 0) & (y_test_tensor == 1)).sum().item()
    
    print(f"  混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# ==================== 6. 可视化 ====================
print("\n[可视化] 保存训练曲线...")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('day1_training_curves.png')
print("  图表已保存: day1_training_curves.png")

print("\n" + "=" * 50)
print("Day 1 完成！你成功训练了第一个神经网络")
print("=" * 50)