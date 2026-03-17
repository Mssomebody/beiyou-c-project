"""
Day 1 - PyTorch实现MLP解决XOR问题
对比NumPy手动实现，展示PyTorch自动求导的便利性
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据（和NumPy版完全一样）
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print("="*50)
print("XOR问题数据集")
print("输入:\n", X.numpy())
print("目标输出:\n", y.numpy())
print("="*50)

# 2. 定义模型（结构和NumPy版一致：2-4-1）
class XORMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层：输入2维 -> 输出4维
        self.hidden = nn.Linear(2, 4)
        # 输出层：输入4维 -> 输出1维
        self.output = nn.Linear(4, 1)
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 前向传播（和你手推的公式一模一样）
        x = self.sigmoid(self.hidden(x))  # h = sigmoid(W1·X + b1)
        x = self.sigmoid(self.output(x))  # out = sigmoid(W2·h + b2)
        return x

# 3. 初始化模型、损失函数、优化器
model = XORMLP()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.5)  # 随机梯度下降

# 打印模型结构
print("\n模型结构:")
print(model)
print("\n可训练参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 4. 训练循环
print("\n开始训练...")
losses = []

for epoch in range(10000):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # 反向传播（PyTorch自动计算梯度！）
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()        # 自动反向传播
    optimizer.step()       # 更新权重
    
    # 记录损失
    losses.append(loss.item())
    
    # 每2000轮打印一次
    if epoch % 2000 == 0:
        print(f'Epoch {epoch:5d}, Loss: {loss.item():.6f}')

# 5. 测试结果
print("\n" + "="*50)
print("训练完成！")
with torch.no_grad():  # 测试时不计算梯度
    y_pred = model(X)
    y_pred_binary = (y_pred > 0.5).float()  # 转为0/1
    
    print("\n预测结果对比:")
    print("输入  | 目标 | 预测(概率) | 预测(二值)")
    print("-" * 40)
    for i in range(4):
        print(f"{X[i].numpy()} |   {int(y[i].item())}   |   {y_pred[i].item():.3f}   |     {int(y_pred_binary[i].item())}")
    
    accuracy = (y_pred_binary == y).float().mean().item()
    print("-" * 40)
    print(f"准确率: {accuracy*100:.0f}%")

# 6. 绘制训练曲线
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(losses[:100])  # 前100轮细节
plt.title('Loss Curve (First 100 epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('day1_training_curves.png', dpi=150)
plt.show()

print("\n✅ 训练曲线已保存为: day1_training_curves.png")

# 7. 对比NumPy和PyTorch
print("\n" + "="*50)
print("NumPy vs PyTorch 实现对比")
print("="*50)
print("1. 代码量: PyTorch约50行 vs NumPy约80行")
print("2. 反向传播: PyTorch自动求导 vs NumPy手动推导")
print("3. 扩展性: PyTorch易扩展CNN/RNN vs NumPy需重写")
print("4. 速度: PyTorch GPU加速 vs NumPy CPU only")
print("5. 理解深度: NumPy手动实现更理解原理 ✅")
print("="*50)
print("📌 结论: NumPy理解原理，PyTorch实际应用，两者都重要")