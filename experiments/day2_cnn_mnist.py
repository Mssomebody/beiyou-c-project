import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 1. 加载MNIST数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载训练集
trainset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=64,
    shuffle=True
)

# 下载测试集
testset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=64,
    shuffle=False
)

print(f'训练集大小: {len(trainset)}')
print(f'测试集大小: {len(testset)}')

# 2. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1: 1通道 -> 32通道, 3x3卷积
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 卷积层2: 32通道 -> 64通道, 3x3卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # 第一层: conv1 -> bn1 -> relu -> pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # 第二层: conv2 -> bn2 -> relu -> pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 创建模型
model = CNN().to(device)
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'总参数量: {total_params}')

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练
train_losses = []
train_accs = []
test_accs = []

print("\n开始训练...")
start_time = time.time()

for epoch in range(5):  # 训练5个epoch
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 每100个batch打印一次
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/5], Step [{i+1}/938], Loss: {loss.item():.4f}')
    
    # 计算训练准确率
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    test_accs.append(test_acc)
    
    print(f'Epoch [{epoch+1}/5] 完成:')
    print(f'  训练损失: {epoch_loss:.4f}')
    print(f'  训练准确率: {epoch_acc:.2f}%')
    print(f'  测试准确率: {test_acc:.2f}%')
    print('-' * 50)

end_time = time.time()
print(f'\n训练总时间: {end_time - start_time:.2f}秒')

# 5. 绘制训练曲线
plt.figure(figsize=(15, 5))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, 'b-o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 训练准确率
plt.subplot(1, 3, 2)
plt.plot(train_accs, 'g-o', label='Train')
plt.plot(test_accs, 'r-o', label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# 显示几个测试样本
plt.subplot(1, 3, 3)
model.eval()
images, labels = next(iter(testloader))
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 显示前9个
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('day2_cnn_results.png', dpi=150)
plt.show()

print("\n✅ 训练完成！")
print(f'最终测试准确率: {test_accs[-1]:.2f}%')