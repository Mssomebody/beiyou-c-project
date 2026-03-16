# day6_gnn_cora.py
# Phase 3 Day 1: 图神经网络基础 - Cora论文分类

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import numpy as np

# ============ 1. 加载Cora数据集 ============

print("=" * 50)
print("加载Cora数据集...")
print("=" * 50)

# 加载Cora数据集（论文引用网络）
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print(f'数据集: {dataset}')
print(f'论文数量: {dataset[0].num_nodes}')
print(f'引用关系数量: {dataset[0].num_edges}')
print(f'特征维度: {dataset.num_features}')
print(f'类别数: {dataset.num_classes}')
print(f'训练节点: {(dataset[0].train_mask).sum()}')
print(f'验证节点: {(dataset[0].val_mask).sum()}')
print(f'测试节点: {(dataset[0].test_mask).sum()}')

data = dataset[0]  # 获取第一个图

# ============ 2. 定义GCN模型 ============

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        torch.manual_seed(42)
        
        # 2层GCN
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index):
        # 第一层：GCN + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第二层：GCN
        x = self.conv2(x, edge_index)
        
        return x

# ============ 3. 初始化模型 ============

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hidden_channels=16).to(device)
data = data.to(device)

print("\n模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'总参数量: {total_params}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# ============ 4. 训练函数 ============

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # 计算各集准确率
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / int(data.train_mask.sum())
    
    val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    val_acc = int(val_correct) / int(data.val_mask.sum())
    
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / int(data.test_mask.sum())
    
    return train_acc, val_acc, test_acc

# ============ 5. 训练循环 ============

print("\n开始训练...")
print("=" * 50)

losses = []
train_accs = []
val_accs = []
test_accs = []

for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    
    losses.append(loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# ============ 6. 最终结果 ============

print("\n" + "=" * 50)
print("训练完成！")
print("=" * 50)

train_acc, val_acc, test_acc = test()
print(f'最终训练集准确率: {train_acc:.4f}')
print(f'最终验证集准确率: {val_acc:.4f}')
print(f'最终测试集准确率: {test_acc:.4f}')

# ============ 7. 可视化 ============

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
ax1 = axes[0]
ax1.plot(losses, label='Training Loss', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('GCN Training Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 准确率曲线
ax2 = axes[1]
ax2.plot(train_accs, label='Train', color='green', alpha=0.7)
ax2.plot(val_accs, label='Validation', color='orange', alpha=0.7)
ax2.plot(test_accs, label='Test', color='red', alpha=0.7)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('GCN Accuracy')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('day6_gnn_results.png', dpi=150)
plt.show()

print("\n结果图已保存: day6_gnn_results.png")

# ============ 8. 可视化节点嵌入（可选） ============

@torch.no_grad()
def visualize_embedding(hidden_channels=16):
    """可视化节点嵌入（使用t-SNE降维）"""
    model = GCN(hidden_channels=hidden_channels).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    # 获取第二层前的嵌入
    out = model.conv1(data.x, data.edge_index)
    out = F.relu(out)
    
    # 转换为numpy
    z = out.cpu().numpy()
    y = data.y.cpu().numpy()
    
    # t-SNE降维
    from sklearn.manifold import TSNE
    z_tsne = TSNE(n_components=2, random_state=42).fit_transform(z)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, cmap='tab10', s=20, alpha=0.8)
    plt.colorbar(scatter, ticks=range(7))
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('day6_node_embeddings.png', dpi=150)
    plt.show()
    print("节点嵌入图已保存: day6_node_embeddings.png")

# 如果想看嵌入可视化，取消下面的注释
# visualize_embedding()

print("\n" + "=" * 50)
print("Day 6 完成！下一步: GCN/GAT实现")
print("=" * 50)