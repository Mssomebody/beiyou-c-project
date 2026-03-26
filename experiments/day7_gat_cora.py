import time
timestamp = time.strftime("%Y%m%d_%H%M%S")
# day7_gat_cora.py
# Phase 3 Day 2: 图注意力网络(GAT)实现 - Cora论文分类

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import numpy as np
import time

# ============ 1. 加载Cora数据集 ============

print("=" * 50)
print("加载Cora数据集...")
print("=" * 50)

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

# ============ 2. 定义GAT模型 ============

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels=8, heads=8, dropout=0.6):
        super().__init__()
        torch.manual_seed(42)
        
        # 第一层GAT：多头注意力
        self.conv1 = GATConv(
            dataset.num_features, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout
        )
        
        # 第二层GAT：输出层（单头）
        self.conv2 = GATConv(
            hidden_channels * heads, 
            dataset.num_classes, 
            heads=1,
            concat=False,
            dropout=dropout
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # 第一层：GAT + ELU + Dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层：GAT
        x = self.conv2(x, edge_index)
        
        return x

# ============ 3. 初始化模型 ============

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(hidden_channels=8, heads=8, dropout=0.6).to(device)
data = data.to(device)

print("\n模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'总参数量: {total_params}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
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

print("\n开始训练GAT...")
print("=" * 50)

losses = []
train_accs = []
val_accs = []
test_accs = []
best_val_acc = 0
best_test_acc = 0
best_epoch = 0

start_time = time.time()

for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    
    losses.append(loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_epoch = epoch
        torch.save(model.state_dict(), "checkpoints/best_gat_model.pth")
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

end_time = time.time()

# ============ 6. 最终结果 ============

print("\n" + "=" * 50)
print("训练完成！")
print("=" * 50)

# 加载最佳模型
model.load_state_dict(torch.load("checkpoints/best_gat_model.pth"))
train_acc, val_acc, test_acc = test()

print(f'训练时间: {end_time - start_time:.2f}秒')
print(f'最佳模型 (Epoch {best_epoch}):')
print(f'  验证集准确率: {best_val_acc:.4f}')
print(f'  测试集准确率: {best_test_acc:.4f}')
print(f'最终测试集准确率: {test_acc:.4f}')

# ============ 7. 与GCN对比 ============

print("\n" + "=" * 50)
print("GAT vs GCN 对比")
print("=" * 50)

# GCN结果（从Day6获取）
gcn_test = 0.799  # 您的Day6结果

print(f'GCN 测试准确率: {gcn_test:.4f}')
print(f'GAT 测试准确率: {test_acc:.4f}')
print(f'GAT 提升: +{(test_acc - gcn_test)*100:.2f}%')

# ============ 8. 可视化 ============

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 损失曲线
ax1 = axes[0, 0]
ax1.plot(losses, label='Training Loss', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('GAT Training Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 准确率曲线
ax2 = axes[0, 1]
ax2.plot(train_accs, label='Train', color='green', alpha=0.7)
ax2.plot(val_accs, label='Validation', color='orange', alpha=0.7)
ax2.plot(test_accs, label='Test', color='red', alpha=0.7)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('GAT Accuracy')
ax2.grid(True, alpha=0.3)
ax2.legend()

# GAT vs GCN 对比柱状图
ax3 = axes[1, 0]
models = ['GCN', 'GAT']
accuracies = [gcn_test, test_acc]
colors = ['#ff9999', '#66b3ff']
bars = ax3.bar(models, accuracies, color=colors)
ax3.set_ylabel('Test Accuracy')
ax3.set_title('GAT vs GCN Performance')
ax3.set_ylim([0.7, 0.9])

# 在柱状图上显示数值
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.4f}', ha='center', va='bottom')

# 注意力头数影响分析
ax4 = axes[1, 1]
head_options = [1, 4, 8, 16]
# 模拟不同头数的效果（实际应该跑多次实验，这里用示意）
gat_performance = [0.75, 0.78, 0.80, 0.79]  # 示意数据
ax4.plot(head_options, gat_performance, 'o-', color='purple')
ax4.set_xlabel('Number of Attention Heads')
ax4.set_ylabel('Test Accuracy')
ax4.set_title('Effect of Attention Heads (Illustrative)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/beautified/day7_gat_results_{timestamp}.png', dpi=150)
print(f"结果图已保存: results/beautified/day7_gat_results_{timestamp}.png")
plt.show()

print("\n结果图已保存: results/beautified/{timestamp}.png")

# ============ 9. 可视化注意力权重（可选） ============

def visualize_attention():
    """可视化第一层的注意力权重（简化版）"""
    from torch_geometric.utils import to_networkx
    import networkx as nx
    
    # 获取注意力权重
    model.eval()
    with torch.no_grad():
        # 前向传播时获取注意力系数
        x = data.x
        edge_index = data.edge_index
        
        # 第一层GAT的注意力权重
        x, (edge_index, att_weights) = model.conv1(x, edge_index, return_attention_weights=True)
        
        # 将注意力权重转移到CPU
        att_weights = att_weights.cpu().numpy()
        
        # 创建图
        G = to_networkx(data, to_undirected=True)
        
        # 取前100个节点可视化
        plt.figure(figsize=(12, 10))
        
        # 选择一部分边来显示（避免太密集）
        num_edges = min(500, len(att_weights))
        sample_indices = np.random.choice(len(att_weights), num_edges, replace=False)
        
        # 归一化注意力权重用于边的颜色
        edge_colors = att_weights[sample_indices]
        
        # 绘制图
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, edgelist=[(edge_index[0][i].item(), edge_index[1][i].item()) 
                                                  for i in sample_indices],
                                edge_color=edge_colors, edge_cmap=plt.cm.Reds, width=2)
        plt.title('Attention Weights Visualization')
        plt.axis('off')
        plt.savefig('results/beautified/day7_attention_weights_{timestamp}.png', dpi=150)
        plt.show()
        print("注意力权重图已保存: results/beautified/day7_attention_weights_{timestamp}.png")

# 取消注释查看注意力可视化
# visualize_attention()

print("\n" + "=" * 50)
print("Day 7 完成！下一步: GAT参数调优")
print("=" * 50)