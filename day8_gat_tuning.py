# day8_gat_tuning.py
# Phase 3 Day 3: GAT参数调优实验

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import product

# ============ 1. 加载数据 ============

print("=" * 60)
print("GAT参数调优实验")
print("=" * 60)

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

print(f'数据集: Cora')
print(f'节点数: {data.num_nodes}')
print(f'边数: {data.num_edges}')
print(f'特征维度: {dataset.num_features}')
print(f'类别数: {dataset.num_classes}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# ============ 2. 定义GAT模型（可调参数） ============

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 heads=8, dropout=0.6, concat=True):
        super().__init__()
        
        # 第一层：多头注意力
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout,
            concat=concat
        )
        
        # 第二层：输出层
        self.conv2 = GATConv(
            hidden_channels * heads if concat else hidden_channels, 
            out_channels, 
            heads=1,
            concat=False,
            dropout=dropout
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ============ 3. 训练函数 ============

def train_model(model, data, epochs=200, lr=0.005, weight_decay=5e-4):
    """训练单个模型并返回最佳测试准确率"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
            val_acc = int(val_correct) / int(data.val_mask.sum())
            
            test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            test_acc = int(test_correct) / int(data.test_mask.sum())
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
    
    return best_val_acc, best_test_acc, best_epoch

# ============ 4. 参数网格搜索 ============

print("\n开始参数网格搜索...")
print("=" * 60)

# 定义要搜索的参数
param_grid = {
    'hidden_channels': [4, 8, 16],
    'heads': [1, 4, 8],
    'dropout': [0.3, 0.6, 0.8],
    'lr': [0.005, 0.01]
}

# 生成所有参数组合
keys = param_grid.keys()
values = param_grid.values()
experiments = [dict(zip(keys, v)) for v in product(*values)]

print(f"总实验数: {len(experiments)}")
print("\n参数空间:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# 存储结果
results = []

for i, params in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] 实验参数: {params}")
    
    # 创建模型
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=params['hidden_channels'],
        out_channels=dataset.num_classes,
        heads=params['heads'],
        dropout=params['dropout']
    )
    
    # 训练
    start_time = time.time()
    val_acc, test_acc, best_epoch = train_model(
        model, data, 
        epochs=200,
        lr=params['lr']
    )
    train_time = time.time() - start_time
    
    # 记录结果
    result = {
        **params,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'time': train_time
    }
    results.append(result)
    
    print(f"  验证准确率: {val_acc:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  最佳轮次: {best_epoch}")
    print(f"  训练时间: {train_time:.2f}秒")

# ============ 5. 结果分析 ============

print("\n" + "=" * 60)
print("实验结果汇总")
print("=" * 60)

# 按测试准确率排序
results.sort(key=lambda x: x['test_acc'], reverse=True)

print("\nTop 5 最佳配置:")
print("-" * 60)
print(f"{'排名':<4} {'hidden':<6} {'heads':<6} {'dropout':<8} {'lr':<6} {'test_acc':<10} {'val_acc':<10}")
print("-" * 60)

for i, r in enumerate(results[:5]):
    print(f"{i+1:<4} {r['hidden_channels']:<6} {r['heads']:<6} {r['dropout']:<8} "
          f"{r['lr']:<6} {r['test_acc']:.4f}    {r['val_acc']:.4f}")

# ============ 6. 可视化分析 ============

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 不同head数的影响
ax1 = axes[0, 0]
head_data = {}
for r in results:
    heads = r['heads']
    if heads not in head_data:
        head_data[heads] = []
    head_data[heads].append(r['test_acc'])

head_means = {h: np.mean(accs) for h, accs in head_data.items()}
head_stds = {h: np.std(accs) for h, accs in head_data.items()}

heads = list(head_means.keys())
means = list(head_means.values())
stds = list(head_stds.values())

ax1.bar(heads, means, yerr=stds, capsize=5, color='skyblue', edgecolor='navy')
ax1.set_xlabel('Attention Heads')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Effect of Attention Heads')
ax1.set_ylim([0.75, 0.85])

# 2. 不同hidden维度的影响
ax2 = axes[0, 1]
hidden_data = {}
for r in results:
    h = r['hidden_channels']
    if h not in hidden_data:
        hidden_data[h] = []
    hidden_data[h].append(r['test_acc'])

hidden_means = {h: np.mean(accs) for h, accs in hidden_data.items()}
hidden_stds = {h: np.std(accs) for h, accs in hidden_data.items()}

hiddens = list(hidden_means.keys())
means = list(hidden_means.values())
stds = list(hidden_stds.values())

ax2.bar(hiddens, means, yerr=stds, capsize=5, color='lightcoral', edgecolor='darkred')
ax2.set_xlabel('Hidden Dimension')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Effect of Hidden Dimension')

# 3. 不同dropout的影响
ax3 = axes[0, 2]
dropout_data = {}
for r in results:
    d = r['dropout']
    if d not in dropout_data:
        dropout_data[d] = []
    dropout_data[d].append(r['test_acc'])

dropout_means = {d: np.mean(accs) for d, accs in dropout_data.items()}
dropout_stds = {d: np.std(accs) for d, accs in dropout_data.items()}

dropouts = list(dropout_means.keys())
means = list(dropout_means.values())
stds = list(dropout_stds.values())

ax3.bar([str(d) for d in dropouts], means, yerr=stds, capsize=5, color='lightgreen', edgecolor='darkgreen')
ax3.set_xlabel('Dropout Rate')
ax3.set_ylabel('Test Accuracy')
ax3.set_title('Effect of Dropout')

# 4. 学习率影响
ax4 = axes[1, 0]
lr_data = {}
for r in results:
    l = r['lr']
    if l not in lr_data:
        lr_data[l] = []
    lr_data[l].append(r['test_acc'])

lr_means = {l: np.mean(accs) for l, accs in lr_data.items()}
lr_stds = {l: np.std(accs) for l, accs in lr_data.items()}

lrs = list(lr_means.keys())
means = list(lr_means.values())
stds = list(lr_stds.values())

ax4.bar([str(l) for l in lrs], means, yerr=stds, capsize=5, color='gold', edgecolor='orange')
ax4.set_xlabel('Learning Rate')
ax4.set_ylabel('Test Accuracy')
ax4.set_title('Effect of Learning Rate')

# 5. 最佳参数组合的收敛曲线（重新训练最佳模型）
ax5 = axes[1, 1]
best_params = results[0]  # 最佳配置
print(f"\n重新训练最佳配置: {best_params}")

# 重新训练最佳模型并记录曲线
model = GAT(
    in_channels=dataset.num_features,
    hidden_channels=best_params['hidden_channels'],
    out_channels=dataset.num_classes,
    heads=best_params['heads'],
    dropout=best_params['dropout']
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

train_losses = []
val_accs = []
test_accs = []

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # 评估
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        val_acc = int(val_correct) / int(data.val_mask.sum())
        val_accs.append(val_acc)
        
        test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(test_correct) / int(data.test_mask.sum())
        test_accs.append(test_acc)

ax5.plot(train_losses, label='Loss', color='blue')
ax5_twin = ax5.twinx()
ax5_twin.plot(val_accs, label='Val Acc', color='orange', alpha=0.7)
ax5_twin.plot(test_accs, label='Test Acc', color='red', alpha=0.7)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Loss', color='blue')
ax5_twin.set_ylabel('Accuracy', color='red')
ax5.set_title(f"Best Config: h={best_params['hidden_channels']}, heads={best_params['heads']}")
ax5.legend(loc='upper left')
ax5_twin.legend(loc='upper right')

# 6. 所有实验结果分布
ax6 = axes[1, 2]
all_tests = [r['test_acc'] for r in results]
ax6.hist(all_tests, bins=10, edgecolor='black', alpha=0.7, color='purple')
ax6.axvline(x=max(all_tests), color='red', linestyle='--', label=f"Best: {max(all_tests):.4f}")
ax6.axvline(x=np.mean(all_tests), color='blue', linestyle='--', label=f"Mean: {np.mean(all_tests):.4f}")
ax6.set_xlabel('Test Accuracy')
ax6.set_ylabel('Frequency')
ax6.set_title('Distribution of Results')
ax6.legend()

plt.tight_layout()
plt.savefig('day8_gat_tuning.png', dpi=150)
plt.show()

print("\n结果图已保存: day8_gat_tuning.png")

# ============ 7. 保存最佳模型 ============

print("\n" + "=" * 60)
print("保存最佳模型")
print("=" * 60)

best_config = results[0]
print(f"最佳配置:")
print(f"  hidden_channels: {best_config['hidden_channels']}")
print(f"  heads: {best_config['heads']}")
print(f"  dropout: {best_config['dropout']}")
print(f"  lr: {best_config['lr']}")
print(f"  验证准确率: {best_config['val_acc']:.4f}")
print(f"  测试准确率: {best_config['test_acc']:.4f}")

# 保存最佳配置
import json
with open('best_gat_config.json', 'w') as f:
    json.dump(best_config, f, indent=4)

print("\n最佳配置已保存: best_gat_config.json")

print("\n" + "=" * 60)
print("Day 8 完成！下一步: 图同构网络(GIN)")
print("=" * 60)