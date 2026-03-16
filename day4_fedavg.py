# day4_fedavg.py
# Day 4: FedAvg 联邦学习算法实现

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ============ 1. 定义LSTM模型 ============

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

# ============ 2. 加载联邦数据 ============

def load_federated_data(data_dir='fl_data', n_sites=5):
    """加载各站点的本地数据"""
    site_data = []
    
    print("=" * 50)
    print("加载联邦学习数据")
    print("=" * 50)
    
    for site_id in range(n_sites):
        site_dir = os.path.join(data_dir, f'site_{site_id}')
        
        X_train = torch.FloatTensor(np.load(os.path.join(site_dir, 'X_train.npy')))
        y_train = torch.FloatTensor(np.load(os.path.join(site_dir, 'y_train.npy'))).reshape(-1, 1)
        X_test = torch.FloatTensor(np.load(os.path.join(site_dir, 'X_test.npy')))
        y_test = torch.FloatTensor(np.load(os.path.join(site_dir, 'y_test.npy'))).reshape(-1, 1)
        
        site_data.append({
            'site_id': site_id,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
        print(f"站点 {site_id}: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
    
    # 加载全局测试集
    global_dir = os.path.join(data_dir, 'global_test')
    X_global = torch.FloatTensor(np.load(os.path.join(global_dir, 'X_test.npy')))
    y_global = torch.FloatTensor(np.load(os.path.join(global_dir, 'y_test.npy'))).reshape(-1, 1)
    
    print(f"\n全局测试集: {len(X_global)} 样本")
    
    return site_data, (X_global, y_global)

# ============ 3. 本地训练函数 ============

def train_local(model, X_train, y_train, epochs=5, lr=0.001):
    """在单个站点上训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return model, losses

# ============ 4. 评估函数 ============

def evaluate(model, X_test, y_test):
    """计算模型在测试集上的MSE和MAE"""
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mse = nn.MSELoss()(pred, y_test).item()
        mae = torch.mean(torch.abs(pred - y_test)).item()
    return mse, mae

# ============ 5. FedAvg 聚合 ============

def fedavg_aggregate(local_models, weights):
    """联邦平均聚合"""
    global_model = copy.deepcopy(local_models[0])
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = sum(w * m.state_dict()[key] for w, m in zip(weights, local_models))
    
    global_model.load_state_dict(global_dict)
    return global_model

# ============ 6. FedAvg 主流程 ============

def run_fedavg(site_data, global_test, n_rounds=10, local_epochs=5):
    """运行FedAvg算法"""
    X_global, y_global = global_test
    
    # 计算各站点权重（按样本数）
    total_samples = sum([s['n_train'] for s in site_data])
    site_weights = [s['n_train'] / total_samples for s in site_data]
    
    print("\n" + "=" * 50)
    print("FedAvg 联邦学习")
    print("=" * 50)
    print(f"通信轮数: {n_rounds}, 本地训练轮数: {local_epochs}")
    print(f"站点权重: {[round(w, 3) for w in site_weights]}")
    
    # 初始化全局模型
    global_model = LSTMPredictor(input_dim=3, hidden_dim=64)
    
    # 记录训练过程
    global_maes = []
    site_maes = [[] for _ in range(len(site_data))]
    
    for round_idx in range(n_rounds):
        print(f"\n--- 第 {round_idx+1}/{n_rounds} 轮 ---")
        
        # 分发全局模型到各站点
        local_models = []
        
        for site_idx, site in enumerate(site_data):
            # 复制全局模型
            local_model = copy.deepcopy(global_model)
            
            # 本地训练
            X_train, y_train = site['X_train'], site['y_train']
            local_model, _ = train_local(local_model, X_train, y_train, epochs=local_epochs)
            local_models.append(local_model)
            
            # 评估本地模型
            _, mae = evaluate(local_model, site['X_test'], site['y_test'])
            site_maes[site_idx].append(mae)
        
        # FedAvg聚合
        global_model = fedavg_aggregate(local_models, site_weights)
        
        # 评估全局模型
        _, mae = evaluate(global_model, X_global, y_global)
        global_maes.append(mae)
        print(f"  全局模型 MAE: {mae:.4f}")
    
    return global_model, global_maes, site_maes

# ============ 7. 本地独立训练（基线） ============

def train_baselines(site_data, global_test, epochs=50):
    """训练本地独立模型作为对比基线"""
    X_global, y_global = global_test
    local_models = []
    local_maes = []
    
    print("\n" + "=" * 50)
    print("本地独立训练（基线）")
    print("=" * 50)
    
    for site_idx, site in enumerate(site_data):
        print(f"\n--- 站点 {site_idx} ---")
        model = LSTMPredictor(input_dim=3, hidden_dim=64)
        X_train, y_train = site['X_train'], site['y_train']
        
        # 训练
        trained_model, losses = train_local(model, X_train, y_train, epochs=epochs)
        local_models.append(trained_model)
        
        # 评估
        _, mae = evaluate(trained_model, site['X_test'], site['y_test'])
        local_maes.append(mae)
        print(f"  本地测试 MAE: {mae:.4f}")
    
    # 评估本地模型在全局测试集的表现
    global_maes = []
    for idx, model in enumerate(local_models):
        _, mae = evaluate(model, X_global, y_global)
        global_maes.append(mae)
        print(f"站点 {idx} 模型在全局测试集 MAE: {mae:.4f}")
    
    return local_models, local_maes, global_maes

# ============ 8. 可视化对比 ============

def plot_results(global_maes, site_maes, baseline_local, baseline_global):
    """可视化对比 FedAvg vs 本地独立模型"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 图1: FedAvg 收敛曲线
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(global_maes)+1), global_maes, 'b-o', linewidth=2)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('MAE')
    ax1.set_title('FedAvg: Global Model Performance')
    ax1.grid(True)
    
    # 图2: 各站点在FedAvg中的表现
    ax2 = axes[0, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, site_mae in enumerate(site_maes):
        ax2.plot(range(1, len(site_mae)+1), site_mae, color=colors[i], 
                linestyle='--', marker='o', label=f'Site {i}', alpha=0.7)
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('MAE')
    ax2.set_title('FedAvg: Per-Site Performance')
    ax2.legend()
    ax2.grid(True)
    
    # 图3: 本地测试对比
    ax3 = axes[1, 0]
    x_pos = np.arange(len(baseline_local))
    width = 0.35
    
    ax3.bar(x_pos - width/2, baseline_local, width, label='Local Models', color='orange', alpha=0.8)
    ax3.bar(x_pos + width/2, [site_maes[i][-1] for i in range(len(baseline_local))], 
            width, label=f'FedAvg (Round {len(global_maes)})', color='blue', alpha=0.8)
    ax3.set_xlabel('Site')
    ax3.set_ylabel('MAE on Local Test Set')
    ax3.set_title('Local Test Performance: Local vs FedAvg')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Site {i}' for i in range(len(baseline_local))])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 全局测试对比
    ax4 = axes[1, 1]
    ax4.bar(x_pos, baseline_global, color='orange', alpha=0.8, label='Local Models')
    ax4.axhline(y=global_maes[-1], color='blue', linestyle='--', linewidth=2,
                label=f'FedAvg Global ({global_maes[-1]:.4f})')
    ax4.set_xlabel('Site')
    ax4.set_ylabel('MAE on Global Test Set')
    ax4.set_title('Global Test Performance: Local vs FedAvg')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Site {i}' for i in range(len(baseline_local))])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day4_fedavg_results.png', dpi=150)
    plt.show()
    print("\n结果图已保存: day4_fedavg_results.png")

# ============ 9. 主程序 ============

if __name__ == "__main__":
    # 1. 加载数据
    site_data, global_test = load_federated_data(data_dir='fl_data', n_sites=5)
    
    # 2. 训练本地独立模型（基线）
    local_models, local_maes, local_global_maes = train_baselines(site_data, global_test, epochs=50)
    
    # 3. 运行 FedAvg
    global_model, global_maes, site_maes = run_fedavg(
        site_data, global_test, 
        n_rounds=10,     # 通信轮数
        local_epochs=5   # 每轮本地训练轮数
    )
    
    # 4. 打印最终结果
    print("\n" + "=" * 50)
    print("最终结果对比")
    print("=" * 50)
    print(f"本地独立模型平均 MAE (本地测试): {np.mean(local_maes):.4f}")
    print(f"FedAvg 最终轮全局模型 MAE: {global_maes[-1]:.4f}")
    print(f"FedAvg 各站点平均 MAE (最后一轮): {np.mean([m[-1] for m in site_maes]):.4f}")
    
    # 5. 可视化
    plot_results(global_maes, site_maes, local_maes, local_global_maes)
    
    print("\n" + "=" * 50)
    print("Day 4 完成！FedAvg 算法实现成功")
    print("=" * 50)