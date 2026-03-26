"""
FedProx 联邦学习 - 自适应 μ 版本
- 只用节点0-3
- μ 根据测试损失动态调整
- 每轮 checkpoint
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import time
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)
from src.data_loader.barcelona_loader import BarcelonaEnergyLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============= LSTM模型 =============
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ============= 基础 FedProx 客户端 =============
class BaseFedProxClient:
    def __init__(self, client_id, X_train, y_train, X_val, y_val, X_test, y_test, 
                 mean, std, mu=0.01):
        self.client_id = client_id
        self.mu = mu
        self.y_mean = mean
        self.y_std = std

        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=32, shuffle=True
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=32, shuffle=False
        )
        
        self.model = LSTMPredictor().to(device)
        print(f"客户端 {client_id}: 训练y范围 [{y_train.min():.2f}, {y_train.max():.2f}], "
              f"测试y范围 [{y_test.min():.2f}, {y_test.max():.2f}], y_std={std:.4f}")

    def train(self, global_model, epochs=3, lr=0.001):
        self.model.load_state_dict(global_model.state_dict())
        global_params = [p.detach().clone() for p in global_model.parameters()]
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                
                if self.mu > 0:
                    prox_loss = 0.0
                    for local_p, global_p in zip(self.model.parameters(), global_params):
                        prox_loss += torch.sum((local_p - global_p) ** 2)
                    loss += (self.mu / 2) * prox_loss
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (epochs * len(self.train_loader))
        return self.model.state_dict(), avg_loss

    def evaluate(self, model):
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss_norm = criterion(output, y_batch)
                total_loss += loss_norm.item() * (self.y_std ** 2)
                count += 1
        
        return total_loss / count

# ============= 自适应 μ 客户端 =============
class AdaptiveFedProxClient(BaseFedProxClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_loss = float('inf')
        self.mu_history = [self.mu]
    
    def adapt_mu(self, test_loss):
        """根据测试损失动态调整 μ"""
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.mu *= 0.95  # 效果好，减小约束
        else:
            self.mu *= 1.05  # 效果差，增加约束
        
        # 限制 μ 范围 [0.0001, 1.0]
        self.mu = max(0.0001, min(1.0, self.mu))
        self.mu_history.append(self.mu)
        return self.mu

# ============= FedAvg聚合 =============
def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i][key].float() 
                                        for i in range(len(client_models))]).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# ============= 运行自适应实验 =============
def run_adaptive_experiment(clients, num_rounds=5, initial_mu=0.01):
    print(f"\n{'='*60}")
    print(f"🔬 开始实验：自适应 μ (初始 {initial_mu})")
    print(f"{'='*60}")
    
    global_model = LSTMPredictor().to(device)
    train_losses = []
    test_losses = []
    mu_values = []
    
    # 为每个客户端设置初始 μ
    for client in clients:
        client.mu = initial_mu
    
    y_mean = clients[0].y_mean
    
    for round_idx in tqdm(range(num_rounds), desc="自适应μ"):
        client_models = []
        round_train_loss = 0
        
        # 训练所有客户端
        for client in clients:
            model_state, loss = client.train(global_model, epochs=3, lr=0.001)
            client_models.append(model_state)
            round_train_loss += loss
        
        # FedAvg聚合
        global_model = fed_avg(global_model, client_models)
        
        # 评估
        test_loss = 0
        for client in clients:
            test_loss += client.evaluate(global_model)
        test_loss /= len(clients)
        
        # 自适应调整 μ（用第一个客户端的值）
        current_mu = clients[0].adapt_mu(test_loss)
        mu_values.append(current_mu)
        
        train_losses.append(round_train_loss / len(clients))
        test_losses.append(test_loss)
        
        rmse = test_loss ** 0.5
        rel_error = (rmse / y_mean) * 100
        
        print(f"\n  轮次 {round_idx+1}/{num_rounds}")
        print(f"    当前 μ = {current_mu:.6f}")
        print(f"    训练损失: {train_losses[-1]:.6f}")
        print(f"    测试损失: {test_loss:.2f}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    相对误差: {rel_error:.2f}%")
        
        # 保存 checkpoint
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"adaptive_round{round_idx+1}.pth")
        torch.save({
            'round': round_idx + 1,
            'model_state_dict': global_model.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'mu_values': mu_values
        }, checkpoint_path)
        print(f"    💾 checkpoint已保存: {checkpoint_path}")
    
    return train_losses, test_losses, mu_values

# ============= 主函数 =============
def main():
    print("\n" + "="*80)
    print("FedProx 联邦学习 - 自适应 μ 版本")
    print("="*80)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print("\n📊 加载巴塞罗那能耗数据...")
    loader = BarcelonaEnergyLoader(
        data_path=os.path.join(project_root, "data", "processed"),
        years=['2019', '2020', '2021', '2022', '2023', '2024', '2025'],
        num_nodes=4,  # 只用4个节点
        seq_length=24,
        pred_length=6,
        filter_sectors=['Residencial', 'Serveis']
    )
    
    fed_data = loader.prepare_federated_data()
    
    print("\n🤝 初始化4个自适应客户端...")
    clients = []
    for node_id, data in fed_data.items():
        client = AdaptiveFedProxClient(
            client_id=node_id,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            mean=data['mean'],
            std=data['std'],
            mu=0.01  # 初始 μ
        )
        clients.append(client)
    
    # 运行自适应实验
    train_losses, test_losses, mu_values = run_adaptive_experiment(
        clients, num_rounds=5, initial_mu=0.01
    )
    
    # 打印结果
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    
    print("\n📊 μ 值变化：")
    for i, mu in enumerate(mu_values):
        print(f"  轮次 {i+1}: {mu:.6f}")
    
    print(f"\n最终测试损失: {test_losses[-1]:.2f}")
    rmse = test_losses[-1] ** 0.5
    y_mean = clients[0].y_mean
    rel_error = (rmse / y_mean) * 100
    print(f"最终相对误差: {rel_error:.2f}%")
    
    # 保存日志
    log_dir = os.path.join(project_root, "experiments", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"adaptive_results_{timestamp}.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FedProx 自适应 μ 实验结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"初始 μ: 0.01\n")
        f.write(f"最终 μ: {mu_values[-1]:.6f}\n")
        f.write(f"最终测试损失: {test_losses[-1]:.2f}\n")
        f.write(f"最终相对误差: {rel_error:.2f}%\n")
        f.write("\nμ 变化历史:\n")
        for i, mu in enumerate(mu_values):
            f.write(f"  轮次 {i+1}: {mu:.6f}\n")
    
    print(f"\n✅ 结果日志已保存: {log_path}")
    
    # 绘制对比图
    print("\n📊 生成对比图...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#f5f5f5')
    
    # 左图：损失曲线
    ax1.set_facecolor('#ffffff')
    ax1.plot(train_losses, linewidth=2.5, color='#2563eb', label='训练损失')
    ax1.plot(test_losses, linewidth=2.5, color='#dc2626', label='测试损失')
    ax1.set_title('损失曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('通信轮次')
    ax1.set_ylabel('MSE损失')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 右图：μ 变化
    ax2.set_facecolor('#ffffff')
    ax2.plot(mu_values, linewidth=2.5, color='#059669', marker='o')
    ax2.set_title('μ 自适应变化', fontsize=14, fontweight='bold')
    ax2.set_xlabel('通信轮次')
    ax2.set_ylabel('μ 值')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img_dir = os.path.join(project_root, "results", "beautified")
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"adaptive_comparison_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()