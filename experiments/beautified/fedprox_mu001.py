"""
FedProx 联邦学习 - 完美复现版
- 带近端项 (proximal term)
- 正确归一化/反归一化
- 多 μ 值对比
- 每轮保存 checkpoint（可断点续训）
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

# 添加项目根目录到路径
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ============= FedProx 客户端 =============
class FedProxClient:
    def __init__(self, client_id, X_train, y_train, X_val, y_val, X_test, y_test, 
                 mean, std, mu=0.0):
        self.client_id = client_id
        self.mu = mu
        self.y_mean = mean
        self.y_std = std

        self.train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            ),
            batch_size=32, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            ),
            batch_size=32, shuffle=False
        )
        self.test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test)
            ),
            batch_size=32, shuffle=False
        )
        
        self.model = LSTMPredictor().to(device)
        
        print(f"客户端 {client_id}: 训练y范围 [{y_train.min():.2f}, {y_train.max():.2f}], "
              f"测试y范围 [{y_test.min():.2f}, {y_test.max():.2f}], "
              f"y_std={std:.4f}")

    def train(self, global_model, epochs=5, lr=0.001):
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

# ============= FedAvg聚合 =============
def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i][key].float() 
                                        for i in range(len(client_models))]).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# ============= 运行单组实验 =============
def run_single_experiment(mu, clients, num_rounds=20, resume_from=None):
    print(f"\n{'='*60}")
    print(f"🔬 开始实验：μ = {mu}" + (" (FedAvg)" if mu == 0 else ""))
    print(f"{'='*60}")
    
    global_model = LSTMPredictor().to(device)
    train_losses = []
    test_losses = []
    start_round = 0
    
    if resume_from and os.path.exists(resume_from):
        print(f"📂 加载 checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        global_model.load_state_dict(checkpoint['model_state_dict'])
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        start_round = checkpoint['round']
        print(f"   从第 {start_round} 轮继续训练")
    
    y_mean = clients[0].y_mean
    
    for round_idx in tqdm(range(start_round, num_rounds), desc=f"μ={mu}", initial=start_round):
        print(f"\n🔥 开始第 {round_idx+1} 轮训练")
        client_models = []
        round_train_loss = 0
        
        for client in clients:
            client.mu = mu
            model_state, loss = client.train(global_model, epochs=3, lr=0.001)
            client_models.append(model_state)
            round_train_loss += loss
        
        global_model = fed_avg(global_model, client_models)
        
        test_loss = 0
        for client in clients:
            test_loss += client.evaluate(global_model)
        test_loss /= len(clients)
        
        train_losses.append(round_train_loss / len(clients))
        test_losses.append(test_loss)
        
        rmse = test_loss ** 0.5
        rel_error = (rmse / y_mean) * 100
        
        print(f"\n  轮次 {round_idx+1}/{num_rounds}")
        print(f"    训练损失: {train_losses[-1]:.6f}")
        print(f"    测试损失: {test_loss:.2f}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    相对误差: {rel_error:.2f}%")
        
        if test_loss > 1e6:
            print(f"    ⚠️ 警告：测试损失异常大 ({test_loss:.2e})")
        
        # 每轮都保存 checkpoint
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"fedprox_mu{mu}_round{round_idx+1}.pth"
        )
        torch.save({
            'round': round_idx + 1,
            'model_state_dict': global_model.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'mu': mu
        }, checkpoint_path)
        print(f"    💾 checkpoint已保存: {checkpoint_path}")
    
    return train_losses, test_losses

# ============= 主函数 =============
def main():
    print("\n" + "="*80)
    print("FedProx 联邦学习 - 完美复现版")
    print("="*80)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print("\n📊 加载巴塞罗那能耗数据...")
    loader = BarcelonaEnergyLoader(
        data_path=os.path.join(project_root, "data", "processed"),
        years=['2019', '2020', '2021', '2022', '2023', '2024', '2025'],
        num_nodes=5,
        seq_length=24,
        pred_length=6,
        filter_sectors=['Residencial', 'Serveis']
    )
    
    fed_data = loader.prepare_federated_data()
    
    print("\n🤝 初始化5个联邦学习客户端...")
    clients = []
    for node_id, data in fed_data.items():
        client = FedProxClient(
            client_id=node_id,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            mean=data['mean'],
            std=data['std'],
            mu=0.0
        )
        clients.append(client)
    
    mus = [0.01]
    all_results = {}
    
    for mu in mus:
        # 可以指定从之前的checkpoint恢复
        resume_path = None  # 设为None表示从头开始
        # resume_path = f"checkpoints/fedprox_mu{mu}_round10.pth"  # 取消注释可恢复
        
        train_losses, test_losses = run_single_experiment(
            mu, clients, num_rounds=20, resume_from=resume_path
        )
        all_results[mu] = {
            'train': train_losses,
            'test': test_losses,
            'final_train': train_losses[-1],
            'final_test': test_losses[-1]
        }
    
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    
    result_table = []
    for mu in mus:
        final_test = all_results[mu]['final_test']
        rmse = final_test ** 0.5
        y_mean = clients[0].y_mean
        rel_error = rmse / y_mean * 100
        
        result_table.append({
            'μ': f"{mu}" + (" (FedAvg)" if mu == 0 else ""),
            '最终测试损失': f"{final_test:.4f}",
            'RMSE': f"{rmse:.2f}",
            '相对误差': f"{rel_error:.2f}%"
        })
    
    df_results = pd.DataFrame(result_table)
    print("\n" + df_results.to_string(index=False))
    
    log_dir = os.path.join(project_root, "experiments", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"fedprox_results_{timestamp}.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FedProx 实验结果\n")
        f.write("="*80 + "\n\n")
        f.write(df_results.to_string(index=False))
    
    print(f"\n✅ 结果日志已保存: {log_path}")
    
    print("\n📊 生成对比图...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#f5f5f5')
    
    colors = ['#059669', '#2563eb', '#dc2626', '#7c3aed']
    
    ax1.set_facecolor('#ffffff')
    for i, mu in enumerate(mus):
        label = f'μ={mu}' + (" (FedAvg)" if mu == 0 else "")
        ax1.plot(all_results[mu]['train'],
                 linewidth=2.5, color=colors[i], label=label)
    ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('通信轮次')
    ax1.set_ylabel('MSE损失')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_facecolor('#ffffff')
    for i, mu in enumerate(mus):
        label = f'μ={mu}' + (" (FedAvg)" if mu == 0 else "")
        ax2.plot(all_results[mu]['test'],
                 linewidth=2.5, color=colors[i], label=label)
    ax2.set_title('测试损失对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('通信轮次')
    ax2.set_ylabel('MSE损失')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    img_dir = os.path.join(project_root, "results", "beautified")
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"fedprox_comparison_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()