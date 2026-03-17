"""
FedAvg 联邦学习 - 使用巴塞罗那真实能耗数据
基于 day4_fedavg.py 修改
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader.barcelona_loader import BarcelonaEnergyLoader

# ============= 设备配置 =============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============= LSTM模型定义（和你原来的一样） =============
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# ============= 联邦学习客户端 =============
class FedClient:
    def __init__(self, client_id, X_train, y_train, X_test, y_test):
        self.client_id = client_id
        self.model = LSTMPredictor().to(device)
        self.train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train).unsqueeze(-1), 
                torch.FloatTensor(y_train)
            ), 
            batch_size=32, shuffle=True
        )
        self.test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test).unsqueeze(-1), 
                torch.FloatTensor(y_test)
            ), 
            batch_size=32, shuffle=False
        )
        
    def train(self, global_model, epochs=5, lr=0.01):
        # 加载全局模型
        self.model.load_state_dict(global_model.state_dict())
        
        # 本地训练
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        return self.model.state_dict(), total_loss / len(self.train_loader)
    
    def evaluate(self, model):
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)

# ============= FedAvg聚合 =============
def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    
    # 对每个参数层进行平均
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i][key].float() for i in range(len(client_models))]).mean(0)
    
    global_model.load_state_dict(global_dict)
    return global_model

# ============= 主训练函数 =============
def main():
    print("="*50)
    print("FedAvg联邦学习 - 巴塞罗那真实能耗数据")
    print("="*50)
    
    # 1. 加载真实数据
    print("\n📊 加载巴塞罗那能耗数据...")
    loader = BarcelonaEnergyLoader(
        data_path="data/processed",
        years=['2019', '2020', '2021', '2022', '2023', '2024', '2025'],
        num_nodes=5,  # 用5个节点测试
        seq_length=24,
        pred_length=1,
        filter_sectors=['Residencial', 'Serveis']  # 居民+服务业
    )
    
    fed_data = loader.prepare_federated_data()
    
    # 2. 创建客户端
    print("\n🤝 初始化联邦客户端...")
    clients = []
    for node_id, data in fed_data.items():
        client = FedClient(
            client_id=node_id,
            X_train=data['X_train'],
            y_train=data['y_train'].reshape(-1, 1),
            X_test=data['X_test'],
            y_test=data['y_test'].reshape(-1, 1)
        )
        clients.append(client)
        print(f"  客户端 {node_id} (邮编 {data['postal_code']}): 训练 {len(data['X_train'])} 样本")
    
    # 3. 初始化全局模型
    global_model = LSTMPredictor().to(device)
    
    # 4. 联邦训练
    print("\n🚀 开始联邦训练...")
    num_rounds = 50  # 50轮通信
    train_losses = []
    test_losses = []
    
    for round_idx in range(num_rounds):
        # 选择所有客户端参与
        selected_clients = clients
        
        # 客户端本地训练
        client_models = []
        round_train_loss = 0
        
        for client in selected_clients:
            model_state, loss = client.train(global_model, epochs=5)
            client_models.append(model_state)
            round_train_loss += loss
        
        # FedAvg聚合
        global_model = fed_avg(global_model, client_models)
        
        # 评估
        test_loss = 0
        for client in clients:
            test_loss += client.evaluate(global_model)
        test_loss /= len(clients)
        
        train_losses.append(round_train_loss / len(selected_clients))
        test_losses.append(test_loss)
        
        if (round_idx + 1) % 10 == 0:
            print(f"轮次 {round_idx+1}/{num_rounds} | 训练损失: {train_losses[-1]:.4f} | 测试损失: {test_loss:.4f}")
    
    # 5. 绘制结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('联邦轮次')
    plt.ylabel('MSE损失')
    plt.title('FedAvg训练曲线 - 巴塞罗那真实数据')
    plt.legend()
    plt.grid(True)
    
    # 保存模型
     torch.save(global_model.state_dict(), 'checkpoints/best_fedavg_barcelona.pth')
    print(f"\n✅ 训练完成！模型已保存为 best_fedavg_barcelona.pth")
    
    # 保存图片
    plt.savefig('results/day4_fedavg_barcelona_results.png', dpi=150, bbox_inches='tight')
    print("📊 结果图片已保存: day4_fedavg_barcelona_results.png")
    plt.show()

if __name__ == "__main__":
    main()