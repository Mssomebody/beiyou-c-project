"""
FedProx 联邦学习 - 巴塞罗那真实能耗数据
对比 FedAvg，解决 Non-IID 数据异构问题
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
from tqdm import tqdm
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader.barcelona_loader import BarcelonaEnergyLoader

# ============= 设备配置 =============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============= LSTM模型定义 =============
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
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

# ============= FedProx 客户端（关键修改在这里） =============
class FedProxClient:
    def __init__(self, client_id, X_train, y_train, X_test, y_test, proximal_weight=0.01):
        """
        Args:
            proximal_weight: μ参数，控制本地模型离全局模型的远近
                             越大越"听话"，越小越自由
        """
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
        self.proximal_weight = proximal_weight
        
    def train(self, global_model, epochs=5, lr=0.01):
        """
        FedProx 训练：在普通损失上加 proximal term
        """
        # 加载全局模型
        self.model.load_state_dict(global_model.state_dict())
        
        # 保存全局模型的参数，用于计算 proximal term
        global_params = [param.clone().detach() for param in global_model.parameters()]
        
        # 本地训练
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in tqdm(range(epochs), desc=f"客户端 {self.client_id} 本地训练", leave=False):
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                output = self.model(X_batch)
                
                # 1. 正常损失
                loss = criterion(output, y_batch)
                
                # 2. FedProx 核心：proximal term
                #    (μ/2) * ||w - w^t||^2
                proximal_loss = 0.0
                for local_p, global_p in zip(self.model.parameters(), global_params):
                    proximal_loss += torch.sum((local_p - global_p) ** 2)
                
                # 总损失 = 任务损失 + proximal term
                total_loss_value = loss + (self.proximal_weight / 2) * proximal_loss
                
                total_loss_value.backward()
                optimizer.step()
                
                total_loss += loss.item()  # 记录任务损失（不含proximal）
        
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

# ============= FedAvg聚合（和原来一样） =============
def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i][key].float() for i in range(len(client_models))]).mean(0)
    
    global_model.load_state_dict(global_dict)
    return global_model

# ============= 对比实验：不同 proximal_weight 的效果 =============
def run_fedprox_experiment(proximal_weight, clients, num_rounds=30):
    """运行一次 FedProx 实验"""
    print(f"\n🔬 测试 proximal_weight = {proximal_weight}")
    
    global_model = LSTMPredictor().to(device)
    train_losses = []
    test_losses = []
    
    for round_idx in tqdm(range(num_rounds), desc=f"μ={proximal_weight} 实验", unit="轮"):
        # 所有客户端参与
        client_models = []
        round_train_loss = 0
        
        for client in clients:
            model_state, loss = client.train(global_model, epochs=3, lr=0.01)
            client_models.append(model_state)
            round_train_loss += loss
        
        # FedAvg聚合
        global_model = fed_avg(global_model, client_models)
        
        # 评估
        test_loss = 0
        for client in clients:
            test_loss += client.evaluate(global_model)
        test_loss /= len(clients)
        
        train_losses.append(round_train_loss / len(clients))
        test_losses.append(test_loss)
        
        if (round_idx + 1) % 10 == 0:
            print(f"  轮次 {round_idx+1}/{num_rounds} | 训练损失: {train_losses[-1]:.4f} | 测试损失: {test_loss:.4f}")
    
    return train_losses, test_losses

# ============= 主函数 =============
def main():
    print("="*50)
    print("FedProx 联邦学习 - 巴塞罗那真实能耗数据")
    print("="*50)
    
    # 1. 加载真实数据
    print("\n📊 加载巴塞罗那能耗数据...")
    loader = BarcelonaEnergyLoader(
        data_path="data/processed",
        years=['2019', '2020', '2021', '2022', '2023', '2024', '2025'],
        num_nodes=5,  # 用5个节点测试
        seq_length=24,
        pred_length=1,
        filter_sectors=['Residencial', 'Serveis']
    )
    
    fed_data = loader.prepare_federated_data()
    
    # 2. 创建客户端（用同一个数据，但不同 proximal_weight）
    print("\n🤝 初始化客户端...")
    base_clients = []
    for node_id, data in fed_data.items():
        client = FedProxClient(
            client_id=node_id,
            X_train=data['X_train'],
            y_train=data['y_train'].reshape(-1, 1),
            X_test=data['X_test'],
            y_test=data['y_test'].reshape(-1, 1),
            proximal_weight=0.01  # 这个会被覆盖，先随便设
        )
        base_clients.append(client)
        print(f"  客户端 {node_id} (邮编 {data['postal_code']}): {len(data['X_train'])} 样本")
    
    # 3. 对比不同 proximal_weight
    print("\n🔬 开始对比实验...")
    weights = [0.0, 0.001, 0.01, 0.1]  # 0.0 相当于 FedAvg
    results = {}
    
    for w in weights:
        # 为每个客户端设置不同的 proximal_weight
        for client in base_clients:
            client.proximal_weight = w
        
        train_losses, test_losses = run_fedprox_experiment(w, base_clients, num_rounds=30)
        results[w] = {
            'train': train_losses,
            'test': test_losses
        }
    
    # 4. 绘制对比图
    plt.figure(figsize=(14, 6))
    
    # 训练损失对比
    plt.subplot(1, 2, 1)
    for w in weights:
        label = f'μ={w}' + (" (FedAvg)" if w == 0 else "")
        plt.plot(results[w]['train'], label=label)
    plt.xlabel('联邦轮次')
    plt.ylabel('训练损失 (MSE)')
    plt.title('不同 proximal_weight 的训练损失对比')
    plt.legend()
    plt.grid(True)
    
    # 测试损失对比
    plt.subplot(1, 2, 2)
    for w in weights:
        label = f'μ={w}' + (" (FedAvg)" if w == 0 else "")
        plt.plot(results[w]['test'], label=label)
    plt.xlabel('联邦轮次')
    plt.ylabel('测试损失 (MSE)')
    plt.title('不同 proximal_weight 的测试损失对比')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fedprox_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ 对比实验完成！结果已保存: fedprox_comparison.png")
    plt.show()
    
    # 5. 输出最佳参数
    final_losses = {w: results[w]['test'][-1] for w in weights}
    best_w = min(final_losses, key=final_losses.get)
    print(f"\n🏆 最佳 proximal_weight: {best_w}, 最终测试损失: {final_losses[best_w]:.4f}")

if __name__ == "__main__":
    main()
