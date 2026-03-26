"""
FedAvg 联邦学习 - 真实损失版
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
import time

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

# 添加项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)
from src.data_loader.barcelona_loader import BarcelonaEnergyLoader

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# LSTM模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 客户端
class FedClient:
    def __init__(self, client_id, X_train, y_train, X_test, y_test):
        self.client_id = client_id
        self.model = LSTMPredictor().to(device)
        
        # 保存原始测试数据的最大值（用于反标准化预测值）
        self.scale = y_test.max()
        print(f"  客户端 {client_id} scale: {self.scale:.2f}")
        
        # 转换为tensor
        self.train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train).unsqueeze(-1), 
                torch.FloatTensor(y_train).unsqueeze(-1)
            ), 
            batch_size=32, shuffle=True
        )
        self.test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test).unsqueeze(-1), 
                torch.FloatTensor(y_test).unsqueeze(-1)
            ), 
            batch_size=32, shuffle=False
        )
    
    def train(self, global_model, epochs=3, lr=0.001):
        self.model.load_state_dict(global_model.state_dict())
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0
            for X, y in self.train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_losses.append(total_loss / len(self.train_loader))
        
        return self.model.state_dict(), epoch_losses[-1]
    
    def evaluate(self, model):
        model.eval()
        preds, truths = [], []
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                
                preds.extend(output.cpu().numpy().flatten())
                truths.extend(y.cpu().numpy().flatten())
        
        # 反标准化预测值
        preds_original = np.array(preds) * self.scale
        truths_original = np.array(truths)
        
        # 🔥 计算真实尺度的MSE
        mse = np.mean((preds_original - truths_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_original - truths_original))
        
        return mse, {
            'pred_mean': np.mean(preds_original),
            'pred_std': np.std(preds_original),
            'actual_mean': np.mean(truths_original),
            'actual_std': np.std(truths_original),
            'pred_range': [np.min(preds_original), np.max(preds_original)],
            'actual_range': [np.min(truths_original), np.max(truths_original)],
            'rmse': rmse,
            'mae': mae
        }

# FedAvg聚合
def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.stack([cm[key] for cm in client_models]).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def main():
    print("="*60)
    print("FedAvg联邦学习 - 真实损失版")
    print("="*60)
    
    # 1. 加载真实数据
    print("\n📊 加载巴塞罗那能耗数据...")
    loader = BarcelonaEnergyLoader(
        data_path="data/processed",
        years=['2019', '2020', '2021', '2022', '2023'],
        num_nodes=5,
        seq_length=24,
        pred_length=1,
        filter_sectors=['Residencial', 'Serveis']
    )
    
    fed_data = loader.prepare_federated_data()
    
    # 2. 直接使用原始数据
    print("\n📈 原始数据范围:")
    clients = []
    
    for node_id, data in fed_data.items():
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        print(f"\n  节点 {node_id}:")
        print(f"    训练样本: {len(X_train)}")
        print(f"    训练y范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
        print(f"    训练y均值: {y_train.mean():.2f}")
        print(f"    测试y范围: [{y_test.min():.2f}, {y_test.max():.2f}]")
        print(f"    测试y均值: {y_test.mean():.2f}")
        
        # 创建客户端
        client = FedClient(
            client_id=node_id,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        clients.append(client)
    
    # 3. 初始化全局模型
    global_model = LSTMPredictor().to(device)
    
    # 4. 联邦训练
    print("\n🚀 开始联邦训练...")
    num_rounds = 20
    train_losses = []  # 归一化损失
    test_losses = []    # 真实尺度MSE
    test_rmses = []     # 真实尺度RMSE
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        
        # 客户端训练
        client_models = []
        round_train_loss = 0
        
        for client in clients:
            model_state, train_loss = client.train(global_model, epochs=3, lr=0.001)
            client_models.append(model_state)
            round_train_loss += train_loss
        
        # 聚合
        global_model = fed_avg(global_model, client_models)
        
        # 评估
        test_loss_sum = 0
        all_stats = []
        for client in clients:
            loss, stats = client.evaluate(global_model)
            test_loss_sum += loss
            all_stats.append(stats)
        
        avg_test_loss = test_loss_sum / len(clients)
        avg_train_loss = round_train_loss / len(clients)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # 输出
        round_time = time.time() - round_start
        print(f"\n🔄 Round {round_idx+1:2d}/{num_rounds} | 耗时: {round_time:.1f}s")
        print(f"   归一化损失: {avg_train_loss:.4f}")
        
        if all_stats:
            s = all_stats[0]
            test_rmses.append(s['rmse'])
            print(f"   真实尺度MSE: {avg_test_loss:.2f}  | RMSE: {s['rmse']:.2f} | MAE: {s['mae']:.2f}")
            print(f"   预测均值: {s['pred_mean']:.2f} | 实际均值: {s['actual_mean']:.2f}")
            print(f"   预测范围: [{s['pred_range'][0]:.2f}, {s['pred_range'][1]:.2f}]")
            print(f"   实际范围: [{s['actual_range'][0]:.2f}, {s['actual_range'][1]:.2f}]")
    
    # 5. 保存结果
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    # 绘制损失曲线
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 左图：归一化损失
    axes[0].plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Communication Rounds')
    axes[0].set_ylabel('Normalized MSE')
    axes[0].set_title('Normalized Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：真实RMSE
    axes[1].plot(test_rmses, 'r-', label='Test RMSE', linewidth=2)
    axes[1].set_xlabel('Communication Rounds')
    axes[1].set_ylabel('RMSE (kWh)')
    axes[1].set_title('Real-scale RMSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    save_path = os.path.join(project_root, "results", "fedavg_barcelona.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 图片已保存: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()