#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedRep: 个性化联邦学习
4G和5G共享LSTM底层，保留个性化头
创新点：解决4G/5G数据异质性问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 数据加载
# ============================================================

def load_aligned_data(data_dir, max_stations=500):
    """加载数据，取前5维公共特征"""
    station_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
    station_dirs = station_dirs[:max_stations]
    
    all_features = []
    all_targets = []
    
    for station_dir in station_dirs:
        with open(station_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        features = data['features'][:, :5]  # 取前5维公共特征
        target = data['target']
        all_features.append(features)
        all_targets.append(target)
    
    features = np.concatenate(all_features, axis=0)
    target = np.concatenate(all_targets, axis=0)
    
    return features, target


def create_dataloaders(features, target, seq_len=24, pred_len=1, 
                       batch_size=64, train_ratio=0.8):
    """创建数据加载器"""
    class TimeSeriesDataset(Dataset):
        def __init__(self, f, t):
            self.f = f
            self.t = t
        def __len__(self):
            return len(self.f) - seq_len - pred_len + 1
        def __getitem__(self, idx):
            x = self.f[idx:idx+seq_len]
            y = self.t[idx+seq_len:idx+seq_len+pred_len]
            return torch.FloatTensor(x), torch.FloatTensor(y.flatten())
    
    dataset = TimeSeriesDataset(features, target)
    n = len(dataset)
    train_size = int(n * train_ratio)
    test_size = n - train_size
    
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============================================================
# FedRep 模型
# ============================================================

class FedRepLSTM(nn.Module):
    """FedRep: 共享底层 + 个性化头"""
    
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        # 共享层（所有客户端共享）
        self.shared_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        self.shared_dropout = nn.Dropout(dropout)
        
        # 个性化头（每个客户端独立）
        self.personal_heads = nn.ModuleDict()
        
    def add_client(self, client_id):
        """添加客户端的个性化头"""
        self.personal_heads[client_id] = nn.Linear(64, 1)
        
    def forward(self, x, client_id):
        lstm_out, _ = self.shared_lstm(x)
        shared_features = self.shared_dropout(lstm_out[:, -1, :])
        out = self.personal_heads[client_id](shared_features)
        return out


# ============================================================
# 评估指标
# ============================================================

def compute_smape(y_true, y_pred):
    """计算sMAPE"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def evaluate(model, test_loader, client_id, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x, client_id)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return compute_smape(all_targets, all_preds)


# ============================================================
# 训练函数
# ============================================================

def train_single(model, client_id, train_loader, test_loader, 
                 epochs=50, lr=0.001, verbose=True):
    """单独训练（基线）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'smape': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, client_id)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            smape = evaluate(model, test_loader, client_id, device)
            history['smape'].append(smape)
            if verbose:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, sMAPE={smape:.2f}%")
    
    final_smape = evaluate(model, test_loader, client_id, device)
    return final_smape, history


def train_fedrep(model, client_loaders, client_ids, test_loaders,
                 rounds=15, local_epochs=5, lr=0.001, verbose=True):
    """FedRep训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    
    history = {'rounds': [], '4g_smape': [], '5g_smape': []}
    
    for round_num in range(1, rounds+1):
        if verbose:
            print(f"\n--- Round {round_num}/{rounds} ---")
        
        # 收集共享层更新
        shared_updates = []
        
        for client_id in client_ids:
            loader = client_loaders[client_id]
            # 本地训练（只更新共享层）
            optimizer = optim.Adam(model.shared_lstm.parameters(), lr=lr)
            
            for _ in range(local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    output = model(x, client_id)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
            
            shared_updates.append({k: v.cpu().clone() for k, v in model.shared_lstm.state_dict().items()})
        
        # 聚合共享层（FedAvg）
        avg_state = {}
        for key in shared_updates[0].keys():
            avg_state[key] = torch.stack([u[key].float() for u in shared_updates]).mean(dim=0)
        model.shared_lstm.load_state_dict(avg_state)
        
        # 更新个性化头
        for client_id in client_ids:
            optimizer = optim.Adam(model.personal_heads[client_id].parameters(), lr=lr)
            loader = client_loaders[client_id]
            for _ in range(local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    output = model(x, client_id)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
        
        # 评估
        smape_4g = evaluate(model, test_loaders['4g'], '4g', device)
        smape_5g = evaluate(model, test_loaders['5g'], '5g', device)
        history['rounds'].append(round_num)
        history['4g_smape'].append(smape_4g)
        history['5g_smape'].append(smape_5g)
        
        if verbose and round_num % 5 == 0:
            print(f"  Round {round_num}: 4G sMAPE={smape_4g:.2f}%, 5G sMAPE={smape_5g:.2f}%")
    
    return model, history


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*70)
    print("FedRep: 4G+5G 个性化联邦学习")
    print("创新点：解决4G/5G数据异质性，5G从4G学习稳定模式")
    print("="*70)
    
    data_dir = Path("D:/Desk/desk/beiyou_c_project/data/processed/tsinghua")
    
    # 加载数据
    print("\n1. 加载数据...")
    features_4g, target_4g = load_aligned_data(data_dir / '4g', max_stations=500)
    features_5g, target_5g = load_aligned_data(data_dir / '5g', max_stations=500)
    print(f"   4G: {len(features_4g)} 样本")
    print(f"   5G: {len(features_5g)} 样本")
    
    # 创建加载器
    train_4g, test_4g = create_dataloaders(features_4g, target_4g)
    train_5g, test_5g = create_dataloaders(features_5g, target_5g)
    
    client_loaders = {'4g': train_4g, '5g': train_5g}
    client_test_loaders = {'4g': test_4g, '5g': test_5g}
    
    # ============================================================
    # 实验1: 4G单独训练
    # ============================================================
    print("\n" + "="*60)
    print("实验1: 4G单独训练（基线）")
    print("="*60)
    
    model_4g = FedRepLSTM(input_dim=5, hidden_dim=64, num_layers=2)
    model_4g.add_client('4g')
    smape_4g_alone, _ = train_single(model_4g, '4g', train_4g, test_4g, epochs=30)
    print(f"\n✅ 4G单独训练 sMAPE: {smape_4g_alone:.2f}%")
    
    # ============================================================
    # 实验2: 5G单独训练
    # ============================================================
    print("\n" + "="*60)
    print("实验2: 5G单独训练（基线）")
    print("="*60)
    
    model_5g = FedRepLSTM(input_dim=5, hidden_dim=64, num_layers=2)
    model_5g.add_client('5g')
    smape_5g_alone, _ = train_single(model_5g, '5g', train_5g, test_5g, epochs=30)
    print(f"\n✅ 5G单独训练 sMAPE: {smape_5g_alone:.2f}%")
    
    # ============================================================
    # 实验3: FedRep (4G+5G协同)
    # ============================================================
    print("\n" + "="*60)
    print("实验3: FedRep (4G+5G协同)")
    print("="*60)
    print("创新点：共享LSTM底层，保留个性化头")
    print("预期：5G从4G学到稳定模式，精度提升")
    
    model_fedrep = FedRepLSTM(input_dim=5, hidden_dim=64, num_layers=2)
    model_fedrep.add_client('4g')
    model_fedrep.add_client('5g')
    
    # 预训练个性化头
    print("\n预训练个性化头...")
    train_single(model_fedrep, '4g', train_4g, test_4g, epochs=10, verbose=False)
    train_single(model_fedrep, '5g', train_5g, test_5g, epochs=10, verbose=False)
    
    # FedRep训练
    print("\nFedRep协同训练...")
    model_fedrep, history = train_fedrep(
        model_fedrep, client_loaders, ['4g', '5g'], client_test_loaders,
        rounds=15, local_epochs=5
    )
    
    # 最终评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fedrep.to(device)
    smape_4g_fedrep = evaluate(model_fedrep, test_4g, '4g', device)
    smape_5g_fedrep = evaluate(model_fedrep, test_5g, '5g', device)
    
    # ============================================================
    # 结果汇总
    # ============================================================
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    print(f"\n{'模型':<20} {'4G sMAPE':<15} {'5G sMAPE':<15}")
    print("-"*50)
    print(f"{'4G单独训练':<20} {smape_4g_alone:<15.2f} {'-':<15}")
    print(f"{'5G单独训练':<20} {'-':<15} {smape_5g_alone:<15.2f}")
    print(f"{'FedRep (个性化联邦)':<20} {smape_4g_fedrep:<15.2f} {smape_5g_fedrep:<15.2f}")
    
    print(f"\n创新点验证:")
    print(f"  ✅ 5G精度提升: {smape_5g_alone:.2f}% → {smape_5g_fedrep:.2f}% (提升 {smape_5g_alone - smape_5g_fedrep:.2f}%)")
    print(f"  ✅ 4G精度变化: {smape_4g_alone:.2f}% → {smape_4g_fedrep:.2f}% (变化 {smape_4g_fedrep - smape_4g_alone:.2f}%)")
    
    if smape_5g_fedrep < smape_5g_alone:
        print(f"\n✅ 个性化联邦学习有效！5G从4G学到稳定模式")
    else:
        print(f"\n⚠️ 需要调整参数继续优化")
    
    # 画图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    models = ['4G Alone', '5G Alone', 'FedRep 4G', 'FedRep 5G']
    values = [smape_4g_alone, smape_5g_alone, smape_4g_fedrep, smape_5g_fedrep]
    colors = ['#2E8B57', '#2E86AB', '#F4A261', '#E76F51']
    bars = plt.bar(models, values, color=colors)
    plt.ylabel('sMAPE (%)')
    plt.title('4G+5G 预测精度对比')
    plt.ylim(0, max(values) + 10)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.1f}%', ha='center', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['rounds'], history['4g_smape'], marker='o', label='4G sMAPE')
    plt.plot(history['rounds'], history['5g_smape'], marker='s', label='5G sMAPE')
    plt.xlabel('联邦轮数')
    plt.ylabel('sMAPE (%)')
    plt.title('FedRep 训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/fedrep_results.png', dpi=150)
    print(f"\n✅ 图片保存: results/fedrep_results.png")
    
    return {
        '4g_alone': smape_4g_alone,
        '5g_alone': smape_5g_alone,
        '4g_fedrep': smape_4g_fedrep,
        '5g_fedrep': smape_5g_fedrep,
        'improvement': smape_5g_alone - smape_5g_fedrep
    }


if __name__ == "__main__":
    results = main()
