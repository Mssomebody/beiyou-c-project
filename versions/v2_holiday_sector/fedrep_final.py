"""
FedRep: 4G+5G 个性化联邦学习
使用正确预处理的数据，评估时用原始能耗计算 sMAPE
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 配置
# ============================================================

DATA_DIR = Path("D:/Desk/desk/beiyou_c_project/data/processed/tsinghua_v2")
MAX_STATIONS = 500  # 测试用，完整跑设 None


# ============================================================
# 数据加载
# ============================================================

def load_stations(data_type, max_stations=None):
    """加载所有基站数据"""
    data_dir = DATA_DIR / data_type
    stations = []
    
    for station_dir in data_dir.iterdir():
        if station_dir.is_dir():
            with open(station_dir / 'data.pkl', 'rb') as f:
                data = pickle.load(f)
            stations.append(data)
    
    if max_stations:
        stations = stations[:max_stations]
    
    return stations


def create_sequences(features, target, seq_len=24, pred_len=1):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(len(features) - seq_len - pred_len + 1):
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y).flatten()


# ============================================================
# 数据集
# ============================================================

class StationDataset(Dataset):
    def __init__(self, features_norm, target_norm, seq_len=24, pred_len=1):
        self.X, self.y = create_sequences(features_norm, target_norm, seq_len, pred_len)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])


# ============================================================
# FedRep 模型
# ============================================================

class FedRepLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.shared_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.shared_dropout = nn.Dropout(dropout)
        self.personal_heads = nn.ModuleDict()
        
    def add_client(self, client_id):
        self.personal_heads[client_id] = nn.Linear(64, 1)
        
    def forward(self, x, client_id):
        lstm_out, _ = self.shared_lstm(x)
        features = self.shared_dropout(lstm_out[:, -1, :])
        return self.personal_heads[client_id](features)


# ============================================================
# 评估（用原始能耗计算 sMAPE）
# ============================================================

def compute_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def evaluate_raw(model, stations, client_id, device):
    """用原始能耗计算 sMAPE"""
    model.eval()
    all_preds_raw = []
    all_targets_raw = []
    
    for station in stations:
        X_norm = station['features_norm']
        y_norm = station['target_norm']
        y_raw = station['target_raw']
        scaler_y = station['scaler_y']
        
        # 创建序列
        X_seq, _ = create_sequences(X_norm, y_norm, seq_len=24, pred_len=1)
        if len(X_seq) == 0:
            continue
        
        # 预测
        X_tensor = torch.FloatTensor(X_seq).to(device)
        with torch.no_grad():
            pred_norm = model(X_tensor, client_id).cpu().numpy()
        
        # 反归一化
        pred_raw = scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        
        # 取对应的真实值（测试集）
        # 简化：取后 20% 作为测试
        n = len(y_raw)
        test_start = int(n * 0.8)
        test_targets = y_raw[test_start + 24:test_start + 24 + len(pred_raw)]
        
        if len(test_targets) != len(pred_raw):
            min_len = min(len(test_targets), len(pred_raw))
            test_targets = test_targets[:min_len]
            pred_raw = pred_raw[:min_len]
        
        all_preds_raw.extend(pred_raw)
        all_targets_raw.extend(test_targets)
    
    return compute_smape(np.array(all_targets_raw), np.array(all_preds_raw))


# ============================================================
# 训练函数
# ============================================================

def train_single(model, client_id, stations, epochs=30, lr=0.001):
    """单独训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for station in stations:
            X_norm = station['features_norm']
            y_norm = station['target_norm']
            
            X_seq, y_seq = create_sequences(X_norm, y_norm)
            if len(X_seq) == 0:
                continue
            
            dataset = StationDataset(X_norm, y_norm)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x, client_id)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        
        if (epoch+1) % 10 == 0:
            smape = evaluate_raw(model, stations, client_id, device)
            print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, sMAPE={smape:.2f}%")
    
    return evaluate_raw(model, stations, client_id, device)


def train_fedrep(model, stations_4g, stations_5g, rounds=15, local_epochs=5, lr=0.001):
    """FedRep 训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    
    for round_num in range(1, rounds+1):
        # 4G 本地训练（只更新共享层）
        optimizer_4g = torch.optim.Adam(model.shared_lstm.parameters(), lr=lr)
        for _ in range(local_epochs):
            for station in stations_4g:
                X_norm = station['features_norm']
                y_norm = station['target_norm']
                dataset = StationDataset(X_norm, y_norm)
                loader = DataLoader(dataset, batch_size=64, shuffle=True)
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer_4g.zero_grad()
                    output = model(x, '4g')
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer_4g.step()
        
        # 5G 本地训练（只更新共享层）
        optimizer_5g = torch.optim.Adam(model.shared_lstm.parameters(), lr=lr)
        for _ in range(local_epochs):
            for station in stations_5g:
                X_norm = station['features_norm']
                y_norm = station['target_norm']
                dataset = StationDataset(X_norm, y_norm)
                loader = DataLoader(dataset, batch_size=64, shuffle=True)
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer_5g.zero_grad()
                    output = model(x, '5g')
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer_5g.step()
        
        # 更新个性化头
        for _ in range(local_epochs):
            for station in stations_4g:
                X_norm = station['features_norm']
                y_norm = station['target_norm']
                dataset = StationDataset(X_norm, y_norm)
                loader = DataLoader(dataset, batch_size=64, shuffle=True)
                optimizer = torch.optim.Adam(model.personal_heads['4g'].parameters(), lr=lr)
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    output = model(x, '4g')
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
        
        for _ in range(local_epochs):
            for station in stations_5g:
                X_norm = station['features_norm']
                y_norm = station['target_norm']
                dataset = StationDataset(X_norm, y_norm)
                loader = DataLoader(dataset, batch_size=64, shuffle=True)
                optimizer = torch.optim.Adam(model.personal_heads['5g'].parameters(), lr=lr)
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    output = model(x, '5g')
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
        
        if round_num % 5 == 0:
            smape_4g = evaluate_raw(model, stations_4g, '4g', device)
            smape_5g = evaluate_raw(model, stations_5g, '5g', device)
            print(f"  Round {round_num}: 4G sMAPE={smape_4g:.2f}%, 5G sMAPE={smape_5g:.2f}%")
    
    return model


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*60)
    print("FedRep: 4G+5G 个性化联邦学习")
    print("="*60)
    
    # 加载数据
    print("\n1. 加载数据...")
    stations_4g = load_stations('4g', max_stations=MAX_STATIONS)
    stations_5g = load_stations('5g', max_stations=MAX_STATIONS)
    print(f"   4G: {len(stations_4g)} 个基站")
    print(f"   5G: {len(stations_5g)} 个基站")
    
    # 实验1: 4G单独训练
    print("\n" + "="*50)
    print("实验1: 4G单独训练")
    print("="*50)
    model_4g = FedRepLSTM(input_dim=5)
    model_4g.add_client('4g')
    smape_4g_alone = train_single(model_4g, '4g', stations_4g, epochs=20)
    print(f"\n✅ 4G单独训练 sMAPE: {smape_4g_alone:.2f}%")
    
    # 实验2: 5G单独训练
    print("\n" + "="*50)
    print("实验2: 5G单独训练")
    print("="*50)
    model_5g = FedRepLSTM(input_dim=5)
    model_5g.add_client('5g')
    smape_5g_alone = train_single(model_5g, '5g', stations_5g, epochs=20)
    print(f"\n✅ 5G单独训练 sMAPE: {smape_5g_alone:.2f}%")
    
    # 实验3: FedRep
    print("\n" + "="*50)
    print("实验3: FedRep (4G+5G协同)")
    print("="*50)
    model_fedrep = FedRepLSTM(input_dim=5)
    model_fedrep.add_client('4g')
    model_fedrep.add_client('5g')
    
    # 预训练
    print("\n预训练个性化头...")
    train_single(model_fedrep, '4g', stations_4g, epochs=10, lr=0.001)
    train_single(model_fedrep, '5g', stations_5g, epochs=10, lr=0.001)
    
    print("\nFedRep协同训练...")
    model_fedrep = train_fedrep(model_fedrep, stations_4g, stations_5g, rounds=15, local_epochs=3)
    
    # 评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fedrep.to(device)
    smape_4g_fedrep = evaluate_raw(model_fedrep, stations_4g, '4g', device)
    smape_5g_fedrep = evaluate_raw(model_fedrep, stations_5g, '5g', device)
    
    # 结果
    print("\n" + "="*50)
    print("结果汇总")
    print("="*50)
    print(f"\n4G单独训练: {smape_4g_alone:.2f}%")
    print(f"4G FedRep:   {smape_4g_fedrep:.2f}%")
    print(f"\n5G单独训练: {smape_5g_alone:.2f}%")
    print(f"5G FedRep:   {smape_5g_fedrep:.2f}%")
    
    if smape_5g_fedrep < smape_5g_alone:
        print(f"\n✅ 5G精度提升: {smape_5g_alone - smape_5g_fedrep:.2f}%")
    else:
        print(f"\n⚠️ 需要调整参数")
    
    # 画图
    plt.figure(figsize=(8, 5))
    models = ['4G Alone', '4G FedRep', '5G Alone', '5G FedRep']
    values = [smape_4g_alone, smape_4g_fedrep, smape_5g_alone, smape_5g_fedrep]
    colors = ['#2E8B57', '#F4A261', '#2E86AB', '#E76F51']
    bars = plt.bar(models, values, color=colors)
    plt.ylabel('sMAPE (%)')
    plt.title('4G+5G 预测精度对比 (FedRep)')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.1f}%', ha='center', fontweight='bold')
    plt.ylim(0, max(values) + 10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/fedrep_final.png', dpi=150)
    print(f"\n✅ 图片保存: results/fedrep_final.png")


if __name__ == "__main__":
    main()
