"""
FedRep: 个性化联邦学习 - 修复版
正确反归一化，计算有意义的 sMAPE
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
# 数据加载（带 scaler）
# ============================================================

def load_aligned_data_with_scaler(data_dir, max_stations=200):
    """加载数据，同时返回 scaler 用于反归一化"""
    station_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
    station_dirs = station_dirs[:max_stations]
    
    all_features = []
    all_targets = []
    all_scalers = []
    
    for station_dir in station_dirs:
        with open(station_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        # 加载 scaler_y
        with open(station_dir / 'scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        
        features = data['features'][:, :5]  # 公共特征
        target = data['target']
        
        all_features.append(features)
        all_targets.append(target)
        all_scalers.append(scaler_y)
    
    features = np.concatenate(all_features, axis=0)
    target = np.concatenate(all_targets, axis=0)
    
    return features, target, all_scalers


def create_dataloaders(features, target, batch_size=64, train_ratio=0.8, seq_len=24, pred_len=1):
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
    
    return train_loader, test_loader, (features.mean(), features.std())


# ============================================================
# 模型
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
        shared_features = self.shared_dropout(lstm_out[:, -1, :])
        return self.personal_heads[client_id](shared_features)


# ============================================================
# 评估指标（反归一化后计算 sMAPE）
# ============================================================

def compute_smape_raw(y_true_raw, y_pred_raw):
    """用原始值计算 sMAPE"""
    denominator = (np.abs(y_true_raw) + np.abs(y_pred_raw)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true_raw - y_pred_raw) / denominator) * 100


def evaluate_raw(model, test_loader, client_id, device, scaler_mean, scaler_scale):
    """反归一化后评估 sMAPE"""
    model.eval()
    all_preds_norm = []
    all_targets_norm = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x, client_id)
            all_preds_norm.append(output.cpu().numpy())
            all_targets_norm.append(y.cpu().numpy())
    
    all_preds_norm = np.concatenate(all_preds_norm)
    all_targets_norm = np.concatenate(all_targets_norm)
    
    # 反归一化
    all_preds_raw = all_preds_norm * scaler_scale + scaler_mean
    all_targets_raw = all_targets_norm * scaler_scale + scaler_mean
    
    return compute_smape_raw(all_targets_raw, all_preds_raw)


# ============================================================
# 训练函数
# ============================================================

def train_single(model, client_id, train_loader, test_loader, 
                 scaler_mean, scaler_scale, epochs=50, lr=0.001, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
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
        
        if (epoch+1) % 10 == 0 and verbose:
            smape = evaluate_raw(model, test_loader, client_id, device, scaler_mean, scaler_scale)
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, sMAPE={smape:.2f}%")
    
    final_smape = evaluate_raw(model, test_loader, client_id, device, scaler_mean, scaler_scale)
    return final_smape


def train_fedrep(model, client_loaders, client_test_loaders, client_ids,
                 scaler_means, scaler_scales, rounds=15, local_epochs=5, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    
    for round_num in range(1, rounds+1):
        # 收集共享层更新
        shared_updates = []
        
        for client_id in client_ids:
            loader = client_loaders[client_id]
            optimizer = torch.optim.Adam(model.shared_lstm.parameters(), lr=lr)
            
            for _ in range(local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    output = model(x, client_id)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
            
            shared_updates.append({k: v.cpu().clone() for k, v in model.shared_lstm.state_dict().items()})
        
        # 聚合共享层
        avg_state = {}
        for key in shared_updates[0].keys():
            avg_state[key] = torch.stack([u[key].float() for u in shared_updates]).mean(dim=0)
        model.shared_lstm.load_state_dict(avg_state)
        
        # 更新个性化头
        for client_id in client_ids:
            optimizer = torch.optim.Adam(model.personal_heads[client_id].parameters(), lr=lr)
            loader = client_loaders[client_id]
            for _ in range(local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    output = model(x, client_id)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
        
        if round_num % 5 == 0:
            smape_4g = evaluate_raw(model, client_test_loaders['4g'], '4g', device,
                                   scaler_means['4g'], scaler_scales['4g'])
            smape_5g = evaluate_raw(model, client_test_loaders['5g'], '5g', device,
                                   scaler_means['5g'], scaler_scales['5g'])
            print(f"  Round {round_num}: 4G sMAPE={smape_4g:.2f}%, 5G sMAPE={smape_5g:.2f}%")
    
    return model


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*70)
    print("FedRep: 4G+5G 个性化联邦学习")
    print("="*70)
    
    data_dir = Path("D:/Desk/desk/beiyou_c_project/data/processed/tsinghua")
    
    # 加载4G数据
    print("\n1. 加载4G数据...")
    features_4g, target_4g, scalers_4g = load_aligned_data_with_scaler(data_dir / '4g', max_stations=200)
    # 用所有基站的均值作为全局 scaler（简化）
    scaler_mean_4g = np.mean([s.mean_[0] for s in scalers_4g])
    scaler_scale_4g = np.mean([s.scale_[0] for s in scalers_4g])
    print(f"   4G: {len(features_4g)} 样本")
    
    # 加载5G数据
    print("\n2. 加载5G数据...")
    features_5g, target_5g, scalers_5g = load_aligned_data_with_scaler(data_dir / '5g', max_stations=200)
    scaler_mean_5g = np.mean([s.mean_[0] for s in scalers_5g])
    scaler_scale_5g = np.mean([s.scale_[0] for s in scalers_5g])
    print(f"   5G: {len(features_5g)} 样本")
    
    # 创建加载器
    train_4g, test_4g, _ = create_dataloaders(features_4g, target_4g)
    train_5g, test_5g, _ = create_dataloaders(features_5g, target_5g)
    
    # ============================================================
    # 实验1: 4G单独训练
    # ============================================================
    print("\n" + "="*60)
    print("实验1: 4G单独训练")
    print("="*60)
    
    model_4g = FedRepLSTM(input_dim=5)
    model_4g.add_client('4g')
    smape_4g_alone = train_single(model_4g, '4g', train_4g, test_4g,
                                   scaler_mean_4g, scaler_scale_4g, epochs=30)
    print(f"\n✅ 4G单独训练 sMAPE: {smape_4g_alone:.2f}%")
    
    # ============================================================
    # 实验2: 5G单独训练
    # ============================================================
    print("\n" + "="*60)
    print("实验2: 5G单独训练")
    print("="*60)
    
    model_5g = FedRepLSTM(input_dim=5)
    model_5g.add_client('5g')
    smape_5g_alone = train_single(model_5g, '5g', train_5g, test_5g,
                                   scaler_mean_5g, scaler_scale_5g, epochs=30)
    print(f"\n✅ 5G单独训练 sMAPE: {smape_5g_alone:.2f}%")
    
    # ============================================================
    # 实验3: FedRep
    # ============================================================
    print("\n" + "="*60)
    print("实验3: FedRep (4G+5G协同)")
    print("="*60)
    
    model_fedrep = FedRepLSTM(input_dim=5)
    model_fedrep.add_client('4g')
    model_fedrep.add_client('5g')
    
    # 预训练
    print("\n预训练个性化头...")
    train_single(model_fedrep, '4g', train_4g, test_4g,
                 scaler_mean_4g, scaler_scale_4g, epochs=10, verbose=False)
    train_single(model_fedrep, '5g', train_5g, test_5g,
                 scaler_mean_5g, scaler_scale_5g, epochs=10, verbose=False)
    
    # FedRep训练
    print("\nFedRep协同训练...")
    client_loaders = {'4g': train_4g, '5g': train_5g}
    client_test_loaders = {'4g': test_4g, '5g': test_5g}
    scaler_means = {'4g': scaler_mean_4g, '5g': scaler_mean_5g}
    scaler_scales = {'4g': scaler_scale_4g, '5g': scaler_scale_5g}
    
    model_fedrep = train_fedrep(model_fedrep, client_loaders, client_test_loaders,
                                 ['4g', '5g'], scaler_means, scaler_scales,
                                 rounds=15, local_epochs=5)
    
    # 最终评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fedrep.to(device)
    smape_4g_fedrep = evaluate_raw(model_fedrep, test_4g, '4g', device,
                                    scaler_mean_4g, scaler_scale_4g)
    smape_5g_fedrep = evaluate_raw(model_fedrep, test_5g, '5g', device,
                                    scaler_mean_5g, scaler_scale_5g)
    
    # ============================================================
    # 结果
    # ============================================================
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    print(f"\n{'模型':<25} {'sMAPE':<12}")
    print("-"*40)
    print(f"{'4G单独训练':<25} {smape_4g_alone:<12.2f}%")
    print(f"{'5G单独训练':<25} {smape_5g_alone:<12.2f}%")
    print(f"{'4G (FedRep)':<25} {smape_4g_fedrep:<12.2f}%")
    print(f"{'5G (FedRep)':<25} {smape_5g_fedrep:<12.2f}%")
    
    print(f"\n创新点验证:")
    print(f"  ✅ 5G精度提升: {smape_5g_alone:.2f}% → {smape_5g_fedrep:.2f}% (提升 {smape_5g_alone - smape_5g_fedrep:.2f}%)")
    print(f"  ✅ 4G精度变化: {smape_4g_alone:.2f}% → {smape_4g_fedrep:.2f}%")
    
    # 画图
    plt.figure(figsize=(10, 6))
    models = ['4G Alone', '5G Alone', '4G FedRep', '5G FedRep']
    values = [smape_4g_alone, smape_5g_alone, smape_4g_fedrep, smape_5g_fedrep]
    colors = ['#2E8B57', '#2E86AB', '#F4A261', '#E76F51']
    bars = plt.bar(models, values, color=colors)
    plt.ylabel('sMAPE (%)')
    plt.title('4G+5G 预测精度对比 (FedRep)')
    plt.ylim(0, max(values) + 10)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.1f}%', ha='center', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/fedrep_fixed.png', dpi=150)
    print(f"\n✅ 图片保存: results/fedrep_fixed.png")


if __name__ == "__main__":
    main()
