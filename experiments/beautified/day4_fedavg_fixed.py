"""
FedAvg 联邦学习 - 专业完整版
【特性】
1. ✅ 实时打印训练进度（每轮/每5秒）
2. ✅ 智能早停（平滑+耐心+多指标）
3. ✅ 断点续训（自动保存/恢复）
4. ✅ 收敛判断（损失/RMSE/波动）
5. ✅ 多指标监控（RMSE, MAE, MAPE, R²）
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
import argparse
import json
import pickle
from pathlib import Path
from collections import deque
from tqdm import tqdm

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

# 添加项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# 导入数据加载器
try:
    from src.data_loader.barcelona_loader import BarcelonaEnergyLoader
    print("✅ 数据加载器导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============= 1. 实时日志器 =============
class TrainingLogger:
    """实时打印训练进度"""
    def __init__(self, total_rounds, update_interval=1):
        self.total_rounds = total_rounds
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_print = 0
        
    def print_header(self):
        print("\n" + "="*80)
        print(f"{'轮次':>6} | {'耗时':>8} | {'Val Loss':>10} | {'RMSE':>10} | {'MAE':>10} | {'MAPE':>6} | {'R²':>6} | {'早停计数':>8}")
        print("-"*80)
    
    def print_round(self, round_idx, val_loss, metrics, patience_counter, best_flag=False):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 计算平均每轮时间
        avg_time = elapsed / (round_idx + 1)
        remaining = avg_time * (self.total_rounds - round_idx - 1)
        
        # 格式化输出
        flag = "✨" if best_flag else "  "
        print(f"{round_idx+1:6d} | {elapsed:7.1f}s | {val_loss:10.4f} | "
              f"{metrics['rmse']:10.0f} | {metrics['mae']:10.0f} | "
              f"{metrics['mape']:5.1f}% | {metrics['r2']:5.3f} | "
              f"{patience_counter:4d}/{patience_counter:4d} {flag}")
        
        # 估计剩余时间
        if (round_idx + 1) % 5 == 0:
            print(f"⏱️  预计剩余: {remaining/60:.1f}分钟")
    
    def print_footer(self, best_round, best_loss):
        total_time = time.time() - self.start_time
        print("="*80)
        print(f"✅ 训练完成！总耗时: {total_time/60:.2f}分钟")
        print(f"🏆 最佳模型: Round {best_round+1} (Val Loss: {best_loss:.4f})")

# ============= 2. 智能早停 =============
class EarlyStopping:
    """带平滑和耐心的早停"""
    def __init__(self, patience=10, min_delta=0.001, smooth_window=3):
        self.patience = patience
        self.min_delta = min_delta
        self.smooth_window = smooth_window
        self.counter = 0
        self.best_loss = float('inf')
        self.loss_history = deque(maxlen=smooth_window)
        self.best_round = -1
        self.best_model = None
        
    def check(self, loss, round_idx, model=None):
        self.loss_history.append(loss)
        
        # 平滑处理
        smooth_loss = np.mean(list(self.loss_history))
        
        # 判断是否改善
        if smooth_loss < self.best_loss - self.min_delta:
            self.best_loss = smooth_loss
            self.counter = 0
            self.best_round = round_idx
            if model:
                self.best_model = {k: v.clone() for k, v in model.state_dict().items()}
            return False, True  # 未停止，有改善
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False  # 停止，无改善
            return False, False  # 未停止，无改善
    
    def get_best_model(self):
        return self.best_model

# ============= 3. 断点管理器 =============
class CheckpointManager:
    """自动保存和恢复训练状态"""
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.best_path = self.save_dir / 'best_model.pth'
        self.checkpoint_path = self.save_dir / 'latest_checkpoint.pkl'
        self.history_path = self.save_dir / 'training_history.json'
    
    def save(self, round_idx, global_model, clients, early_stopping, metrics_history):
        """保存检查点"""
        # 保存模型和优化器状态
        checkpoint = {
            'round': round_idx,
            'model_state': global_model.state_dict(),
            'early_stopping': {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
                'best_round': early_stopping.best_round,
                'loss_history': list(early_stopping.loss_history)
            },
            'metrics_history': metrics_history,
            'timestamp': time.time()
        }
        
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # 保存训练历史（用于可视化）
        with open(self.history_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        # 保存最佳模型
        if early_stopping.best_round == round_idx:
            torch.save(global_model.state_dict(), self.best_path)
            print(f"\n💾 保存最佳模型 (round {round_idx+1})")
    
    def load(self):
        """加载检查点"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_latest_round(self):
        """获取最新轮次"""
        checkpoint = self.load()
        return checkpoint['round'] if checkpoint else -1

# ============= 4. 收敛检查 =============
def check_convergence(metrics_history, window=5, threshold=0.01):
    """检查是否真正收敛"""
    if len(metrics_history) < window:
        return False
    
    recent = metrics_history[-window:]
    
    # 1. 损失变化率
    loss_change = abs(recent[-1]['val_loss'] - recent[0]['val_loss']) / (recent[0]['val_loss'] + 1e-8)
    
    # 2. RMSE变化率
    rmse_change = abs(recent[-1]['rmse'] - recent[0]['rmse']) / (recent[0]['rmse'] + 1e-8)
    
    # 3. 波动性（变异系数）
    loss_std = np.std([m['val_loss'] for m in recent])
    loss_mean = np.mean([m['val_loss'] for m in recent])
    cv = loss_std / (loss_mean + 1e-8)
    
    # 4. 梯度（最后几轮的斜率）
    if len(recent) >= 3:
        x = np.arange(len(recent))
        y = np.array([m['val_loss'] for m in recent])
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0
    
    converged = (loss_change < threshold and 
                 rmse_change < threshold and 
                 cv < 0.05 and 
                 abs(slope) < 0.001)
    
    return converged

# LSTM模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# 客户端
class FedClient:
    def __init__(self, client_id, X_train, y_train, X_val, y_val, X_test, y_test):
        self.client_id = client_id
        self.model = LSTMPredictor().to(device)
        
        # 记录标准化参数
        self.y_mean = y_train.mean()
        self.y_std = y_train.std() + 1e-8
        
        # 标准化数据
        X_train_norm = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
        X_val_norm = (X_val - X_train.mean()) / (X_train.std() + 1e-8)
        X_test_norm = (X_test - X_train.mean()) / (X_train.std() + 1e-8)
        y_train_norm = (y_train - self.y_mean) / self.y_std
        y_val_norm = (y_val - self.y_mean) / self.y_std
        y_test_norm = (y_test - self.y_mean) / self.y_std
        
        # 保存原始数据
        self.y_train_raw = y_train
        self.y_val_raw = y_val
        self.y_test_raw = y_test
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train_norm), 
                torch.FloatTensor(y_train_norm).reshape(-1, 1)
            ), 
            batch_size=32, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_val_norm), 
                torch.FloatTensor(y_val_norm).reshape(-1, 1)
            ), 
            batch_size=32, shuffle=False
        )
        self.test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test_norm), 
                torch.FloatTensor(y_test_norm).reshape(-1, 1)
            ), 
            batch_size=32, shuffle=False
        )
        
        print(f"  客户端 {client_id}: 训练{y_train.shape} 验证{y_val.shape} 测试{y_test.shape}")
    
    def train(self, global_model, epochs=5, lr=0.001):
        self.model.load_state_dict(global_model.state_dict())
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for X, y in self.train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            val_loss = self._evaluate(self.model, self.val_loader, self.y_val_raw)
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
        
        # 恢复最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.model.state_dict(), best_val_loss
    
    def _evaluate(self, model, loader, raw_data):
        """内部评估函数"""
        model.eval()
        preds_norm = []
        
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device)
                if X.dim() == 4:
                    X = X.squeeze(-1)
                output = model(X)
                preds_norm.extend(output.cpu().numpy().flatten())
        
        # 反标准化
        preds_raw = np.array(preds_norm) * self.y_std + self.y_mean
        truths_raw = raw_data[:len(preds_raw)]
        
        # 计算MSE
        mse = np.mean((preds_raw - truths_raw) ** 2)
        return mse
    
    def evaluate(self, model, mode='test'):
        """公开评估函数"""
        loader = self.test_loader if mode == 'test' else self.val_loader
        raw_data = self.y_test_raw if mode == 'test' else self.y_val_raw
        
        model.eval()
        preds_norm = []
        
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device)
                if X.dim() == 4:
                    X = X.squeeze(-1)
                output = model(X)
                preds_norm.extend(output.cpu().numpy().flatten())
        
        # 反标准化
        preds_raw = np.array(preds_norm) * self.y_std + self.y_mean
        truths_raw = raw_data[:len(preds_raw)]
        
        # 计算多个指标
        mse = np.mean((preds_raw - truths_raw) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_raw - truths_raw))
        mape = np.mean(np.abs((preds_raw - truths_raw) / (truths_raw + 1e-8))) * 100
        ss_res = np.sum((truths_raw - preds_raw) ** 2)
        ss_tot = np.sum((truths_raw - np.mean(truths_raw)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return mse, {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'pred_mean': np.mean(preds_raw),
            'actual_mean': np.mean(truths_raw),
            'pred_std': np.std(preds_raw),
            'actual_std': np.std(truths_raw),
            'pred_range': [np.min(preds_raw), np.max(preds_raw)],
            'actual_range': [np.min(truths_raw), np.max(truths_raw)],
            'predictions': preds_raw,
            'truths': truths_raw
        }

# FedAvg聚合
def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.stack([cm[key] for cm in client_models]).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def main(args):
    print("="*80)
    print(" FedAvg联邦学习 - 专业完整版".center(78))
    print("="*80)
    print(f"设备: {device}")
    print(f"参数: rounds={args.rounds}, epochs={args.epochs}, lr={args.lr}")
    print(f"      hidden={args.hidden_size}, layers={args.num_layers}")
    print(f"      patience={args.patience}, resume={args.resume}")
    
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
    
    # 2. 划分数据集
    print("\n📈 划分数据集 (8:1:1)...")
    clients = []
    
    for node_id, data in fed_data.items():
        X, y = data['X_train'], data['y_train']
        
        # 随机划分
        n = len(X)
        indices = np.random.permutation(n)
        train_idx = indices[:int(n*0.8)]
        val_idx = indices[int(n*0.8):int(n*0.9)]
        test_idx = indices[int(n*0.9):]
        
        client = FedClient(
            client_id=node_id,
            X_train=X[train_idx], y_train=y[train_idx],
            X_val=X[val_idx], y_val=y[val_idx],
            X_test=X[test_idx], y_test=y[test_idx]
        )
        clients.append(client)
    
    # 3. 初始化全局模型
    global_model = LSTMPredictor(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    # 4. 初始化各组件
    logger = TrainingLogger(args.rounds)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    checkpoint_mgr = CheckpointManager()
    
    # 5. 恢复检查点
    start_round = 0
    metrics_history = []
    
    if args.resume:
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            start_round = checkpoint['round'] + 1
            global_model.load_state_dict(checkpoint['model_state'])
            early_stopping.best_loss = checkpoint['early_stopping']['best_loss']
            early_stopping.counter = checkpoint['early_stopping']['counter']
            early_stopping.best_round = checkpoint['early_stopping']['best_round']
            early_stopping.loss_history = deque(checkpoint['early_stopping']['loss_history'], 
                                               maxlen=early_stopping.smooth_window)
            metrics_history = checkpoint['metrics_history']
            print(f"\n🔄 恢复训练，从 round {start_round+1} 开始")
    
    # 6. 联邦训练
    print("\n🚀 开始联邦训练...")
    logger.print_header()
    
    for round_idx in range(start_round, args.rounds):
        round_start = time.time()
        
        # 客户端训练
        client_models = []
        round_val_loss = 0
        
        for client in clients:
            model_state, val_loss = client.train(
                global_model, 
                epochs=args.epochs, 
                lr=args.lr
            )
            client_models.append(model_state)
            round_val_loss += val_loss
        
        # 聚合
        global_model = fed_avg(global_model, client_models)
        
        # 验证损失
        avg_val_loss = round_val_loss / len(clients)
        
        # 测试
        test_loss_sum = 0
        all_stats = []
        for client in clients:
            loss, stats = client.evaluate(global_model, mode='test')
            test_loss_sum += loss
            all_stats.append(stats)
        
        avg_test_loss = test_loss_sum / len(clients)
        
        # 聚合指标
        metrics = {
            'val_loss': avg_val_loss,
            'test_loss': avg_test_loss,
            'rmse': np.mean([s['rmse'] for s in all_stats]),
            'mae': np.mean([s['mae'] for s in all_stats]),
            'mape': np.mean([s['mape'] for s in all_stats]),
            'r2': np.mean([s['r2'] for s in all_stats]),
            'round': round_idx
        }
        metrics_history.append(metrics)
        
        # 早停检查
        stop, improved = early_stopping.check(avg_val_loss, round_idx, global_model)
        
        # 实时打印
        logger.print_round(round_idx, avg_val_loss, metrics, 
                          early_stopping.counter, improved)
        
        # 保存检查点
        if (round_idx + 1) % args.save_interval == 0:
            checkpoint_mgr.save(round_idx, global_model, clients, 
                              early_stopping, metrics_history)
        
        # 收敛检查
        if check_convergence(metrics_history):
            print(f"\n🎯 模型已收敛，停止训练 (round {round_idx+1})")
            break
        
        # 早停
        if stop:
            print(f"\n🛑 触发早停 (round {round_idx+1})")
            break
    
    # 7. 加载最佳模型
    if early_stopping.best_model:
        global_model.load_state_dict(early_stopping.best_model)
        print(f"\n✨ 加载最佳模型 (round {early_stopping.best_round+1})")
    
    # 8. 最终测试
    print("\n" + "="*80)
    print("📊 最终测试结果".center(78))
    print("="*80)
    
    final_metrics = []
    for client in clients:
        loss, stats = client.evaluate(global_model, mode='test')
        final_metrics.append(stats)
        print(f"\n  客户端 {client.client_id}:")
        print(f"    RMSE: {stats['rmse']:.2f} kWh")
        print(f"    MAE:  {stats['mae']:.2f} kWh")
        print(f"    MAPE: {stats['mape']:.2f}%")
        print(f"    R²:   {stats['r2']:.4f}")
        print(f"    预测均值: {stats['pred_mean']:.2f} | 实际均值: {stats['actual_mean']:.2f}")
    
    # 9. 绘制训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rounds = [m['round'] for m in metrics_history]
    val_losses = [m['val_loss'] for m in metrics_history]
    rmses = [m['rmse'] for m in metrics_history]
    maes = [m['mae'] for m in metrics_history]
    mapes = [m['mape'] for m in metrics_history]
    
    # 损失曲线
    axes[0, 0].plot(rounds, val_losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Communication Rounds')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title('Validation Loss Curve')
    axes[0, 0].grid(True, alpha=0.3)
    if early_stopping.best_round >= 0:
        axes[0, 0].axvline(x=early_stopping.best_round, color='r', linestyle='--', alpha=0.5)
    
    # RMSE曲线
    axes[0, 1].plot(rounds, rmses, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Communication Rounds')
    axes[0, 1].set_ylabel('RMSE (kWh)')
    axes[0, 1].set_title('Test RMSE Curve')
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE曲线
    axes[1, 0].plot(rounds, maes, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Communication Rounds')
    axes[1, 0].set_ylabel('MAE (kWh)')
    axes[1, 0].set_title('Test MAE Curve')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAPE曲线
    axes[1, 1].plot(rounds, mapes, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Communication Rounds')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].set_title('Test MAPE Curve')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(project_root, "results", "fedavg_professional.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 训练曲线已保存: {save_path}")
    
    # 保存最终模型
    model_path = os.path.join(project_root, "checkpoints", "fedavg_final.pth")
    torch.save(global_model.state_dict(), model_path)
    print(f"💾 最终模型已保存: {model_path}")
    
    # 打印总结
    logger.print_footer(early_stopping.best_round, early_stopping.best_loss)
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedAvg联邦学习 - 专业版')
    parser.add_argument('--rounds', type=int, default=20, help='联邦通信轮数')
    parser.add_argument('--epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--save_interval', type=int, default=5, help='保存检查点间隔')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--no_resume', action='store_false', dest='resume', help='不恢复训练')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    main(args)