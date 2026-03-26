"""
单节点LSTM训练脚本
功能：
1. 在单个节点上训练LSTM模型
2. 验证损失计算正确（用归一化y）
3. 反归一化评估预测结果
4. 统一美化可视化输出（修复中文乱码）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import sys
import pickle
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.barcelona_dataset import get_node_data_loader


# ============================================================
# 统一图片风格配置（专业版 + 中文支持）
# ============================================================

# 设置中文字体（解决乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置seaborn风格
sns.set_theme(style="darkgrid", context="notebook")

# 配色方案（符合FedGreen-C绿色节能主题）
COLORS = {
    'primary': '#2E8B57',      # 主绿 - SeaGreen
    'secondary': '#2E86AB',    # 辅蓝 - Ocean Blue
    'accent': '#F18F01',       # 强调橙 - Warm Orange
    'warning': '#E76F51',      # 警示红 - Coral
    'success': '#6AAB9E',      # 成功绿
    'background': '#F5F5F5',   # 背景浅灰
    'text': '#2C3E50',         # 文字深灰
    'grid': '#D3D3D3'          # 网格浅灰
}

# 设置调色板
sns.set_palette([COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning']])

# matplotlib详细配置
plt.rcParams.update({
    # 图片尺寸和分辨率
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'white',
    
    # 字体设置
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    
    # 线条
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    
    # 网格
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.color': COLORS['grid'],
    
    # 保存格式
    'savefig.format': 'png'
})


# ============================================================
# 模型定义
# ============================================================
class LSTMPredictor(nn.Module):
    """
    LSTM能耗预测模型
    输入: [batch, window_size, features]
    输出: [batch, predict_size]
    """
    
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, (hidden, cell) = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # [batch, hidden_dim]
        output = self.fc(last_out)     # [batch, output_dim]
        return output


# ============================================================
# 训练函数
# ============================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)
    
    return total_loss / len(dataloader.dataset)


def inverse_normalize(predictions, scaler):
    """
    反归一化预测值
    predictions: numpy数组，形状 [n_samples, predict_size]
    scaler: MinMaxScaler对象
    """
    orig_shape = predictions.shape
    pred_flat = predictions.reshape(-1, 1)
    pred_orig = scaler.inverse_transform(pred_flat)
    return pred_orig.reshape(orig_shape)


def evaluate_original_scale(model, dataloader, scaler, device, min_energy_threshold=100):
    """
    在原始尺度上评估模型
    修复：过滤小值避免MAPE爆炸
    
    Args:
        min_energy_threshold: 过滤低于此值的样本（kWh）
    
    Returns: rmse, mae, mape, preds, targets
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 反归一化
    all_preds_orig = inverse_normalize(all_preds, scaler)
    all_targets_orig = inverse_normalize(all_targets, scaler)
    
    # 计算RMSE和MAE（使用所有样本）
    rmse = np.sqrt(np.mean((all_preds_orig - all_targets_orig) ** 2))
    mae = np.mean(np.abs(all_preds_orig - all_targets_orig))
    
    # MAPE: 过滤小值（避免除零或极小值导致爆炸）
    mask = all_targets_orig > min_energy_threshold
    if mask.sum() > 0:
        mape = np.mean(np.abs((all_preds_orig[mask] - all_targets_orig[mask]) / all_targets_orig[mask])) * 100
    else:
        mape = float('inf')
    
    return rmse, mae, mape, all_preds_orig, all_targets_orig


def plot_predictions(all_preds, all_targets, node_id, save_path=None):
    """
    绘制预测对比图（统一美化版 + 中文支持）
    2×2网格布局，4个子图对应4个预测步长
    """
    n_samples = min(200, len(all_preds))  # 显示前200个样本
    n_steps = 4
    
    # 定义子图标题（中文）
    step_names = [
        '步长1: 未来 0-6 小时',
        '步长2: 未来 6-12 小时',
        '步长3: 未来 12-18 小时',
        '步长4: 未来 18-24 小时'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    for i, step in enumerate(range(n_steps)):
        ax = axes[i // 2, i % 2]
        step_preds = all_preds[:n_samples, step]
        step_targets = all_targets[:n_samples, step]
        
        # 绘制曲线
        ax.plot(step_targets, label='真实值', color=COLORS['primary'], linewidth=1.8)
        ax.plot(step_preds, label='预测值', color=COLORS['secondary'], linewidth=1.5, linestyle='--', alpha=0.9)
        
        # 设置标题和标签（中文）
        ax.set_title(step_names[i], fontsize=12, fontweight='bold', color=COLORS['text'])
        ax.set_xlabel('样本序号', fontsize=10)
        ax.set_ylabel('能耗 (kWh)', fontsize=10)
        
        # 设置Y轴范围（根据数据自动调整）
        y_max = max(step_targets.max(), step_preds.max()) * 1.05
        ax.set_ylim(0, y_max)
        
        # 图例（中文）
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    # 添加总标题（中文）
    fig.suptitle(f'节点 {node_id} - LSTM能耗预测结果对比', 
                 fontsize=16, fontweight='bold', color=COLORS['primary'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 预测图已保存: {save_path}")
    
    plt.show()


def plot_loss_curve(train_losses, val_losses, node_id, save_path=None):
    """
    绘制损失曲线（统一美化版 + 中文支持）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='训练损失', color=COLORS['primary'], linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_losses, label='验证损失', color=COLORS['secondary'], linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax.set_ylabel('损失值 (MSE)', fontsize=12, fontweight='bold')
    ax.set_title(f'节点 {node_id} - 训练曲线', fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 损失曲线已保存: {save_path}")
    
    plt.show()


# ============================================================
# 主训练函数
# ============================================================
def train_single_node(node_id=8001, epochs=20, batch_size=64, 
                      window_size=28, predict_size=4,
                      hidden_dim=64, num_layers=2, lr=0.001,
                      min_energy_threshold=100):
    """
    训练单个节点
    
    Args:
        node_id: 节点编号
        epochs: 训练轮数
        batch_size: 批次大小
        window_size: 输入窗口
        predict_size: 输出窗口
        hidden_dim: LSTM隐藏层维度
        num_layers: LSTM层数
        lr: 学习率
        min_energy_threshold: MAPE计算时过滤低于此值的样本（kWh）
    """
    print("=" * 60)
    print(f"单节点训练: 节点 {node_id}")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建保存目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(base_dir, "results", "beautified")
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, scaler_path, train_dataset = get_node_data_loader(
        node_id=node_id,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        window_size=window_size,
        predict_size=predict_size,
        sector_feature=False
    )
    
    val_loader, _, _ = get_node_data_loader(
        node_id=node_id,
        split='val',
        batch_size=batch_size,
        shuffle=False,
        window_size=window_size,
        predict_size=predict_size,
        sector_feature=False
    )
    
    test_loader, _, _ = get_node_data_loader(
        node_id=node_id,
        split='test',
        batch_size=batch_size,
        shuffle=False,
        window_size=window_size,
        predict_size=predict_size,
        sector_feature=False
    )
    
    # 加载scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"训练批次: {len(train_loader)}")
    print(f"验证样本: {len(val_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")
    
    # 创建模型
    model = LSTMPredictor(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=predict_size,
        dropout=0.2
    ).to(device)
    
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    print("\n开始训练...")
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # 最终评估
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    # 归一化尺度评估
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\n归一化尺度:")
    print(f"  测试损失 (MSE): {test_loss:.6f}")
    
    # 原始尺度评估（修复MAPE）
    rmse, mae, mape, preds, targets = evaluate_original_scale(
        model, test_loader, scaler, device, min_energy_threshold
    )
    
    print(f"\n原始尺度 (kWh):")
    print(f"  RMSE: {rmse:.2f} kWh")
    print(f"  MAE:  {mae:.2f} kWh")
    print(f"  MAPE: {mape:.2f}% (过滤 < {min_energy_threshold} kWh 的样本)")
    
    # 判断是否成功
    if mape < 30:
        print(f"\n✅ 训练成功！相对误差 {mape:.2f}% < 30%")
    else:
        print(f"\n⚠️ 相对误差 {mape:.2f}% > 30%，需要调参")
    
    # 绘制预测对比图
    pred_plot_path = os.path.join(save_dir, f"node_{node_id}_predictions_{timestamp}.png")
    plot_predictions(preds, targets, node_id, pred_plot_path)
    
    # 绘制损失曲线
    loss_plot_path = os.path.join(save_dir, f"node_{node_id}_loss_curve_{timestamp}.png")
    plot_loss_curve(train_losses, val_losses, node_id, loss_plot_path)
    
    # 保存结果摘要
    results = {
        'node_id': node_id,
        'timestamp': timestamp,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'min_energy_threshold': min_energy_threshold
    }
    
    results_path = os.path.join(save_dir, f"node_{node_id}_results_{timestamp}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ 结果已保存: {results_path}")
    
    return model, results


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FedGreen-C 单节点LSTM训练")
    print("=" * 60)
    
    # 训练节点8001
    model, results = train_single_node(
        node_id=8001,
        epochs=20,
        batch_size=64,
        window_size=28,
        predict_size=4,
        hidden_dim=64,
        num_layers=2,
        lr=0.001,
        min_energy_threshold=100  # 过滤低于100 kWh的样本
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最终MAPE: {results['mape']:.2f}%")
    print("=" * 60)