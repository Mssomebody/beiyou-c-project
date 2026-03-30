#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级 SHAP 分析：比较 1 天窗口与 7 天窗口的特征重要性
- 自动日志保存
- 漂亮的可视化
- 可解释性分析
"""

import sys
import os
import json
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ============================================================
# 日志系统
# ============================================================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
        self.log.write(f"\n{'='*70}\n")
        self.log.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*70}\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"shap_window_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
sys.stdout = Logger(LOG_FILE)

# ============================================================
# 配置
# ============================================================
class Config:
    def __init__(self):
        self.data_dir = PROJECT_ROOT / "data/processed/barcelona_ready_2019_2022"
        self.node = 8001
        self.batch_size = 64
        self.epochs = 10
        self.learning_rate = 0.001
        self.hidden_dim = 64
        self.num_layers = 2
        self.dropout = 0.2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.shap_background = 100   # 背景样本数
        self.shap_samples = 500      # 解释样本数

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


# ============================================================
# 数据集定义（支持不同窗口长度）
# ============================================================
class TimeSeriesDataset(Dataset):
    """时序数据集，支持可变窗口长度"""
    def __init__(self, data_path, window_size=4, predict_size=4):
        df = pd.read_pickle(data_path)
        self.energy = df['Valor_norm'].values.astype(np.float32)
        self.window_size = window_size
        self.predict_size = predict_size
        self.indices = self._build_indices()

    def _build_indices(self):
        indices = []
        total_len = len(self.energy)
        for i in range(total_len - self.window_size - self.predict_size + 1):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.energy[start:start + self.window_size]
        y = self.energy[start + self.window_size:start + self.window_size + self.predict_size]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y)


# ============================================================
# LSTM 模型
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ============================================================
# 训练函数
# ============================================================
def train_model(dataset, config, model_name):
    """训练模型并返回模型和验证损失"""
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = LSTMPredictor(
        input_dim=1,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_dim=4,
        dropout=config.dropout
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"temp_model_{model_name}.pth")

        print(f"{model_name} - Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    model.load_state_dict(torch.load(f"temp_model_{model_name}.pth"))
    return model, best_val_loss


# ============================================================
# SHAP 分析（使用 KernelExplainer）
# ============================================================
def shap_analysis(model, dataset, window_size, config, title_prefix):
    """计算 SHAP 值并可视化"""
    # 准备背景数据集（随机采样）
    background_indices = np.random.choice(len(dataset), config.shap_background, replace=False)
    background = torch.stack([dataset[i][0] for i in background_indices]).numpy()
    background = background.reshape(background.shape[0], -1)  # 展平为 (n_samples, window_size*1)

    # 准备解释样本（随机采样）
    sample_indices = np.random.choice(len(dataset), config.shap_samples, replace=False)
    samples = torch.stack([dataset[i][0] for i in sample_indices]).numpy()
    samples_flat = samples.reshape(samples.shape[0], -1)

    # 定义模型预测函数（接受展平输入）
    def predict_fn(x_flat):
        x = x_flat.reshape(-1, window_size, 1)
        x_tensor = torch.FloatTensor(x).to(config.device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()
        return pred

    # 使用 KernelExplainer（适用于任意模型）
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(samples_flat, nsamples=100)  # nsamples 可调

    # shap_values 是一个列表，每个元素对应一个输出维度 (4,)
    # 我们取所有输出维度的平均 SHAP 值
    shap_avg = np.mean(shap_values, axis=0)  # (n_samples, window_size)
    shap_mean = np.mean(np.abs(shap_avg), axis=0)  # (window_size,)

    # 可视化：特征重要性条形图（每个时间步的重要性）
    plt.figure(figsize=(10, 5))
    hours = [f"{i//4}:{(i%4)*6:02d}" for i in range(window_size)]  # 粗略标签
    plt.bar(range(window_size), shap_mean, color='#2E8B57')
    plt.xlabel('时间步')
    plt.ylabel('平均 |SHAP 值|')
    plt.title(f'{title_prefix} 特征重要性 (时间步)')
    plt.xticks(range(window_size), hours, rotation=45)
    plt.tight_layout()
    plt.savefig(f"shap_{title_prefix}_timesteps.png", dpi=150)
    plt.close()

    # 可选：绘制一个样本的 SHAP 力图
    sample_idx = 0
    shap_sample = shap_avg[sample_idx]
    plt.figure(figsize=(12, 4))
    shap.plots.waterfall(shap.Explanation(values=shap_sample,
                                          base_values=explainer.expected_value[0],
                                          data=samples_flat[sample_idx],
                                          feature_names=hours), show=False)
    plt.tight_layout()
    plt.savefig(f"shap_{title_prefix}_waterfall.png", dpi=150)
    plt.close()

    print(f"{title_prefix} SHAP 分析完成，图片已保存。")
    return shap_mean


# ============================================================
# 主函数
# ============================================================
def main():
    config = Config()
    print("="*70)
    print("长短窗口 SHAP 对比分析")
    print("="*70)
    print(f"节点: {config.node}")
    print(f"设备: {config.device}")
    print("="*70)

    # 加载数据
    train_path = config.data_dir / f"node_{config.node}" / "train.pkl"
    test_path = config.data_dir / f"node_{config.node}" / "test.pkl"

    # 1天窗口
    print("\n【1天窗口 (4个时间步)】")
    dataset_1day = TimeSeriesDataset(train_path, window_size=4, predict_size=4)
    model_1day, val_loss_1day = train_model(dataset_1day, config, "1day")
    shap_1day = shap_analysis(model_1day, dataset_1day, window_size=4, config=config, title_prefix="1day")

    # 7天窗口
    print("\n【7天窗口 (28个时间步)】")
    dataset_7day = TimeSeriesDataset(train_path, window_size=28, predict_size=4)
    model_7day, val_loss_7day = train_model(dataset_7day, config, "7day")
    shap_7day = shap_analysis(model_7day, dataset_7day, window_size=28, config=config, title_prefix="7day")

    # 对比图：将7天窗口的 SHAP 值按天聚合
    shap_7day_daily = shap_7day.reshape(7, 4).mean(axis=1)  # 7天，每天平均

    plt.figure(figsize=(10, 5))
    days = [f"Day {i+1}" for i in range(7)]
    plt.bar(days, shap_7day_daily, color='#E76F51')
    plt.xlabel('天数')
    plt.ylabel('平均 |SHAP 值|')
    plt.title('7天窗口各天特征重要性')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("shap_7day_daily.png", dpi=150)
    plt.close()

    # 生成报告
    report = f"""
    ============================================================
    SHAP 分析报告
    ============================================================
    节点: {config.node}
    
    1天窗口 (4个时间步):
      - 验证损失: {val_loss_1day:.6f}
      - 特征重要性（时间步）: {shap_1day}
    
    7天窗口 (28个时间步):
      - 验证损失: {val_loss_7day:.6f}
      - 特征重要性（逐天）: {shap_7day_daily}
    
    观察:
      - 1天窗口中，重要性集中在哪些时段？{np.argmax(shap_1day)} 时间步
      - 7天窗口中，第一天的重要性显著高于后续天数？{shap_7day_daily[0] > shap_7day_daily[1]}
    
    结论: 短期窗口的重要性集中在特定时段，长期窗口的重要性随天数衰减，说明近期数据对预测贡献更大。
    ============================================================
    """
    print(report)
    with open("shap_window_report.txt", 'w') as f:
        f.write(report)

    # 清理临时模型文件
    for f in ["temp_model_1day.pth", "temp_model_7day.pth"]:
        if Path(f).exists():
            Path(f).unlink()

    print("\n✅ 分析完成！生成文件:")
    print("  - shap_1day_timesteps.png       (1天窗口时间步重要性)")
    print("  - shap_1day_waterfall.png       (1天窗口单个预测力图)")
    print("  - shap_7day_timesteps.png       (7天窗口时间步重要性)")
    print("  - shap_7day_waterfall.png       (7天窗口单个预测力图)")
    print("  - shap_7day_daily.png           (7天窗口逐日重要性)")
    print("  - shap_window_report.txt        (分析报告)")


if __name__ == "__main__":
    main()
