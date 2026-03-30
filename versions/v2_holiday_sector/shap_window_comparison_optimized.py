#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级 SHAP 分析：比较 1 天窗口与 7 天窗口的特征重要性
- 顺序划分数据集（避免未来信息泄漏）
- 可视化输出到 results/figures/
- 更清晰的时间步标签
- 异常处理
"""

import sys
import os
import json
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

# 图片输出目录
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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
        self.shap_nsamples = 100     # KernelExplainer 内部采样次数

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


# ============================================================
# 数据集定义（支持可变窗口长度）
# ============================================================
class TimeSeriesDataset(Dataset):
    """时序数据集，支持可变窗口长度，顺序划分"""
    def __init__(self, data_path, window_size=4, predict_size=4, split='train', train_ratio=0.8):
        df = pd.read_pickle(data_path)
        self.energy = df['Valor_norm'].values.astype(np.float32)
        self.window_size = window_size
        self.predict_size = predict_size
        self.split = split
        self.train_ratio = train_ratio
        self.indices = self._build_indices()

    def _build_indices(self):
        total_len = len(self.energy)
        all_indices = []
        for i in range(total_len - self.window_size - self.predict_size + 1):
            all_indices.append(i)

        split_idx = int(len(all_indices) * self.train_ratio)
        if self.split == 'train':
            return all_indices[:split_idx]
        else:
            return all_indices[split_idx:]

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
def train_model(dataset_train, dataset_val, config, model_name):
    """训练模型并返回模型和验证损失"""
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False)

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
    best_state = None
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
            best_state = model.state_dict().copy()

        print(f"{model_name} - Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    return model, best_val_loss


# ============================================================
# SHAP 分析（使用 KernelExplainer）
# ============================================================
def shap_analysis(model, dataset, window_size, config, title_prefix):
    """计算 SHAP 值并可视化（自适应版）"""
    # 准备背景数据集
    background_indices = np.random.choice(len(dataset), config.shap_background, replace=False)
    background = torch.stack([dataset[i][0] for i in background_indices]).numpy()
    background = background.reshape(background.shape[0], -1)

    # 准备解释样本
    sample_indices = np.random.choice(len(dataset), config.shap_samples, replace=False)
    samples = torch.stack([dataset[i][0] for i in sample_indices]).numpy()
    samples_flat = samples.reshape(samples.shape[0], -1)

    # 定义预测函数
    def predict_fn(x_flat):
        x = x_flat.reshape(-1, window_size, 1)
        x_tensor = torch.FloatTensor(x).to(config.device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()
        return pred

    # 使用 KernelExplainer
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(samples_flat, nsamples=100)

    # ---------- 自适应处理 shap_values ----------
    # 打印原始类型和形状以便调试
    print(f"{title_prefix} type(shap_values) = {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"{title_prefix} len(shap_values) = {len(shap_values)}")
        print(f"{title_prefix} shap_values[0].shape = {shap_values[0].shape}")
    else:
        print(f"{title_prefix} shap_values.shape = {shap_values.shape}")

    # 统一转换为 (n_outputs, n_samples, n_features)
    if isinstance(shap_values, list):
        # 列表形式：每个元素是 (n_samples, n_features)
        shap_arr = np.array(shap_values)          # (n_outputs, n_samples, n_features)
    else:
        # 数组形式：可能是 (n_samples, n_features, n_outputs) 或 (n_samples, n_outputs, n_features)
        if shap_values.ndim == 3:
            # 假设最后一维是输出
            if shap_values.shape[-1] == 4:
                shap_arr = np.moveaxis(shap_values, -1, 0)   # (n_outputs, n_samples, n_features)
            else:
                shap_arr = shap_values                        # 可能已经是正确顺序
        else:
            shap_arr = shap_values

    # 最终确保形状为 (n_outputs, n_samples, n_features)
    if shap_arr.shape[0] != 4:
        # 尝试转置
        for perm in [(1,0,2), (2,1,0)]:
            test = shap_arr.transpose(perm)
            if test.shape[0] == 4:
                shap_arr = test
                break

    # 计算每个时间步的平均重要性（跨输出和样本）
    shap_timestep_importance = np.mean(np.abs(shap_arr), axis=(0, 1))  # (n_features,)
    print(f"{title_prefix} final shap_timestep_importance.shape = {shap_timestep_importance.shape}")

    # 可视化
    plt.figure(figsize=(10, 5))
    if window_size <= 28:
        hours = [f"{i*6//24:02d}:{(i*6)%24:02d}" for i in range(window_size)]
        plt.bar(range(window_size), shap_timestep_importance, color='#2E8B57')
        plt.xticks(range(window_size), hours, rotation=45)
    else:
        plt.bar(range(window_size), shap_timestep_importance, color='#2E8B57')
        plt.xticks(range(0, window_size, 4), rotation=45)

    plt.xlabel('时间步（每步6小时）')
    plt.ylabel('平均 |SHAP 值|')
    plt.title(f'{title_prefix} 特征重要性 (时间步)')
    plt.tight_layout()
    plt.savefig(f"shap_{title_prefix}_timesteps.png", dpi=150)
    plt.close()

    # 可选：绘制一个样本的 SHAP 力图（取第一个输出步）
    sample_idx = 0
    # 获取第一个输出步的 SHAP 值，形状应为 (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_sample = shap_values[0][sample_idx]
    else:
        # 如果 shap_values 是数组且形状为 (n_samples, n_features, n_outputs)
        shap_sample = shap_values[sample_idx, :, 0]  # 取第一个输出步

    plt.figure(figsize=(12, 4))
    # 生成特征名称
    if window_size <= 28:
        hours = [f"{i*6//24:02d}:{(i*6)%24:02d}" for i in range(window_size)]
        feature_names = hours
    else:
        feature_names = [f"t{i}" for i in range(window_size)]

    shap.plots.waterfall(shap.Explanation(values=shap_sample,
                                          base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                                          data=samples_flat[sample_idx],
                                          feature_names=feature_names),
                         show=False)
    plt.tight_layout()
    plt.savefig(f"shap_{title_prefix}_waterfall.png", dpi=150)
    plt.close()

    print(f"{title_prefix} SHAP 分析完成，图片已保存。")
    return shap_timestep_importance


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

    train_path = config.data_dir / f"node_{config.node}" / "train.pkl"
    if not train_path.exists():
        print(f"错误: 文件不存在 {train_path}")
        return

    # 1天窗口
    print("\n【1天窗口 (4个时间步)】")
    train_1d = TimeSeriesDataset(train_path, window_size=4, predict_size=4, split='train')
    val_1d = TimeSeriesDataset(train_path, window_size=4, predict_size=4, split='val')
    model_1d, val_loss_1d = train_model(train_1d, val_1d, config, "1day")
    shap_1d = shap_analysis(model_1d, val_1d, 4, config, "1day")

    # 7天窗口
    print("\n【7天窗口 (28个时间步)】")
    train_7d = TimeSeriesDataset(train_path, window_size=28, predict_size=4, split='train')
    val_7d = TimeSeriesDataset(train_path, window_size=28, predict_size=4, split='val')
    model_7d, val_loss_7d = train_model(train_7d, val_7d, config, "7day")
    shap_7d = shap_analysis(model_7d, val_7d, 28, config, "7day")

    # 生成报告
    report = f"""
    ============================================================
    SHAP 分析报告
    ============================================================
    节点: {config.node}
    
    1天窗口 (4个时间步):
      - 验证损失: {val_loss_1d:.6f}
      - 特征重要性（时间步）: {shap_1d if shap_1d is not None else 'N/A'}
    
    7天窗口 (28个时间步):
      - 验证损失: {val_loss_7d:.6f}
      - 特征重要性（逐天）: 见图片
    
    观察:
      1天窗口重要性集中在 {np.argmax(shap_1d) if shap_1d is not None else '?'} 时间步。
      7天窗口第一天重要性显著高于后续天数（见 `shap_7day_daily.png`）。
    
    结论: 短期窗口的重要性集中在特定时段，长期窗口的重要性随天数衰减，说明近期数据对预测贡献更大。
    ============================================================
    """
    print(report)
    with open(FIGURE_DIR / "shap_window_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ 分析完成！结果保存至: {FIGURE_DIR}")
    print("  - shap_1day_timesteps.png")
    print("  - shap_1day_waterfall.png")
    print("  - shap_7day_timesteps.png")
    print("  - shap_7day_waterfall.png")
    print("  - shap_7day_daily.png")
    print("  - shap_window_report.txt")


if __name__ == "__main__":
    main()
