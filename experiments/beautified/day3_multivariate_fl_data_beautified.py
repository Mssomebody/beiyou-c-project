# day3_multivariate_fl_data_beautified.py
# 第3天：多变量时序数据 + 联邦学习Non-IID数据准备（美化版本）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os

# 设置随机种子
np.random.seed(42)

# ==================== 1. 生成多基站Non-IID数据 ====================

def generate_site_data(site_id, n_days=30):
    """
    生成单个基站的Non-IID数据
    site_id: 0-4，不同站点有不同分布特性
    """
    # 时间序列
    start_time = datetime(2024, 1, 1)
    periods = n_days * 24  # 每小时
    timestamps = pd.date_range(start=start_time, periods=periods, freq='h')
    
    # 站点特性配置（制造Non-IID差异）
    site_configs = {
        0: {'base_power': 3.0, 'temp_mean': 25, 'temp_std': 10, 'traffic_factor': 1.2},  # 大城市
        1: {'base_power': 2.5, 'temp_mean': 30, 'temp_std': 5, 'traffic_factor': 0.8},   # 热带
        2: {'base_power': 3.5, 'temp_mean': 10, 'temp_std': 15, 'traffic_factor': 0.6},  # 寒带
        3: {'base_power': 4.0, 'temp_mean': 22, 'temp_std': 8, 'traffic_factor': 1.5},   # 工业区
        4: {'base_power': 2.8, 'temp_mean': 20, 'temp_std': 6, 'traffic_factor': 0.9}    # 居民区
    }
    
    cfg = site_configs[site_id]
    
    data = []
    for t in timestamps:
        hour = t.hour
        weekday = t.weekday()
        
        # 温度：基础值 + 日周期 + 噪声
        temp = cfg['temp_mean'] + 5 * np.sin(2 * np.pi * hour / 24 - np.pi/2) + np.random.normal(0, cfg['temp_std'])
        
        # 业务量：日周期 + 周周期 + 站点特性
        hour_factor = 0.5 + 0.5 * np.sin(2 * np.pi * hour / 24 - np.pi/3)
        weekday_factor = 0.7 if weekday >= 5 else 1.0  # 周末略低
        traffic = 100 * hour_factor * weekday_factor * cfg['traffic_factor'] + np.random.normal(0, 15)
        traffic = max(0, traffic)
        
        # 能耗：基础功耗 + 温度影响 + 业务量影响
        power = cfg['base_power'] + 0.03 * traffic + 0.02 * (temp - cfg['temp_mean']) + np.random.normal(0, 0.2)
        power = max(0.5, power)
        
        data.append({
            'timestamp': t,
            'site_id': f'site_{site_id}',
            'power': round(power, 3),
            'traffic': round(traffic, 2),
            'temp': round(temp, 2),
            'hour': hour,
            'weekday': weekday
        })
    
    return pd.DataFrame(data)

# ==================== 2. 生成所有站点数据 ====================

print("=" * 50)
print("生成5个基站的Non-IID时序数据")
print("=" * 50)

all_sites_data = []
for site_id in range(5):
    df = generate_site_data(site_id, n_days=30)
    all_sites_data.append(df)
    print(f"站点 {site_id}: {len(df)}条记录, "
          f"功耗均值 {df['power'].mean():.2f}kW, "
          f"温度均值 {df['temp'].mean():.2f}°C")

# ==================== 3. 创建滑动窗口样本 ====================

def create_sequences(data, seq_len=24):
    """创建时序预测的滑动窗口样本"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])  # 过去24小时
        y.append(data[i+seq_len, 0])  # 预测下一时刻功耗（第一列）
    return np.array(X), np.array(y)

# ==================== 4. 生成联邦学习数据格式 ====================

print("\n" + "=" * 50)
print("生成联邦学习数据格式")
print("=" * 50)

# 创建主目录
os.makedirs('fl_data', exist_ok=True)

feature_cols = ['power', 'traffic', 'temp']  # 3个特征

for site_id, df in enumerate(all_sites_data):
    print(f"\n处理站点 {site_id}...")
    
    # 提取特征矩阵
    data_matrix = df[feature_cols].values
    
    # 归一化（每个站点独立归一化，模拟真实场景）
    mean = data_matrix.mean(axis=0)
    std = data_matrix.std(axis=0)
    data_normalized = (data_matrix - mean) / std
    
    # 创建序列
    X, y = create_sequences(data_normalized, seq_len=24)
    print(f"  总样本数: {len(X)}")
    
    # 划分训练/测试 (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 创建站点目录
    site_dir = f'fl_data/site_{site_id}'
    os.makedirs(site_dir, exist_ok=True)
    
    # 保存数据
    np.save(f'{site_dir}/X_train.npy', X_train)
    np.save(f'{site_dir}/y_train.npy', y_train)
    np.save(f'{site_dir}/X_test.npy', X_test)
    np.save(f'{site_dir}/y_test.npy', y_test)
    
    # 保存归一化参数（供后续使用）
    np.save(f'{site_dir}/mean.npy', mean)
    np.save(f'{site_dir}/std.npy', std)
    
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"  归一化参数 - 功耗均值: {mean[0]:.2f}, 温度均值: {mean[1]:.2f}")

# ==================== 5. 创建全局测试集 ====================

print("\n" + "=" * 50)
print("创建全局测试集")
print("=" * 50)

all_X_test = []
all_y_test = []

for site_id in range(5):
    site_dir = f'fl_data/site_{site_id}'
    X_test = np.load(f'{site_dir}/X_test.npy')
    y_test = np.load(f'{site_dir}/y_test.npy')
    all_X_test.append(X_test)
    all_y_test.append(y_test)

# 合并所有站点的测试集
global_X_test = np.concatenate(all_X_test, axis=0)
global_y_test = np.concatenate(all_y_test, axis=0)

# 保存全局测试集
os.makedirs('fl_data/global_test', exist_ok=True)
np.save('fl_data/global_test/X_test.npy', global_X_test)
np.save('fl_data/global_test/y_test.npy', global_y_test)

print(f"全局测试集大小: {global_X_test.shape}")

# ==================== 6. 可视化Non-IID分布（美化版本）====================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 统一配色方案
SITE_COLORS = ['#2563eb', '#059669', '#dc2626', '#7c3aed', '#f59e0b']

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('#f5f5f5')

for site_id in range(5):
    ax = axes[site_id // 3, site_id % 3]
    ax.set_facecolor('#ffffff')
    
    # 加载原始数据（未归一化的）
    df = all_sites_data[site_id]
    
    # 采样避免点太多
    sample = df.sample(n=min(500, len(df)))
    
    ax.scatter(sample['temp'], sample['power'], 
               c=SITE_COLORS[site_id], alpha=0.5, s=10)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Power (kW)', fontsize=11)
    ax.set_title(f'Site {site_id} Distribution',
                 fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', color='#dddddd')

# 合并对比图
ax = axes[1, 2]
ax.set_facecolor('#ffffff')

for site_id in range(5):
    df = all_sites_data[site_id].sample(n=500)
    ax.scatter(df['temp'], df['power'], 
               c=SITE_COLORS[site_id], alpha=0.5, s=10, 
               label=f'Site {site_id}')

ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('Power (kW)', fontsize=11)
ax.set_title('All Sites Comparison (Non-IID)',
             fontsize=14, fontweight='bold', pad=10)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', color='#dddddd')

plt.tight_layout()

# 保存高清图片
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(project_root, "results", f"day3_non_iid_distribution_{timestamp}.png")
plt.savefig(save_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='#f5f5f5')
plt.close()

print("\n✅ 美化后的Non-IID分布图已保存:", save_path)

# ==================== 7. 汇总统计 ====================

print("\n" + "=" * 50)
print("各站点数据统计汇总")
print("=" * 50)

stats = []
for site_id, df in enumerate(all_sites_data):
    stats.append({
        'site': site_id,
        'power_mean': df['power'].mean(),
        'power_std': df['power'].std(),
        'temp_mean': df['temp'].mean(),
        'temp_std': df['temp'].std(),
        'traffic_mean': df['traffic'].mean(),
        'n_samples': len(df)
    })

stats_df = pd.DataFrame(stats)
print(stats_df.round(2))

print("\n" + "=" * 50)
print("Day 3 完成！联邦学习数据已准备就绪")
print("下一步: Day 4 FedAvg算法实现")
print("=" * 50)
