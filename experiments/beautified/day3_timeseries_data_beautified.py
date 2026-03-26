# day3_timeseries_data_beautified.py
# 第3天：时序数据基础 - 基站能耗数据生成与可视化（美化版本）
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os

# 设置随机种子，结果可复现
np.random.seed(42)

# ==================== 1. 生成模拟基站数据 ====================

def generate_base_station_data(station_id, n_days=30, freq='h'):
    '''
    生成单个基站的能耗时序数据

    参数:
        station_id: 基站编号，如 'Station_0'
        n_days: 天数
        freq: 采样频率，'h'=每小时，'15min'=15分钟
    返回:
        DataFrame: [timestamp, station_id, traffic_gb, power_kw]
    '''
    # 生成时间序列
    start_time = datetime(2024, 1, 1)
    periods = n_days * 24 if freq == 'h' else n_days * 96  # 每小时或15分钟
    timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)

    # 基础功耗（不同基站略有差异）
    base_power = 3.0 + int(station_id.split('_')[1]) * 0.1  # 3.0, 3.1, 3.2...

    data = []
    for t in timestamps:
        # 时间特征：小时、星期几
        hour = t.hour
        weekday = t.weekday()  # 0=周一, 6=周日

        # 业务量：白天高(12-20点)，晚上低，周末略低
        # 基础模式 + 周期性 + 随机噪声
        hour_factor = 1 + 0.6 * np.sin((hour - 6) * np.pi / 12)  # 峰值在14点左右
        if hour < 6 or hour > 23:
            hour_factor *= 0.3  # 深夜降低
        weekend_factor = 0.85 if weekday >= 5 else 1.0  # 周末略低

        traffic = 100 * hour_factor * weekend_factor + np.random.normal(0, 15)
        traffic = max(0, traffic)  # 不能为负

        # 功耗 = 基础功耗 + 业务相关功耗 + 温度影响(简化) + 噪声
        power = base_power + 0.025 * traffic + np.random.normal(0, 0.2)
        power = max(0.5, power)  # 最低功耗

        data.append({
            'timestamp': t,
            'station_id': station_id,
            'traffic_gb': round(traffic, 2),
            'power_kw': round(power, 3),
            'hour': hour,
            'weekday': weekday
        })

    return pd.DataFrame(data)


# ==================== 2. 生成多基站数据 ====================

print("=" * 50)
print("生成10个基站，30天，每小时采样的模拟数据")
print("=" * 50)

n_stations = 5
all_data = []

for i in range(n_stations):
    station_id = f'Station_{i}'
    df_station = generate_base_station_data(station_id, n_days=30, freq='h')
    all_data.append(df_station)
    print(f"{station_id}: {len(df_station)}条记录, "
          f"平均功耗{df_station['power_kw'].mean():.2f}kW")

# 合并所有数据
df_all = pd.concat(all_data, ignore_index=True)

print(f"\n总数据量: {len(df_all)}条")
print(f"\n前5行:")
print(df_all.head())

# ==================== 3. 数据探索 ====================

print("\n" + "=" * 50)
print("数据统计信息")
print("=" * 50)

# 按基站分组统计
stats = df_all.groupby('station_id').agg({
    'traffic_gb': ['mean', 'std', 'min', 'max'],
    'power_kw': ['mean', 'std', 'min', 'max']
}).round(2)

print(stats)

# ==================== 4. 可视化（美化版本）====================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 统一配色方案
COLOR_MAIN = '#2563eb'
COLOR_TRAIN = '#059669'
COLOR_VAL = '#dc2626'
COLOR_TEST = '#7c3aed'
STATION_COLORS = ['#2563eb', '#059669', '#dc2626', '#7c3aed', '#f59e0b']

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#f5f5f5')

# 子图1: 5个基站的原始负载曲线
ax1 = axes[0, 0]
ax1.set_facecolor('#ffffff')

for i in range(min(n_stations, 5)):
    station_data = df_all[df_all['station_id'] == f'Station_{i}']
    ax1.plot(station_data['timestamp'], station_data['power_kw'],
             color=STATION_COLORS[i], linewidth=1.5,
             label=f'Station_{i}', alpha=0.7)

ax1.set_title('5 Base Stations: Original Load Curves', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Time', fontsize=11)
ax1.set_ylabel('Power (kW)', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--', color='#dddddd')
ax1.legend(fontsize=9, framealpha=0.9)
ax1.tick_params(axis='x', rotation=45)

# 子图2: 单个基站的周模式（7天）
ax2 = axes[0, 1]
ax2.set_facecolor('#ffffff')

station_0 = df_all[df_all['station_id'] == 'Station_0']
first_7_days = station_0.iloc[:7*24]

ax2.plot(first_7_days['timestamp'], first_7_days['power_kw'],
         color=COLOR_MAIN, linewidth=2.0, marker='o', markersize=3)
ax2.set_title('Station_0: Weekly Pattern (7 Days)',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Time', fontsize=11)
ax2.set_ylabel('Power (kW)', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--', color='#dddddd')
ax2.tick_params(axis='x', rotation=45)

# 子图3: 数据分布直方图
ax3 = axes[1, 0]
ax3.set_facecolor('#ffffff')

ax3.hist(df_all['power_kw'], bins=30, color=COLOR_MAIN, alpha=0.7, edgecolor='white')
ax3.set_title('Power Distribution Histogram',
              fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Power (kW)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.grid(True, alpha=0.3, linestyle='--', color='#dddddd')

# 子图4: 训练/验证/测试划分示意
ax4 = axes[1, 1]
ax4.set_facecolor('#ffffff')

# 绘制划分示意
total_samples = len(station_0)
train_end = int(0.7 * total_samples)
val_end = int(0.85 * total_samples)

ax4.axvspan(0, train_end, color=COLOR_TRAIN, alpha=0.3, label='Train (70%)')
ax4.axvspan(train_end, val_end, color=COLOR_VAL, alpha=0.3, label='Validation (15%)')
ax4.axvspan(val_end, total_samples, color=COLOR_TEST, alpha=0.3, label='Test (15%)')

ax4.plot(range(total_samples), station_0['power_kw'].values,
         color='#333333', linewidth=0.5, alpha=0.5)
ax4.set_title('Train/Validation/Test Split',
              fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('Sample Index', fontsize=11)
ax4.set_ylabel('Power (kW)', fontsize=11)
ax4.grid(True, alpha=0.3, linestyle='--', color='#dddddd')
ax4.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()

# 保存高清图片
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(project_root, "results", f"day3_timeseries_overview_{timestamp}.png")
plt.savefig(save_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='#f5f5f5')
plt.close()

print("\n✅ 美化后的图表已保存:", save_path)

# ==================== 5. 保存数据 ====================

# 保存为CSV，供后续LSTM使用
output_file = "data/raw/base_station_data_10stations_30days.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_all.to_csv(output_file, index=False)
print(f"\n数据已保存: {output_file}")
print(f"文件大小: {len(df_all)}行 × {len(df_all.columns)}列")

print("\n" + "=" * 50)
print("Day3 完成！下一步: Day4 LSTM模型")
print("=" * 50)
