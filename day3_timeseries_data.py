# day3_timeseries_data.py
# 第3天：时序数据基础 - 基站能耗数据生成与可视化
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

n_stations = 10
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

# ==================== 4. 可视化 ====================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 单个基站的时间序列（Station_0）
ax1 = axes[0, 0]
station_0 = df_all[df_all['station_id'] == 'Station_0']
ax1.plot(station_0['timestamp'], station_0['power_kw'],
         label='Power (kW)', color='blue', alpha=0.7)
ax1_twin = ax1.twinx()
ax1_twin.plot(station_0['timestamp'], station_0['traffic_gb'],
              label='Traffic (GB)', color='red', alpha=0.5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Power (kW)', color='blue')
ax1_twin.set_ylabel('Traffic (GB)', color='red')
ax1.set_title('Station_0: 30天能耗与业务量')
ax1.tick_params(axis='x', rotation=45)

# 图2: 所有基站的平均功耗对比
ax2 = axes[0, 1]
station_means = df_all.groupby('station_id')['power_kw'].mean().sort_values()
colors = plt.cm.viridis(np.linspace(0, 1, len(station_means)))
bars = ax2.bar(range(len(station_means)), station_means.values, color=colors)
ax2.set_xticks(range(len(station_means)))
ax2.set_xticklabels(station_means.index, rotation=45, ha='right')
ax2.set_ylabel('Average Power (kW)')
ax2.set_title('各基站平均功耗对比')

# 图3: 日内模式（所有基站平均）
ax3 = axes[1, 0]
hourly_pattern = df_all.groupby('hour').agg({
    'power_kw': 'mean',
    'traffic_gb': 'mean'
})
ax3.plot(hourly_pattern.index, hourly_pattern['power_kw'],
         'o-', label='Power', color='blue')
ax3_twin = ax3.twinx()
ax3_twin.plot(hourly_pattern.index, hourly_pattern['traffic_gb'],
              's-', label='Traffic', color='red')
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Power (kW)', color='blue')
ax3_twin.set_ylabel('Traffic (GB)', color='red')
ax3.set_title('日内平均模式')
ax3.set_xticks(range(0, 24, 2))

# 图4: 功耗vs业务量散点图
ax4 = axes[1, 1]
sample = df_all.sample(n=min(2000, len(df_all)))  # 采样避免点太多
scatter = ax4.scatter(sample['traffic_gb'], sample['power_kw'],
                      c=sample['hour'], cmap='viridis', alpha=0.5)
ax4.set_xlabel('Traffic (GB)')
ax4.set_ylabel('Power (kW)')
ax4.set_title('功耗与业务量关系（颜色=小时）')
plt.colorbar(scatter, ax=ax4, label='Hour')

plt.tight_layout()
plt.savefig('day3_timeseries_overview.png', dpi=150, bbox_inches='tight')
print("\n图表已保存: day3_timeseries_overview.png")

# ==================== 5. 保存数据 ====================

# 保存为CSV，供后续LSTM使用
output_file = 'base_station_data_10stations_30days.csv'
df_all.to_csv(output_file, index=False)
print(f"\n数据已保存: {output_file}")
print(f"文件大小: {len(df_all)}行 × {len(df_all.columns)}列")

# 额外保存一个Station_0的单独文件，方便调试
station_0.to_csv('station_0_single.csv', index=False)
print(f"Station_0单独文件: station_0_single.csv ({len(station_0)}行)")

print("\n" + "=" * 50)
print("Day3 完成！下一步: Day4 LSTM模型")
print("=" * 50)
