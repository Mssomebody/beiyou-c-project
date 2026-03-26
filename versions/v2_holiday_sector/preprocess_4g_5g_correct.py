"""
4G/5G数据预处理 - 正确版
保存原始能耗值，便于计算真实 sMAPE
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 配置
RAW_DIR = Path("D:/Desk/desk/beiyou_c_project/data/raw/tsinghua")
OUT_DIR = Path("D:/Desk/desk/beiyou_c_project/data/processed/tsinghua_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 公共特征列（4G和5G共有的）
COMMON_FEATURES = [
    'PRB Usage Ratio (%)',
    'Traffic Volume (KByte)',
    'Number of Users'
]

# 5G额外特征（暂不使用，保持对齐）
# 先用公共特征，保证4G和5G维度一致


def process_dataset(file_name, data_type, max_stations=None):
    """
    处理单个数据集
    
    Args:
        file_name: 文件名
        data_type: '4g' 或 '5g'
        max_stations: 最大基站数（用于测试）
    
    Returns:
        处理结果统计
    """
    print(f"\n{'='*50}")
    print(f"处理 {data_type} 数据")
    print(f"{'='*50}")
    
    # 加载数据
    df = pd.read_csv(RAW_DIR / file_name)
    print(f"  原始行数: {len(df):,}")
    
    # 计算总能耗
    df['Total_Energy'] = df['BBU Energy (W)'] + df['RRU Energy (W)']
    
    # 获取基站列表
    stations = df['Base Station ID'].unique()
    if max_stations:
        stations = stations[:max_stations]
    print(f"  基站数: {len(stations)}")
    
    # 输出目录
    data_dir = OUT_DIR / data_type
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录所有基站的统计信息
    all_stats = []
    
    for station_id in tqdm(stations, desc=f"处理 {data_type} 基站"):
        station_df = df[df['Base Station ID'] == station_id].copy()
        
        # 按时间排序
        station_df = station_df.sort_values('Timestamp')
        
        # 提取特征（公共特征）
        X = station_df[COMMON_FEATURES].values.astype(np.float32)
        y = station_df['Total_Energy'].values.astype(np.float32).reshape(-1, 1)
        
        # 添加时间特征（小时正弦/余弦）
        hour = station_df['Timestamp'].str.split(':').str[0].astype(int).values
        hour_sin = np.sin(2 * np.pi * hour / 24).reshape(-1, 1)
        hour_cos = np.cos(2 * np.pi * hour / 24).reshape(-1, 1)
        X = np.hstack([X, hour_sin, hour_cos])
        
        # 归一化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_norm = scaler_X.fit_transform(X)
        y_norm = scaler_y.fit_transform(y)
        
        # 保存数据
        station_dir = data_dir / f"station_{station_id}"
        station_dir.mkdir(exist_ok=True)
        
        data = {
            'station_id': station_id,
            'n_samples': len(station_df),
            'features_norm': X_norm,           # 归一化特征
            'target_norm': y_norm,              # 归一化目标
            'target_raw': y,                    # 原始能耗（用于评估）
            'features_raw': X,                  # 原始特征（用于分析）
            'timestamps': station_df['Timestamp'].values,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
        }
        
        with open(station_dir / 'data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        all_stats.append({
            'station_id': station_id,
            'n_samples': len(station_df),
            'energy_mean': y.mean(),
            'energy_std': y.std()
        })
    
    # 保存汇总
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(data_dir / 'stats.csv', index=False)
    
    # 打印统计
    print(f"\n  完成: {len(stations)} 个基站")
    print(f"  平均样本数: {stats_df['n_samples'].mean():.0f}")
    print(f"  平均能耗: {stats_df['energy_mean'].mean():.2f} W")
    
    return stats_df


def main():
    print("="*60)
    print("4G/5G 数据预处理（正确版）")
    print("="*60)
    print("保存内容：")
    print("  - features_norm: 归一化特征")
    print("  - target_norm: 归一化目标")
    print("  - target_raw: 原始能耗（用于计算真实sMAPE）")
    print("  - scaler_X, scaler_y: 归一化器")
    
    # 处理4G数据（取前500个基站测试）
    process_dataset('Performance_4G_Weekday.txt', '4g', max_stations=500)
    
    # 处理5G数据（取前500个基站测试）
    process_dataset('Performance_5G_Weekday.txt', '5g', max_stations=500)
    
    print("\n" + "="*60)
    print("预处理完成！")
    print("="*60)
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
