"""
巴塞罗那基站能耗数据预处理 v1
只包含基础特征：能耗 + 部门 + 节假日 + 周末
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_INPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "barcelona_ready_v1")

WINDOW_SIZE = 28
PREDICT_SIZE = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)


def load_and_merge_data():
    """加载2019-2025年所有数据并合并"""
    print("=" * 60)
    print("阶段1.1: 加载数据")
    print("=" * 60)
    
    files = [f for f in os.listdir(DATA_INPUT_DIR) 
             if f.endswith('.csv') and 'consum' in f]
    
    all_dfs = []
    for file in sorted(files):
        file_path = os.path.join(DATA_INPUT_DIR, file)
        df = pd.read_csv(file_path)
        print(f"  ✓ {file}: {len(df):,} 行")
        all_dfs.append(df)
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  ✅ 合并完成: {len(df):,} 行")
    
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(['Data', 'Codi_Postal']).reset_index(drop=True)
    print(f"  时间范围: {df['Data'].min().date()} ~ {df['Data'].max().date()}")
    
    return df


def clean_data(df):
    """过滤无效数据，添加基础特征"""
    print("\n" + "=" * 60)
    print("阶段1.2: 数据清洗")
    print("=" * 60)
    
    original_len = len(df)
    df = df[df['Tram_Horari'] != 'No consta']
    print(f"  过滤'No consta'时段: {original_len:,} → {len(df):,} 行")
    
    # 时段编码
    hour_mapping = {
        'De 00:00:00 a 05:59:59 h': 0,
        'De 06:00:00 a 11:59:59 h': 1,
        'De 12:00:00 a 17:59:59 h': 2,
        'De 18:00:00 a 23:59:59 h': 3
    }
    df['hour_code'] = df['Tram_Horari'].map(hour_mapping)
    
    # 部门编码
    sector_mapping = {
        'Indústria': 0,
        'Residencial': 1,
        'Serveis': 2,
        'No especificat': 3
    }
    df['sector_code'] = df['Sector_Economic'].map(sector_mapping)
    
    # 周末特征
    df['is_weekend'] = (df['Data'].dt.weekday >= 5).astype(int)
    
    # 节假日特征
    spanish_holidays_fixed = [
        (1, 1), (1, 6), (5, 1), (8, 15), (10, 12),
        (11, 1), (12, 6), (12, 8), (12, 25)
    ]
    holidays = []
    for year in range(2019, 2026):
        for month, day in spanish_holidays_fixed:
            holidays.append(f"{year}-{month:02d}-{day:02d}")
        holidays.append(f"{year}-04-01")
    df['is_holiday'] = df['Data'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)
    
    print(f"  ✅ 清洗完成: {len(df):,} 行")
    
    return df


def group_by_postal_code(df):
    """按邮编分组"""
    print("\n" + "=" * 60)
    print("阶段1.3: 按邮编分组")
    print("=" * 60)
    
    postal_codes = sorted(df['Codi_Postal'].unique())
    print(f"  总邮编数: {len(postal_codes)}")
    
    postal_data = {}
    for code in postal_codes:
        postal_data[code] = df[df['Codi_Postal'] == code].copy()
        if code % 10 == 0 or code == postal_codes[0]:
            print(f"  {code}: {len(postal_data[code]):,} 行")
    
    return postal_data


def add_sector_onehot(df):
    """添加部门 One-Hot 编码"""
    sector_onehot = pd.get_dummies(df['sector_code'], prefix='sector')
    for i in range(4):
        col = f'sector_{i}'
        if col not in sector_onehot.columns:
            sector_onehot[col] = 0
    df = pd.concat([df, sector_onehot], axis=1)
    return df


def time_split(data_dict):
    """按时间顺序划分"""
    print("\n" + "=" * 60)
    print("阶段1.4: 时序划分")
    print("=" * 60)
    
    split_data = {}
    for code, df in data_dict.items():
        df = df.sort_values('Data').reset_index(drop=True)
        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        split_data[code] = {
            'train': df.iloc[:train_end],
            'val': df.iloc[train_end:val_end],
            'test': df.iloc[val_end:]
        }
        
        if code % 10 == 0 or code == list(data_dict.keys())[0]:
            print(f"  {code}: 训练={len(split_data[code]['train']):,} | "
                  f"验证={len(split_data[code]['val']):,} | "
                  f"测试={len(split_data[code]['test']):,}")
    
    return split_data


def normalize_node_data(split_data):
    """归一化"""
    print("\n" + "=" * 60)
    print("阶段1.5: 归一化处理")
    print("=" * 60)
    
    normalized_data = {}
    scalers = {}
    
    for code, splits in split_data.items():
        train_values = splits['train']['Valor'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_values)
        scalers[code] = scaler
        
        normalized_splits = {}
        for split_name in ['train', 'val', 'test']:
            df = splits[split_name].copy()
            values = df['Valor'].values.reshape(-1, 1)
            df['Valor_norm'] = scaler.transform(values).flatten()
            normalized_splits[split_name] = df
        
        normalized_data[code] = normalized_splits
        print(f"  {code}: 归一化完成")
    
    return normalized_data, scalers


def save_preprocessed_data(normalized_data, scalers, metadata):
    """保存数据"""
    print("\n" + "=" * 60)
    print("阶段1.6: 保存数据")
    print("=" * 60)
    
    for code, splits in normalized_data.items():
        node_path = os.path.join(DATA_OUTPUT_DIR, f"node_{code}")
        os.makedirs(node_path, exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            file_path = os.path.join(node_path, f"{split_name}.pkl")
            splits[split_name].to_pickle(file_path)
        
        with open(os.path.join(node_path, "scaler.pkl"), 'wb') as f:
            pickle.dump(scalers[code], f)
    
    metadata_path = os.path.join(DATA_OUTPUT_DIR, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"  ✅ 数据保存到: {DATA_OUTPUT_DIR}")


def preprocess_barcelona():
    """主函数"""
    print("\n" + "=" * 60)
    print("巴塞罗那基站能耗数据预处理 v1")
    print("=" * 60)
    
    df = load_and_merge_data()
    df = clean_data(df)
    postal_data = group_by_postal_code(df)
    
    # 添加部门 One-Hot
    for code in postal_data:
        postal_data[code] = add_sector_onehot(postal_data[code])
    
    split_data = time_split(postal_data)
    normalized_data, scalers = normalize_node_data(split_data)
    
    metadata = {
        'num_nodes': len(normalized_data),
        'postal_codes': list(normalized_data.keys()),
        'window_size': WINDOW_SIZE,
        'predict_size': PREDICT_SIZE,
        'version': 'v1',
        'date_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_preprocessed_data(normalized_data, scalers, metadata)
    
    print("\n✅ v1 预处理完成！")
    return normalized_data, scalers, metadata


if __name__ == "__main__":
    preprocess_barcelona()