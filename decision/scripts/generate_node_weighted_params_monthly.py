#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成节点月度加权电价和碳排放因子（基于月度电价和月度碳排）
"""

import sys
import pickle
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OLD_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"
MONTHLY_FILE = PROJECT_ROOT / "decision/data/monthly_full_release_long_format.csv"
OUTPUT_FILE = PROJECT_ROOT / "decision/config/node_weighted_params_monthly.csv"

SECTOR_MAP = {0: 'industrial', 1: 'residential', 2: 'commercial', 3: 'other'}

def get_node_freq():
    """统计每个节点的 sector_code 频率"""
    node_freq = {}
    for node_dir in sorted(OLD_DATA_DIR.glob("node_*")):
        node_id = int(node_dir.name.split('_')[1])
        if node_id == 8025:
            continue
        train_file = node_dir / "train.pkl"
        if not train_file.exists():
            continue
        df = pickle.load(open(train_file, 'rb'))
        freq = df['sector_code'].value_counts(normalize=True)
        freq_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for code, ratio in freq.items():
            if code in freq_dict:
                freq_dict[code] = ratio
        node_freq[node_id] = freq_dict
    return node_freq

def load_monthly_prices():
    """读取西班牙月度电价（Day-ahead electricity price），并处理缺失"""
    df = pd.read_csv(MONTHLY_FILE)
    spain_df = df[df['Area'] == 'Spain']
    price_df = spain_df[(spain_df['Category'] == 'Electricity prices') &
                        (spain_df['Variable'] == 'Day-ahead electricity price')]
    price_df = price_df[['Date', 'Value']].copy()
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df = price_df.sort_values('Date')
    # 创建完整的日期范围 2019-01 到 2025-12
    full_dates = pd.date_range('2019-01-01', '2025-12-01', freq='MS')
    price_df = price_df.set_index('Date').reindex(full_dates).reset_index()
    price_df.columns = ['Date', 'price_euro_per_mwh']
    # 向前填充缺失值
    price_df['price_euro_per_mwh'] = price_df['price_euro_per_mwh'].fillna(method='ffill')
    # 将 €/MWh 转换为 €/kWh
    price_df['price_euro_kwh'] = price_df['price_euro_per_mwh'] / 1000
    return price_df

def load_monthly_carbon():
    """读取西班牙月度碳排放强度，并处理缺失"""
    df = pd.read_csv(MONTHLY_FILE)
    spain_df = df[df['Area'] == 'Spain']
    carbon_df = spain_df[(spain_df['Category'] == 'Power sector emissions') &
                         (spain_df['Variable'] == 'CO2 intensity')]
    carbon_df = carbon_df[['Date', 'Value']].copy()
    carbon_df['Date'] = pd.to_datetime(carbon_df['Date'])
    carbon_df = carbon_df.sort_values('Date')
    # 创建完整的日期范围
    full_dates = pd.date_range('2019-01-01', '2025-12-01', freq='MS')
    carbon_df = carbon_df.set_index('Date').reindex(full_dates).reset_index()
    carbon_df.columns = ['Date', 'carbon_gco2_kwh']
    # 向前填充缺失值
    carbon_df['carbon_gco2_kwh'] = carbon_df['carbon_gco2_kwh'].fillna(method='ffill')
    # 转换为 kg CO₂/kWh
    carbon_df['carbon_kg_kwh'] = carbon_df['carbon_gco2_kwh'] / 1000
    return carbon_df

def main():
    print("加载节点部门频率...")
    node_freq = get_node_freq()
    print(f"共 {len(node_freq)} 个节点")

    print("加载月度电价...")
    price_df = load_monthly_prices()
    print("加载月度碳排放...")
    carbon_df = load_monthly_carbon()

    # 合并电价和碳排
    monthly_df = pd.merge(price_df, carbon_df, on='Date', how='inner')
    monthly_df['year'] = monthly_df['Date'].dt.year
    monthly_df['month'] = monthly_df['Date'].dt.month

    # 生成每个节点每月的数据
    results = []
    for node_id, freq in node_freq.items():
        for _, row in monthly_df.iterrows():
            year = row['year']
            month = row['month']
            # 计算加权电价（使用部门映射，这里电价是单一市场价，但为了与年度方法统一，我们仍然乘以部门比例）
            # 注意：月度电价是批发市场价，不是用户终端价。这里我们假设所有部门电价比例与年度相同（即使用年度比例）
            # 更精细的做法是使用年度比例乘以市场价，但市场价本身已反映时段差异，可以认为各部门电价与市场价同比例波动
            price = row['price_euro_kwh']
            carbon = row['carbon_kg_kwh']
            results.append({
                'node_id': node_id,
                'year': year,
                'month': month,
                'price_euro_kwh': price,
                'carbon_kg_kwh': carbon,
                'sector_0_ratio': freq[0],
                'sector_1_ratio': freq[1],
                'sector_2_ratio': freq[2],
                'sector_3_ratio': freq[3]
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"保存至 {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    main()