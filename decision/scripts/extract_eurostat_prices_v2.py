#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从欧盟统计局 TSV 文件中提取西班牙家庭和非家庭电价，
生成四个部门 2019-2025 年的年度平均电价（€/kWh）。
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
HOUSEHOLD_FILE = BASE / "decision/data/estat_nrg_pc_204.tsv/estat_nrg_pc_204.tsv"
NONHOUSEHOLD_FILE = BASE / "decision/data/estat_nrg_pc_205.tsv/estat_nrg_pc_205.tsv"
OUTPUT_FILE = BASE / "decision/config/eurostat_prices.csv"

def read_eurostat_tsv(file_path, target_pattern):
    """
    读取 TSV 文件，返回 DataFrame，其中包含西班牙数据，列名为年份-半年
    target_pattern: 用于匹配目标行的字符串，如 'S,E7000,KWH1000-2499,KWH,I_TAX,EUR,ES'
    """
    # 读取所有行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 第一行是列名
    header = lines[0].strip().split('\t')
    # 列名格式：第一列是 'freq,siec,nrg_cons,unit,tax,currency,geo\TIME_PERIOD'
    # 后续列是年份-半年，如 '2007-S1','2007-S2',...
    # 我们提取后续列作为时间列
    time_cols = header[1:]  # 从第二列开始是时间序列
    # 提取年份和半年
    years = []
    for col in time_cols:
        if '-' in col:
            y, s = col.split('-')
            years.append((int(y), s))
        else:
            years.append((None, None))
    
    # 查找目标行
    target_line = None
    for line in lines[1:]:  # 跳过列名行
        if target_pattern in line:
            target_line = line.strip().split('\t')
            break
    
    if target_line is None:
        return None, None
    
    # 价格数据从第二列开始
    price_vals = target_line[1:]
    # 创建 DataFrame
    data = []
    for i, price in enumerate(price_vals):
        if price.strip() == ':' or price.strip() == '':
            continue
        y, s = years[i]
        if y is not None:
            data.append({'year': y, 'half': s, 'price': float(price)})
    df = pd.DataFrame(data)
    return df, years

def main():
    # 家庭用户
    household_df, _ = read_eurostat_tsv(HOUSEHOLD_FILE, 'S,E7000,KWH1000-2499,KWH,I_TAX,EUR,ES')
    if household_df is None:
        print("未找到西班牙家庭用户数据")
        return
    
    # 非家庭用户
    nonhousehold_df, _ = read_eurostat_tsv(NONHOUSEHOLD_FILE, 'S,E7000,MWH20-499,KWH,I_TAX,EUR,ES')
    if nonhousehold_df is None:
        print("未找到西班牙非家庭用户数据")
        return
    
    # 计算年度平均值
    household_annual = household_df.groupby('year')['price'].mean().to_dict()
    nonhousehold_annual = nonhousehold_df.groupby('year')['price'].mean().to_dict()
    
    # 生成四个部门电价（2019-2025）
    results = []
    for y in range(2019, 2026):
        res = household_annual.get(y)
        non = nonhousehold_annual.get(y)
        if res is None or non is None:
            continue
        industrial = non * 0.9
        commercial = non * 1.1
        other = non
        results.append({
            'year': y,
            'industrial': industrial,
            'residential': res,
            'commercial': commercial,
            'other': other
        })
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"电价数据已保存至 {OUTPUT_FILE}")
    print(df)

if __name__ == "__main__":
    main()