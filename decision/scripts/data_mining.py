#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据挖掘：从训练数据中计算每个小时的平均能耗，识别高峰时段
输出：decision/config/peak_hours.json
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"
OUTPUT_JSON = PROJECT_ROOT / "decision" / "config" / "peak_hours.json"

def main():
    if not TRAIN_DATA_DIR.exists():
        print(f"Error: Training data directory not found: {TRAIN_DATA_DIR}")
        sys.exit(1)

    # 收集所有节点的训练数据中的能耗和小时编码
    hour_energy = []
    for node_dir in TRAIN_DATA_DIR.glob("node_*"):
        train_pkl = node_dir / "train.pkl"
        if not train_pkl.exists():
            continue
        df = pd.read_pickle(train_pkl)
        if 'hour_code' not in df.columns or 'Valor' not in df.columns:
            print(f"Warning: {train_pkl} missing hour_code or Valor, skip")
            continue
        # 提取小时和能耗
        hour_energy.append(df[['hour_code', 'Valor']])

    if not hour_energy:
        print("Error: No valid training data found with hour_code and Valor")
        sys.exit(1)

    all_data = pd.concat(hour_energy, ignore_index=True)
    # 按小时计算平均能耗
    hourly_avg = all_data.groupby('hour_code')['Valor'].mean().sort_values(ascending=False)
    # 选择前6个高峰时段（可配置）
    top_n = 6
    peak_hours = hourly_avg.head(top_n).index.tolist()
    print(f"Top {top_n} peak hours (by average energy): {peak_hours}")
    print("Hourly averages:", hourly_avg.to_dict())

    # 保存到JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({
            "peak_hours": peak_hours,
            "top_n": top_n,
            "hourly_avg": hourly_avg.to_dict(),
            "description": "Peak hours mined from training data (hour_code with highest average Valor)"
        }, f, indent=2)
    print(f"Peak hours saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()