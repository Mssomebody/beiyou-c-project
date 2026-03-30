#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成每个节点的 MinMax 参数映射
"""

import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OLD_PATH = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1"

node_minmax = {}
for node_dir in sorted(OLD_PATH.glob("node_*")):
    node_id = int(node_dir.name.split('_')[1])
    if node_id == 8025:
        continue
    scaler_file = node_dir / "scaler.pkl"
    if scaler_file.exists():
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        node_minmax[node_id] = (scaler.data_min_[0], scaler.data_max_[0])

output_path = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(node_minmax, f)

print(f"生成 {len(node_minmax)} 个节点的 MinMax 参数映射，保存至 {output_path}")