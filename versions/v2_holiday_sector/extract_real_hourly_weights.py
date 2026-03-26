#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 SHAP 计算过程中直接提取真实小时级权重
（需要重新运行 SHAP 计算时保存）
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 说明
print("="*60)
print("真实小时级权重提取说明")
print("="*60)
print("""
要获得真实小时级权重，需要：

1. 修改 shap_pytorch_final_fixed.py，在 compute_shap 函数中
   计算 hourly_importance = np.abs(shap_values).mean(axis=0)
   并保存到 JSON 中

2. 或者重新运行全量分析时保存中间结果

当前 hourly_weights_cn.png 是模拟数据，不是真实 SHAP 值。
如果需要真实数据，需要重新运行全量分析并修改代码。
""")

# 检查是否有保存的 hourly_importance
shap_dir = Path("D:/Desk/desk/beiyou_c_project/results/shap_analysis")
files = list(shap_dir.glob("shap_results_*.json"))

for f in files:
    import json
    with open(f, 'r') as fp:
        data = json.load(fp)
    if 'hourly_importance' in data:
        print(f"\n✅ {f.name} 包含小时级权重")
    else:
        print(f"\n❌ {f.name} 不包含小时级权重（只有平均重要性）")
