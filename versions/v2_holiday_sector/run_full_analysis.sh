#!/bin/bash
# 全量分析 + 对比分析脚本
# 自动顺序执行：4G全量 → 5G全量 → 对比分析

echo "=========================================="
echo "开始全量 SHAP 分析"
echo "=========================================="

# 1. 4G 全量分析
echo ""
echo ">>> 1/3: 4G 全量分析 (500基站, 全部样本)"
echo "预计时间: 15-25分钟"
python shap_pytorch_final_fixed.py --data_type 4g --max_stations 0 --samples_per_station 0 --epochs 30
if [ $? -ne 0 ]; then
    echo "4G 分析失败，退出"
    exit 1
fi

# 2. 5G 全量分析
echo ""
echo ">>> 2/3: 5G 全量分析 (500基站, 全部样本)"
echo "预计时间: 15-25分钟"
python shap_pytorch_final_fixed.py --data_type 5g --max_stations 0 --samples_per_station 0 --epochs 30
if [ $? -ne 0 ]; then
    echo "5G 分析失败，退出"
    exit 1
fi

# 3. 对比分析 + 权重映射
echo ""
echo ">>> 3/3: 对比分析 + 权重映射"
python compare_4g_5g_weights.py

echo ""
echo "=========================================="
echo "全部分析完成！"
echo "=========================================="
echo "结果目录: results/shap_analysis/"
echo "权重映射: results/barcelona_weights/"
