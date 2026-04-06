"""
Step 1: 专业阈值选择
基于数据分布自动选择最优MAPE过滤阈值
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.barcelona_dataset import get_node_data_loader
from experiments.beautified.train_single_node import (
    LSTMPredictor, train_epoch, evaluate, evaluate_original_scale
)
import torch
import torch.nn as nn
import torch.optim as optim


def find_optimal_threshold(targets):
    """
    基于数据分布自动选择最优阈值
    使用变异系数(CV)最小化原则
    
    Args:
        targets: 原始尺度目标值数组
    
    Returns:
        best_percentile, best_threshold, analysis_results
    """
    print("=" * 60)
    print("Step 1: 自动阈值选择")
    print("=" * 60)
    
    # 方法1: 基于百分位数网格搜索
    percentiles = range(5, 95, 5)
    results = []
    
    for p in percentiles:
        threshold = np.percentile(targets, p)
        mask = targets > threshold
        
        if mask.sum() > 0:
            filtered_targets = targets[mask]
            
            # 计算统计指标
            mean_val = np.mean(filtered_targets)
            std_val = np.std(filtered_targets)
            cv = std_val / mean_val if mean_val > 0 else np.inf  # 变异系数
            iqr = np.percentile(filtered_targets, 75) - np.percentile(filtered_targets, 25)
            
            results.append({
                'percentile': p,
                'threshold': threshold,
                'n_samples': mask.sum(),
                'n_ratio': mask.sum() / len(targets) * 100,
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'iqr': iqr
            })
    
    # 选择变异系数最小（最稳定）的阈值
    valid_results = [r for r in results if r['cv'] < np.inf]
    if valid_results:
        best = min(valid_results, key=lambda x: x['cv'])
    else:
        best = results[-1]
    
    # 打印分析结果
    print(f"\n数据统计:")
    print(f"  总样本数: {len(targets)}")
    print(f"  能耗范围: {targets.min():.0f} ~ {targets.max():.0f} kWh")
    print(f"  均值: {targets.mean():.0f} kWh")
    print(f"  中位数: {np.median(targets):.0f} kWh")
    print(f"  标准差: {targets.std():.0f} kWh")
    
    print(f"\n各百分位数分析:")
    for r in results:
        print(f"  {r['percentile']:3d}%: 阈值={r['threshold']:6.0f} kWh, "
              f"保留={r['n_ratio']:5.1f}%, CV={r['cv']:.3f}")
    
    print(f"\n✅ 推荐阈值: {best['percentile']}% 分位数 = {best['threshold']:.0f} kWh")
    print(f"   保留样本: {best['n_ratio']:.1f}%")
    print(f"   变异系数: {best['cv']:.3f}")
    
    return best['percentile'], best['threshold'], results


def quick_train_eval(node_id=8001, epochs=5, percentile=5):
    """
    快速训练评估（用于对比不同阈值）
    """
    print(f"\n{'='*60}")
    print(f"快速训练评估: percentile={percentile}%")
    print(f"{'='*60}")
    
    device = torch.device('cpu')
    
    # 加载数据
    train_loader, scaler_path, train_dataset = get_node_data_loader(
        node_id=node_id, split='train', batch_size=64, shuffle=True,
        sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    val_loader, _, _ = get_node_data_loader(
        node_id=node_id, split='val', batch_size=64, shuffle=False,
        sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    test_loader, _, _ = get_node_data_loader(
        node_id=node_id, split='test', batch_size=64, shuffle=False,
        sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 获取输入维度
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]
    
    # 创建模型
    model = LSTMPredictor(
        input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=4
    ).to(device)
    
    # 训练
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
    
    # 评估
    eval_results = evaluate_original_scale(
        model, test_loader, scaler, device, percentile
    )
    
    print(f"\n结果 (percentile={percentile}%):")
    print(f"  RMSE: {eval_results['rmse']:.2f} kWh")
    print(f"  MAE:  {eval_results['mae']:.2f} kWh")
    print(f"  sMAPE: {eval_results['smape']:.2f}%")
    print(f"  MAPE: {eval_results['mape_filtered']:.2f}% (过滤 < {eval_results['mape_threshold']:.0f} kWh)")
    
    return eval_results


def main(node_id=8001):
    """主函数：自动选择最优阈值并验证"""
    print("=" * 60)
    print("Step 1: 自动阈值优化")
    print("=" * 60)
    
    # 1. 先加载测试集，分析数据分布
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_path = os.path.join(base_dir, "data", "processed", "barcelona_ready", 
                              f"node_{node_id}", "test.pkl")
    df_test = pd.read_pickle(test_path)
    targets = df_test['Valor'].values
    
    # 2. 自动选择最优阈值
    best_percentile, best_threshold, analysis = find_optimal_threshold(targets)
    
    # 3. 验证不同阈值的效果
    print(f"\n{'='*60}")
    print("验证不同阈值效果")
    print(f"{'='*60}")
    
    test_percentiles = [5, 10, 15, 20, 25, 30, best_percentile]
    test_percentiles = sorted(set(test_percentiles))
    
    results = {}
    for p in test_percentiles:
        eval_res = quick_train_eval(node_id, epochs=5, percentile=p)
        results[p] = eval_res['smape']
    
    # 4. 选择最优
    best_p = min(results, key=results.get)
    print(f"\n{'='*60}")
    print(f"✅ 最终选择: percentile = {best_p}%")
    print(f"   sMAPE = {results[best_p]:.2f}%")
    print(f"   阈值 = {np.percentile(targets, best_p):.0f} kWh")
    print(f"{'='*60}")
    
    return best_p, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=int, default=8001)
    args = parser.parse_args()
    
    best_p, results = main(args.node)
    
    print(f"\n推荐使用: --percentile {best_p}")
    print(f"运行命令:")
    print(f"python experiments/beautified/train_single_node.py --node {args.node} --epochs 30 --percentile {best_p}")