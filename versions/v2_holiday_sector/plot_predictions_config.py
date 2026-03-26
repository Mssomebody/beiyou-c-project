"""
预测曲线画图 - 可配置版
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='画预测曲线图')
    parser.add_argument('--model', type=str, 
                        default=r'D:\Desk\desk\beiyou_c_project\results\two_stage\two_stage_model_20260323_183730.pth',
                        help='模型文件路径')
    parser.add_argument('--data', type=str,
                        default=r'D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025',
                        help='数据路径')
    parser.add_argument('--node', type=int, default=8001, help='节点ID')
    parser.add_argument('--hidden', type=int, default=96, help='隐藏层维度')
    parser.add_argument('--layers', type=int, default=4, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2298, help='Dropout率')
    parser.add_argument('--samples', type=int, default=50, help='显示样本数')
    parser.add_argument('--output', type=str, default='results/predictions.png', help='输出图片路径')
    
    args = parser.parse_args()
    
    # 加载数据
    data_dir = os.path.join(args.data, f"node_{args.node}")
    test_dataset = BarcelonaDataset(os.path.join(data_dir, "test.pkl"))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载模型
    sample_x, _ = test_dataset[0]
    input_dim = sample_x.shape[1]
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=args.hidden, 
                          num_layers=args.layers, output_dim=4, dropout=args.dropout)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    
    # 预测
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            all_preds.append(output.numpy())
            all_targets.append(y.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 画图
    plt.figure(figsize=(14, 6))
    plt.plot(all_targets[:args.samples, 0], label='真实值', color='blue', linewidth=1.5)
    plt.plot(all_preds[:args.samples, 0], label='预测值', color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('样本序号')
    plt.ylabel('归一化能耗')
    plt.title(f'节点{args.node}预测效果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"✅ 图片保存: {args.output}")
    
    # 误差统计
    errors = np.abs(all_targets - all_preds)
    print(f"\n误差统计:")
    print(f"  平均绝对误差: {np.mean(errors):.4f}")
    print(f"  最大误差: {np.max(errors):.4f}")

if __name__ == "__main__":
    main()
