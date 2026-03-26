#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接评估24节点模型 - 不依赖配置文件"""

import sys
sys.path.insert(0, 'D:/Desk/desk/beiyou_c_project')

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import pickle

from src.raspberry.model import LSTMPredictor
from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset

# ============================================================
# 直接指定数据路径（与训练时一致）
# ============================================================
DATA_PATH = Path('D:/Desk/desk/beiyou_c_project/data/processed/barcelona_ready_v1')

def get_loader(node_id, split, batch_size=64):
    """加载单个节点的数据"""
    data_dir = DATA_PATH / f"node_{node_id}"
    file_path = data_dir / f"{split}.pkl"
    
    if not file_path.exists():
        return None
    
    dataset = BarcelonaDataset(
        data_path=str(file_path),
        window_size=28,
        predict_size=4,
        sector_feature=True,
        holiday_feature=True,
        weekend_feature=True
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

def evaluate_smape(model, test_loaders, device):
    """与 train_federated_pro_full.py 完全一致的评估函数"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for loader in test_loaders.values():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                all_preds.append(output.cpu().numpy())
                all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    denominator = (np.abs(all_targets) + np.abs(all_preds)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(np.abs(all_targets - all_preds) / denominator) * 100
    
    return smape

# ============================================================
# 主函数
# ============================================================
def main():
    model_path = 'D:/Desk/desk/beiyou_c_project/results/beautified/federated_nodes24_rounds30_mu0.1_model_20260323_063955.pth'
    node_ids = list(range(8001, 8025))
    batch_size = 64
    device = torch.device('cpu')
    
    print("="*60)
    print("评估24节点联邦学习模型")
    print("="*60)
    print(f"数据路径: {DATA_PATH}")
    print(f"节点范围: {node_ids[0]} - {node_ids[-1]} ({len(node_ids)} 个)")
    print()
    
    # 检查数据路径
    if not DATA_PATH.exists():
        print(f"❌ 数据路径不存在: {DATA_PATH}")
        print("请确认训练时使用的数据目录")
        return
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 从 state_dict 推断模型参数
    input_size = state_dict['lstm.weight_ih_l0'].shape[1]
    hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
    num_layers = 1
    for key in state_dict.keys():
        if 'lstm.weight_ih_l' in key:
            num_layers = max(num_layers, int(key.split('_ih_l')[1].split('.')[0]) + 1)
    output_size = state_dict['fc.weight'].shape[0]
    
    print(f"模型参数: input={input_size}, hidden={hidden_size}, layers={num_layers}, output={output_size}")
    
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ 模型加载完成\n")
    
    # 加载测试数据
    print("加载测试数据...")
    test_loaders = {}
    valid_nodes = []
    
    for node_id in node_ids:
        loader = get_loader(node_id, 'test', batch_size)
        if loader is not None and len(loader) > 0:
            test_loaders[node_id] = loader
            valid_nodes.append(node_id)
    
    print(f"有效节点: {len(valid_nodes)}/{len(node_ids)}")
    if valid_nodes:
        print(f"节点列表: {valid_nodes[:5]}...{valid_nodes[-5:]}")
    print()
    
    if len(test_loaders) == 0:
        print("❌ 没有找到有效的测试数据")
        return
    
    # 评估
    print("评估中...")
    smape = evaluate_smape(model, test_loaders, device)
    
    print("\n" + "="*60)
    print(f"24节点联邦模型 sMAPE: {smape:.2f}%")
    print("="*60)
    
    # 保存结果
    import json
    result = {
        'model': model_path,
        'data_path': str(DATA_PATH),
        'nodes': valid_nodes,
        'num_nodes': len(valid_nodes),
        'smape': float(smape),
        'model_params': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size
        }
    }
    
    output_file = Path('results/24node_final_smape.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ 结果保存: {output_file}")

if __name__ == "__main__":
    main()
