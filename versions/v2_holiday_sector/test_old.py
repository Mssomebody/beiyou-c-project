#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
贝叶斯优化 - 聚焦版（基于最佳参数缩小范围）
- 基于 trial 18 的最佳参数：hidden=128, layers=3, lr=0.0036, drop=0.15, bs=64
- 缩小搜索空间，进行精细调优
- 保存所有 trial 的完整信息
"""

import sys
import os
import logging
import yaml
import pickle
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 直接导入需要的模块
from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import (
    LSTMPredictor, train_epoch, evaluate, evaluate_original_scale
)
from experiments.beautified.step2_adaptive_early_stopping import AdaptiveEarlyStopping

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 配置日志
def setup_logging():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(base_dir, "experiments", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bayesian_focused_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return log_file

LOG_FILE = setup_logging()

# 重定向 print
import builtins
original_print = builtins.print
def print_with_log(*args, **kwargs):
    original_print(*args, **kwargs)
    logging.info(' '.join(str(arg) for arg in args))
builtins.print = print_with_log

_VERSION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_loaders(node_id=8001, batch_size=64):
    """获取数据加载器"""
    import pickle

    base_data_dir = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"
    base_data_dir = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"
    data_dir = os.path.join(base_data_dir, f"node_{node_id}")

    train_dataset = BarcelonaDataset(os.path.join(data_dir, "train.pkl"))
    val_dataset = BarcelonaDataset(os.path.join(data_dir, "val.pkl"))
    test_dataset = BarcelonaDataset(os.path.join(data_dir, "test.pkl"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    with open(os.path.join(data_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    
    return train_loader, val_loader, test_loader, scaler

def main():
    print("=" * 70)
    print("贝叶斯优化 - 聚焦版（基于最佳参数缩小范围）")
    print("=" * 70)
    print("基准最佳参数: hidden=128, layers=3, lr=0.0036, drop=0.15, bs=64")
    print("优化目标: 验证集损失 (MSE)")
    print("=" * 70)
    
    # 早停参数
    config_dir = os.path.join(_VERSION_ROOT, 'configs')
    params_path = os.path.join(config_dir, 'early_stop_params.yaml')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            early_stop_params = yaml.safe_load(f)
        min_epochs = early_stop_params.get('min_epochs', 10)
        confidence = early_stop_params.get('confidence', 0.95)
        improvement_threshold = early_stop_params.get('improvement_threshold', 0.005)
    else:
        min_epochs, confidence, improvement_threshold = 10, 0.95, 0.005
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', 'bayesian_focused', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    storage_url = f"sqlite:///{os.path.join(log_dir, 'study.db')}"
    study = optuna.create_study(
        direction='minimize',
        study_name=f"fedgreen_focused_{timestamp}",
        storage=storage_url,
        load_if_exists=False
    )
    
    trials_info = []
    
    def objective(trial):
        # 缩小搜索范围（基于最佳参数 ± 范围）
        hidden = trial.suggest_int('hidden', 96, 160, step=32)
        layers = trial.suggest_int('layers', 2, 4)
        lr = trial.suggest_float('lr', 1e-3, 6e-3, log=True)
        drop = trial.suggest_float('drop', 0.1, 0.25)
        bs = trial.suggest_categorical('bs', [64])
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam'])
        scheduler_name = trial.suggest_categorical('scheduler', ['cosine'])
        grad_clip = trial.suggest_float('grad_clip', 0.2, 0.5, log=True)
        
        print(f"\n  Trial {trial.number}: h={hidden}, l={layers}, lr={lr:.2e}, drop={drop:.2f}, "
              f"bs={bs}, opt={optimizer_name}, sched={scheduler_name}, clip={grad_clip:.2f}")
        
        train_loader, val_loader, test_loader, scaler = get_loaders(8001, bs)
        x, _ = next(iter(train_loader))
        input_dim = x.shape[-1]
        device = torch.device('cpu')
        
        model = LSTMPredictor(input_dim, hidden, layers, 4, drop).to(device)
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        criterion = nn.MSELoss()
        early_stopping = AdaptiveEarlyStopping(
            min_epochs=min_epochs, confidence=confidence, 
            improvement_threshold=improvement_threshold
        )
        
        if scheduler_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, eta_min=1e-6
            )
        
        max_epochs = 100
        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            
            if scheduler_name == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            stop, reason = early_stopping(val_loss, model)
            if stop:
                print(f"    {reason}")
                break
        
        best_val_loss = early_stopping.best_loss
        eval_results = evaluate_original_scale(model, test_loader, scaler, device, 15)
        test_smape = eval_results['smape']
        
        print(f"    Val Loss: {best_val_loss:.6f}, Test sMAPE: {test_smape:.2f}%")
        
        trials_info.append({
            'trial_number': trial.number,
            'params': {
                'hidden': hidden, 'layers': layers, 'lr': lr, 'drop': drop,
                'bs': bs, 'optimizer': optimizer_name, 'scheduler': scheduler_name,
                'grad_clip': grad_clip
            },
            'val_loss': best_val_loss,
            'test_smape': test_smape
        })
        
        return best_val_loss
    
    print("\n开始聚焦优化...")
    print("预计时间: 15 次 × 5-10 分钟 = 1-2 小时")
    print("-" * 70)
    
    try:
        study.optimize(objective, n_trials=1, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断，保存当前进度...")
    
    print("\n" + "=" * 70)
    print("聚焦优化完成！")
    print("=" * 70)
    
    # 找到最佳 trial
    best_test_smape = None
    for trial_info in trials_info:
        if abs(trial_info['val_loss'] - study.best_value) < 1e-6:
            best_test_smape = trial_info['test_smape']
            break
    
    if best_test_smape is None and trials_info:
        best_trial_info = min(trials_info, key=lambda x: x['val_loss'])
        best_test_smape = best_trial_info['test_smape']
    
    print(f"\n最佳验证损失: {study.best_value:.6f}")
    print(f"最佳测试sMAPE: {best_test_smape:.2f}%")
    print(f"最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    results_dir = os.path.join(_VERSION_ROOT, 'results', 'beautified')
    os.makedirs(results_dir, exist_ok=True)
    pkl_path = os.path.join(results_dir, f"bayesian_focused_{timestamp}.pkl")
    
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_test_smape': best_test_smape,
            'trials': trials_info,
            'storage': storage_url,
            'timestamp': timestamp,
            'early_stop_params': {'min_epochs': min_epochs, 'confidence': confidence},
            'n_trials': len(trials_info),
            'optimization_completed': True
        }, f)
    print(f"\n✅ 结果保存: {pkl_path}")
    
    # 保存 YAML
    yaml_path = os.path.join(config_dir, 'best_params_focused.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'best_val_loss': study.best_value,
            'best_test_smape': best_test_smape,
            'best_params': study.best_params,
            'timestamp': timestamp,
            'early_stop_params': {'min_epochs': min_epochs, 'confidence': confidence}
        }, f, default_flow_style=False)
    print(f"✅ 参数保存: {yaml_path}")
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("所有 Trials 汇总:")
    print("=" * 70)
    print(f"{'Trial':<6} {'Val Loss':<12} {'Test sMAPE':<12} {'Hidden':<8} {'Layers':<6} {'LR':<10} {'Drop':<6}")
    print("-" * 70)
    for trial_info in sorted(trials_info, key=lambda x: x['val_loss']):
        params = trial_info['params']
        is_best = "★" if trial_info['val_loss'] == study.best_value else " "
        print(f"{is_best}{trial_info['trial_number']:<5} "
              f"{trial_info['val_loss']:.6f}  "
              f"{trial_info['test_smape']:>6.2f}%     "
              f"{params['hidden']:<8} "
              f"{params['layers']:<6} "
              f"{params['lr']:.2e} "
              f"{params['drop']:.3f}")
    print("=" * 70)
    
    return study, trials_info

if __name__ == "__main__":
    study, trials_info = main()
    if trials_info:
        best = min(trials_info, key=lambda x: x['val_loss'])
        print(f"\n✅ 最终最佳测试 sMAPE: {best['test_smape']:.2f}%")
