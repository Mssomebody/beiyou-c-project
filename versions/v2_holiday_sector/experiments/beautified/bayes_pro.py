#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
贝叶斯优化 - 专业版
- 早停机制
- 学习率调度
- 梯度裁剪
- 验证集调参
- SQLite 存储（可恢复）
- 自动日志保存到 experiments/logs/
- 保存所有 trial 的完整信息（包括 test sMAPE）
"""

import sys
import os
import logging
import json
import yaml
import pickle
import argparse
from datetime import datetime

# ============================================================
# 配置日志（自动保存到 experiments/logs/）
# ============================================================
def setup_logging():
    """配置日志，自动保存到 experiments/logs/"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(base_dir, "experiments", "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bayesian_optimization_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info(f"日志文件: {log_file}")
    return log_file

# 初始化日志
LOG_FILE = setup_logging()

# 重定向 print 到 logging
import builtins
original_print = builtins.print
def print_with_log(*args, **kwargs):
    original_print(*args, **kwargs)
    logging.info(' '.join(str(arg) for arg in args))
builtins.print = print_with_log


# ============================================================
# 添加路径
# ============================================================
_CURRENT_FILE = os.path.abspath(__file__)
_CURRENT_DIR = os.path.dirname(_CURRENT_FILE)
_VERSION_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))

if _VERSION_ROOT not in sys.path:
    sys.path.insert(0, _VERSION_ROOT)

print(f"Version root: {_VERSION_ROOT}")


# ============================================================
# 项目模块导入
# ============================================================
from src.data_loader.barcelona_dataset_v1 import get_node_data_loader
from experiments.beautified.train_single_node import (
    LSTMPredictor, train_epoch, evaluate, evaluate_original_scale
)
from experiments.beautified.step2_adaptive_early_stopping import AdaptiveEarlyStopping

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def get_loaders(node_id=8001, batch_size=64):
    """获取数据加载器"""
    import os
    from torch.utils.data import DataLoader
    from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
    import pickle

    # 数据路径
    base_data_dir = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_v1"
    data_dir = os.path.join(base_data_dir, f"node_{node_id}")

    # 创建 Dataset
    train_dataset = BarcelonaDataset(os.path.join(data_dir, "train.pkl"))
    val_dataset = BarcelonaDataset(os.path.join(data_dir, "val.pkl"))
    test_dataset = BarcelonaDataset(os.path.join(data_dir, "test.pkl"))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 加载 scaler
    scaler_path = os.path.join(data_dir, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return train_loader, val_loader, test_loader, scaler


def main():
    """主函数"""
    print("=" * 70)
    print("贝叶斯优化 - 专业版")
    print("=" * 70)
    print("搜索参数: hidden, layers, lr, dropout, batch_size, optimizer, scheduler, grad_clip")
    print("优化目标: 验证集损失 (MSE)")
    print("早停: 自适应（从 Step 2 读取）")
    print("=" * 70)

    # 读取 Step 2 的早停参数
    config_dir = os.path.join(_VERSION_ROOT, 'configs')
    params_path = os.path.join(config_dir, 'early_stop_params.yaml')

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            early_stop_params = yaml.safe_load(f)
        min_epochs = early_stop_params.get('min_epochs', 10)
        confidence = early_stop_params.get('confidence', 0.95)
        improvement_threshold = early_stop_params.get('improvement_threshold', 0.005)
        print(f"✅ 使用 Step 2 早停参数: min_epochs={min_epochs}, confidence={confidence}")
    else:
        min_epochs = 10
        confidence = 0.95
        improvement_threshold = 0.005
        print(f"⚠️ 未找到 Step 2 参数，使用默认值")

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(_CURRENT_DIR, '..', 'logs', 'bayesian', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n日志目录: {log_dir}")

    # 存储所有 trial 的详细信息
    trials_info = []

    # 使用 SQLite 存储，可中断恢复
    storage_url = f"sqlite:///{os.path.join(log_dir, 'study.db')}"
    study = optuna.create_study(
        direction='minimize',
        study_name=f"fedgreen_bayesian_{timestamp}",
        storage=storage_url,
        load_if_exists=False
    )

    # ============================================================
    # 使用闭包捕获早停参数和 trials_info
    # ============================================================
    def objective(trial):
        """Optuna 目标函数 - 用验证集调参"""
        # 超参数搜索空间
        hidden = trial.suggest_int('hidden', 32, 256, step=32)
        layers = trial.suggest_int('layers', 1, 4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        drop = trial.suggest_float('drop', 0.1, 0.5)
        bs = trial.suggest_categorical('bs', [32, 64, 128])
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
        scheduler_name = trial.suggest_categorical('scheduler', ['plateau', 'cosine'])
        grad_clip = trial.suggest_float('grad_clip', 0.1, 5.0, log=True)

        print(f"\n  Trial {trial.number}: h={hidden}, l={layers}, lr={lr:.2e}, drop={drop:.2f}, "
              f"bs={bs}, opt={optimizer_name}, sched={scheduler_name}, clip={grad_clip:.2f}")

        # 加载数据
        train_loader, val_loader, test_loader, scaler = get_loaders(8001, bs)
        x, _ = next(iter(train_loader))
        input_dim = x.shape[-1]
        device = torch.device('cpu')

        # 创建模型
        model = LSTMPredictor(input_dim, hidden, layers, 4, drop).to(device)

        # 优化器
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        criterion = nn.MSELoss()

        # 使用 Step 2 的自适应早停参数（闭包捕获）
        early_stopping = AdaptiveEarlyStopping(
            min_epochs=min_epochs,
            confidence=confidence,
            improvement_threshold=improvement_threshold
        )

        # 学习率调度器
        if scheduler_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, eta_min=1e-6
            )

        # 训练（用验证集早停）
        max_epochs = 100
        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            # 学习率调度
            if scheduler_name == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # 梯度裁剪
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 自适应早停检查
            stop, reason = early_stopping(val_loss, model)
            if stop:
                print(f"    {reason}")
                break

        # 用验证集的最佳损失作为优化目标
        best_val_loss = early_stopping.best_loss

        # 评估测试集（用于记录，不用于优化）
        eval_results = evaluate_original_scale(model, test_loader, scaler, device, 15)
        test_smape = eval_results['smape']
        
        print(f"    Val Loss: {best_val_loss:.6f}, Test sMAPE: {test_smape:.2f}%")

        # 保存当前 trial 的详细信息
        trial_info = {
            'trial_number': trial.number,
            'params': {
                'hidden': hidden,
                'layers': layers,
                'lr': lr,
                'drop': drop,
                'bs': bs,
                'optimizer': optimizer_name,
                'scheduler': scheduler_name,
                'grad_clip': grad_clip
            },
            'val_loss': best_val_loss,
            'test_smape': test_smape,
            'timestamp': datetime.now().isoformat()
        }
        trials_info.append(trial_info)

        # 返回验证损失（不是测试集！）
        return best_val_loss

    print("\n开始优化...")
    print("预计时间: 20 次 × 5-10 分钟 = 1.5-3 小时")
    print("-" * 70)

    try:
        study.optimize(objective, n_trials=20, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断，保存当前进度...")
        logging.warning("Optimization interrupted by user")
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        raise

    print("\n" + "=" * 70)
    print("优化完成！")
    print("=" * 70)
    
    # 找到最佳 trial 对应的 test sMAPE
    best_test_smape = None
    best_trial_info = None
    
    # 方法1：通过 val_loss 匹配
    for trial_info in trials_info:
        if abs(trial_info['val_loss'] - study.best_value) < 1e-6:
            best_test_smape = trial_info['test_smape']
            best_trial_info = trial_info
            break
    
    # 方法2：如果没找到，按 val_loss 排序取最小的
    if best_test_smape is None and trials_info:
        best_trial_info = min(trials_info, key=lambda x: x['val_loss'])
        best_test_smape = best_trial_info['test_smape']
        print(f"⚠️ 通过 val_loss 匹配失败，使用最小 val_loss 的 trial")
    
    print(f"\n最佳验证损失: {study.best_value:.6f}")
    if best_test_smape is not None:
        print(f"最佳测试sMAPE: {best_test_smape:.2f}%")
    else:
        print(f"最佳测试sMAPE: 未找到（可能 trials_info 为空）")
    
    print(f"最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存结果到 PKL
    results_dir = os.path.join(_VERSION_ROOT, 'results', 'beautified')
    os.makedirs(results_dir, exist_ok=True)
    pkl_path = os.path.join(results_dir, f"bayesian_pro_{timestamp}.pkl")
    
    # 准备保存的数据
    save_data = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_test_smape': best_test_smape,  # 新增
        'trials': trials_info,  # 新增：保存所有 trial 信息
        'storage': storage_url,
        'timestamp': timestamp,
        'early_stop_params': {
            'min_epochs': min_epochs,
            'confidence': confidence,
            'improvement_threshold': improvement_threshold
        },
        'n_trials': len(trials_info),
        'optimization_completed': True
    }
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\n✅ 结果保存: {pkl_path}")

    # 保存为 YAML
    yaml_path = os.path.join(config_dir, 'best_params.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'best_val_loss': study.best_value,
            'best_test_smape': best_test_smape,  # 新增
            'best_params': study.best_params,
            'timestamp': timestamp,
            'early_stop_params': {
                'min_epochs': min_epochs,
                'confidence': confidence,
                'improvement_threshold': improvement_threshold
            }
        }, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ 参数保存: {yaml_path}")

    print("\n后续步骤使用:")
    print(f"  --hidden_dim {study.best_params.get('hidden', 64)}")
    print(f"  --num_layers {study.best_params.get('layers', 2)}")
    print(f"  --lr {study.best_params.get('lr', 0.001):.2e}")
    print(f"  --dropout {study.best_params.get('drop', 0.2)}")
    print(f"  --batch_size {study.best_params.get('bs', 64)}")
    print(f"  --optimizer {study.best_params.get('optimizer', 'Adam')}")
    print(f"  --grad_clip {study.best_params.get('grad_clip', 1.0):.2f}")
    print(f"\n早停参数已从 Step 2 继承:")
    print(f"  --min_epochs {min_epochs}")
    print(f"  --confidence {confidence}")
    print(f"  --improvement_threshold {improvement_threshold}")

    # 打印所有 trials 的汇总信息
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
    logging.info("Bayesian optimization completed successfully")

    return study, trials_info


if __name__ == "__main__":
    study, trials_info = main()
    
    # 找到最佳 trial 的 test sMAPE
    best_trial = min(trials_info, key=lambda x: x['val_loss']) if trials_info else None
    if best_trial:
        print(f"\n{'='*70}")
        print(f"✅ 最终结果:")
        print(f"  最佳验证损失: {best_trial['val_loss']:.6f}")
        print(f"  最佳测试sMAPE: {best_trial['test_smape']:.2f}%")
        print(f"  最佳配置: {best_trial['params']}")
        print(f"{'='*70}")
    else:
        print(f"\n⚠️ 未找到有效的 trial 信息")