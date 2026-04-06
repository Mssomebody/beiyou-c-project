"""
Step 2: 真正专业的自适应早停（无需手动设置 patience）
基于统计检验自动判断验证损失是否继续改善
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import sys
import argparse
import yaml
from datetime import datetime
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.barcelona_dataset import get_node_data_loader
from experiments.beautified.train_single_node import (
    LSTMPredictor, train_epoch, evaluate, evaluate_original_scale,
    plot_loss_curve, COLORS
)


class AdaptiveEarlyStopping:
    """
    真正专业的自适应早停
    - 无需手动设置 patience
    - 基于统计检验判断是否还有改善空间
    - 基于运行平均值判断是否开始恶化
    """
    
    def __init__(self, min_epochs=10, confidence=0.95, improvement_threshold=0.005):
        """
        Args:
            min_epochs: 最少训练轮数（避免过早停止）
            confidence: 置信度（用于统计检验）
            improvement_threshold: 最小改善阈值（相对值）
        """
        self.min_epochs = min_epochs
        self.confidence = confidence
        self.improvement_threshold = improvement_threshold
        self.best_loss = None
        self.best_epoch = 0
        self.best_weights = None
        self.loss_history = []
        self.early_stop = False
        self.stop_reason = None
    
    def _statistical_test(self):
        """
        统计检验：比较最近10轮与之前10轮的损失是否有显著差异
        使用 t-test 判断近期损失是否显著高于前期
        """
        if len(self.loss_history) < 20:
            return False
        
        recent = self.loss_history[-10:]
        previous = self.loss_history[-20:-10]
        
        # t-test: 检验近期损失是否显著高于前期
        t_stat, p_value = stats.ttest_ind(recent, previous, alternative='greater')
        
        # p > (1-confidence) 表示没有显著改善，应该停止
        return p_value > (1 - self.confidence)
    
    def _rolling_average_test(self, window=5):
        """
        运行平均值检验：检查是否开始恶化
        """
        if len(self.loss_history) < window * 2:
            return False
        
        recent_avg = np.mean(self.loss_history[-window:])
        earlier_avg = np.mean(self.loss_history[-window*2:-window])
        
        # 如果近期平均值高于前期平均值（恶化），返回 True
        return recent_avg > earlier_avg
    
    def _improvement_test(self):
        """
        改善检验：检查最近是否有实质性改善
        """
        if self.best_loss is None:
            return True
        
        current_loss = self.loss_history[-1]
        relative_improvement = (self.best_loss - current_loss) / self.best_loss
        
        # 如果有实质性改善，继续训练
        return relative_improvement > self.improvement_threshold
    
    def __call__(self, val_loss, model):
        """
        判断是否应该停止
        
        Returns:
            stop: 是否停止
            reason: 停止原因
        """
        self.loss_history.append(val_loss)
        
        # 更新最佳损失
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = len(self.loss_history)
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        
        # 最少训练轮数检查
        if len(self.loss_history) < self.min_epochs:
            return False, None
        
        # 检查是否有实质性改善
        if self._improvement_test():
            return False, None
        
        # 检查是否开始恶化（运行平均值）
        if self._rolling_average_test():
            self.early_stop = True
            self.stop_reason = f"Loss degradation detected (rolling average)"
            return True, self.stop_reason
        
        # 统计检验（没有显著改善）
        if self._statistical_test():
            self.early_stop = True
            self.stop_reason = f"No significant improvement (p > {1-self.confidence:.2f})"
            return True, self.stop_reason
        
        return False, None
    
    def get_best_model(self, model):
        """恢复最佳权重"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        return model


class LearningRateScheduler:
    """
    专业学习率调度器
    """
    def __init__(self, optimizer, mode='plateau', factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        
        if mode == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=5, 
                min_lr=min_lr
            )
        elif mode == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, eta_min=min_lr
            )
        else:
            self.scheduler = None
    
    def step(self, val_loss=None):
        if self.scheduler is None:
            return
        
        if self.mode == 'plateau' and val_loss is not None:
            self.scheduler.step(val_loss)
        elif self.mode == 'cosine':
            self.scheduler.step()
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def train_with_adaptive_early_stopping(model, train_loader, val_loader, criterion, optimizer,
                                        device, max_epochs=100, min_epochs=10, 
                                        confidence=0.95, improvement_threshold=0.005):
    """
    带自适应早停的专业训练
    - 无需手动设置 patience
    - 自动判断停止时机
    """
    early_stopping = AdaptiveEarlyStopping(
        min_epochs=min_epochs,
        confidence=confidence,
        improvement_threshold=improvement_threshold
    )
    lr_scheduler = LearningRateScheduler(optimizer, mode='plateau')
    
    train_losses = []
    val_losses = []
    lr_history = []
    
    print(f"\n开始自适应训练 (最大轮数={max_epochs}, 最少轮数={min_epochs})")
    print(f"停止条件: 统计检验 + 运行平均值检测")
    print("-" * 60)
    
    for epoch in range(1, max_epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 学习率调度
        lr_scheduler.step(val_loss)
        current_lr = lr_scheduler.get_lr()
        lr_history.append(current_lr)
        
        # 自适应早停检查
        stop, reason = early_stopping(val_loss, model)
        
        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{max_epochs} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | "
                  f"Best Loss: {early_stopping.best_loss:.6f} (epoch {early_stopping.best_epoch})")
        
        if stop:
            print(f"\n✅ 自适应早停触发于 epoch {epoch}")
            print(f"   停止原因: {reason}")
            print(f"   最佳验证损失: {early_stopping.best_loss:.6f} (epoch {early_stopping.best_epoch})")
            # 恢复最佳权重
            model = early_stopping.get_best_model(model)
            break
    
    # 如果没有触发早停，达到最大轮数
    if not stop:
        print(f"\n⚠️ 达到最大轮数 {max_epochs}，未触发早停")
        print(f"   最佳验证损失: {early_stopping.best_loss:.6f} (epoch {early_stopping.best_epoch})")
        model = early_stopping.get_best_model(model)
    
    return model, train_losses, val_losses, lr_history, early_stopping.best_loss, early_stopping.best_epoch


def main(args):
    print("=" * 60)
    print("Step 2: 自适应早停（无需手动设置 patience）")
    print(f"节点: {args.node}")
    print(f"阈值百分位数: {args.percentile}%")
    print(f"最少训练轮数: {args.min_epochs}")
    print(f"置信度: {args.confidence}")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # 加载数据
    print("\n加载数据...")
    train_loader, scaler_path, train_dataset = get_node_data_loader(
        node_id=args.node, split='train', batch_size=args.batch_size, shuffle=True,
        sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    val_loader, _, _ = get_node_data_loader(
        node_id=args.node, split='val', batch_size=args.batch_size, shuffle=False,
        sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    test_loader, _, _ = get_node_data_loader(
        node_id=args.node, split='test', batch_size=args.batch_size, shuffle=False,
        sector_feature=True, holiday_feature=True, weekend_feature=True
    )
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 获取输入维度
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]
    
    print(f"输入维度: {input_dim}")
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")
    
    # 创建模型
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=4,
        dropout=args.dropout
    ).to(device)
    
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练
    model, train_losses, val_losses, lr_history, best_val_loss, best_epoch = train_with_adaptive_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, device,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        confidence=args.confidence,
        improvement_threshold=args.improvement_threshold
    )
    
    # 评估
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    test_loss = evaluate(model, test_loader, criterion, device)
    eval_results = evaluate_original_scale(
        model, test_loader, scaler, device, args.percentile
    )
    
    print(f"\n归一化尺度:")
    print(f"  测试损失 (MSE): {test_loss:.6f}")
    
    print(f"\n原始尺度 (kWh):")
    print(f"  RMSE: {eval_results['rmse']:.2f} kWh")
    print(f"  MAE:  {eval_results['mae']:.2f} kWh")
    print(f"  sMAPE: {eval_results['smape']:.2f}%")
    print(f"  MAPE: {eval_results['mape_filtered']:.2f}% (过滤 < {eval_results['mape_threshold']:.0f} kWh)")
    
    # 对比基线
    print(f"\n{'='*60}")
    print("与基线对比")
    print(f"{'='*60}")
    print(f"基线 sMAPE (Step 1): 69.93%")
    print(f"当前 sMAPE: {eval_results['smape']:.2f}%")
    improvement = 69.93 - eval_results['smape']
    print(f"提升: {improvement:+.2f}%")

    early_stop_params = {
        'min_epochs': args.min_epochs,
        'confidence': args.confidence,
        'improvement_threshold': args.improvement_threshold
    }
    
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              'configs')
    os.makedirs(config_dir, exist_ok=True)
    params_path = os.path.join(config_dir, 'early_stop_params.yaml')
    with open(params_path, 'w') as f:
        yaml.dump(early_stop_params, f)
    print(f"✅ 早停参数保存: {params_path}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "beautified")
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'step': 'adaptive_early_stopping',
        'node': args.node,
        'percentile': args.percentile,
        'min_epochs': args.min_epochs,
        'confidence': args.confidence,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history,
        'test_loss': test_loss,
        'rmse': eval_results['rmse'],
        'mae': eval_results['mae'],
        'smape': eval_results['smape'],
        'mape_filtered': eval_results['mape_filtered']
    }
    
    results_path = os.path.join(save_dir, f"adaptive_early_stopping_node{args.node}_results_{timestamp}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # 绘制损失曲线
    loss_plot_path = os.path.join(save_dir, f"adaptive_early_stopping_node{args.node}_loss_{timestamp}.png")
    plot_loss_curve(train_losses, val_losses, 
                   f"Adaptive Early Stopping - Node {args.node}\nBest: {best_val_loss:.6f} @ epoch {best_epoch}", 
                   loss_plot_path)
    
    print(f"\n✅ 结果保存: {results_path}")
    print(f"✅ 损失曲线: {loss_plot_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive Early Stopping (No patience parameter)')
    parser.add_argument('--node', type=int, default=8001)
    parser.add_argument('--percentile', type=int, default=15, help='From Step 1')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--min_epochs', type=int, default=10, help='Minimum epochs before stopping')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence for statistical test')
    parser.add_argument('--improvement_threshold', type=float, default=0.005, 
                        help='Minimum relative improvement to continue')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("自适应早停训练")
    print("=" * 60)
    print(f"特点: 无需手动设置 patience")
    print(f"停止条件: 统计检验 + 运行平均值检测")
    print(f"最少训练: {args.min_epochs} 轮")
    print(f"置信度: {args.confidence}")
    print("=" * 60)
    
    results = main(args)
    
    print(f"\n✅ Step 2 完成！")
    print(f"最佳轮数: {results['best_epoch']}")
    print(f"最佳验证损失: {results['best_val_loss']:.6f}")
    print(f"sMAPE: {results['smape']:.2f}%")