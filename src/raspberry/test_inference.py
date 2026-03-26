# -*- coding: utf-8 -*-
"""
树莓派推理测试脚本
验证完整推理流程
生成预测结果图和量化对比日志
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录和raspberry目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)
sys.path.insert(0, script_dir)

from inference import RaspberryInferenceEngine


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("🧪 树莓派推理 - 完整测试")
    print("="*80)
    
    # ==================== 配置 ====================
    model_path = os.path.join(project_root, "checkpoints", "best_fedavg_barcelona.pth")
    data_dir = os.path.join(project_root, "fl_data", "site_0")
    results_dir = os.path.join(project_root, "results")
    logs_dir = os.path.join(project_root, "experiments", "logs")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # ==================== 初始化推理引擎 ====================
    engine = RaspberryInferenceEngine(model_path, data_dir, device='cpu')
    
    # ==================== 功能1: 测试集评估对比 ====================
    print("\n" + "="*80)
    print("📊 功能1: FP32 vs INT8 精度对比")
    print("="*80)
    
    y_pred_fp32, y_true, mse_fp32 = engine.evaluate_test_set(use_int8=False)
    y_pred_int8, _, mse_int8 = engine.evaluate_test_set(use_int8=True)
    
    print(f"\n📈 精度对比总结:")
    print(f"   FP32 MSE: {mse_fp32:.6f}")
    print(f"   INT8 MSE: {mse_int8:.6f}")
    print(f"   精度下降: {((mse_int8 - mse_fp32) / mse_fp32 * 100):.2f}%")
    
    # ==================== 功能2: 单样本推理演示 ====================
    print("\n" + "="*80)
    print("🎯 功能2: 单样本推理演示 (输入24点 → 输出6点)")
    print("="*80)
    
    sample_idx = 0
    X_sample, y_sample_true = engine.data_loader.get_single_sample(sample_idx)
    
    print(f"\n📌 样本 {sample_idx}:")
    print(f"   输入（过去24小时的3个特征）:")
    print(f"   功耗: {X_sample[:, 0].round(2)}")
    print(f"   业务量: {X_sample[:, 1].round(2)}")
    print(f"   温度: {X_sample[:, 2].round(2)}")
    
    y_pred_fp32_single, time_fp32 = engine.predict_single(X_sample, use_int8=False)
    y_pred_int8_single, time_int8 = engine.predict_single(X_sample, use_int8=True)
    
    y_true_single = engine.data_loader.denormalize_power(y_sample_true)
    
    print(f"\n🎯 预测结果（未来6小时功耗）:")
    print(f"   真实值（第1小时）: {y_true_single:.4f} kW")
    print(f"   FP32预测: {y_pred_fp32_single.round(4)} kW")
    print(f"   INT8预测: {y_pred_int8_single.round(4)} kW")
    print(f"   FP32推理时间: {time_fp32:.4f} ms")
    print(f"   INT8推理时间: {time_int8:.4f} ms")
    print(f"   FP32误差（第1小时）: {abs(y_pred_fp32_single[0] - y_true_single):.6f} kW")
    print(f"   INT8误差（第1小时）: {abs(y_pred_int8_single[0] - y_true_single):.6f} kW")
    
    # ==================== 功能3: 生成预测结果图 ====================
    print("\n" + "="*80)
    print("📊 功能3: 生成预测结果图")
    print("="*80)
    
    # 取前100个样本的第1小时预测进行可视化
    num_plot_samples = min(100, len(y_true))
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#f5f5f5')
    
    # 统一配色
    COLOR_TRUE = '#2563eb'  # 蓝 - 真实值
    COLOR_PRED = '#dc2626'  # 红 - 预测值
    
    # 子图1: FP32预测 vs 真实值
    ax1.set_facecolor('#ffffff')
    ax1.plot(range(num_plot_samples), y_true[:num_plot_samples], 
             color=COLOR_TRUE, linewidth=2.0, label='True Value')
    ax1.plot(range(num_plot_samples), y_pred_fp32[:num_plot_samples, 0], 
             color=COLOR_PRED, linewidth=2.0, linestyle='--', label='FP32 Prediction')
    ax1.set_title('FP32: Prediction vs True Value (First 100 Samples)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Power (kW)', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', color='#dddddd')
    ax1.legend(fontsize=11, framealpha=0.9)
    
    # 子图2: INT8预测 vs 真实值
    ax2.set_facecolor('#ffffff')
    ax2.plot(range(num_plot_samples), y_true[:num_plot_samples], 
             color=COLOR_TRUE, linewidth=2.0, label='True Value')
    ax2.plot(range(num_plot_samples), y_pred_int8[:num_plot_samples, 0], 
             color=COLOR_PRED, linewidth=2.0, linestyle='--', label='INT8 Prediction')
    ax2.set_title('INT8: Prediction vs True Value (First 100 Samples)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Sample Index', fontsize=11)
    ax2.set_ylabel('Power (kW)', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', color='#dddddd')
    ax2.legend(fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存高清图片
    plot_path = os.path.join(results_dir, f"raspberry_prediction_{timestamp}.png")
    plt.savefig(plot_path,
                dpi=300,
                bbox_inches='tight',
                facecolor='#f5f5f5')
    plt.close()
    
    print(f"\n✅ 预测结果图已保存: {plot_path}")
    
    # ==================== 功能4: 保存预测结果 ====================
    print("\n" + "="*80)
    print("💾 功能4: 保存预测结果")
    print("="*80)
    
    predictions_path = os.path.join(results_dir, f"raspberry_predictions_{timestamp}.npy")
    np.save(predictions_path, {
        'y_true': y_true,
        'y_pred_fp32': y_pred_fp32,
        'y_pred_int8': y_pred_int8,
        'mse_fp32': mse_fp32,
        'mse_int8': mse_int8
    })
    
    print(f"\n✅ 预测结果已保存: {predictions_path}")
    
    # ==================== 功能5: 保存量化对比日志 ====================
    print("\n" + "="*80)
    print("📝 功能5: 保存量化对比日志")
    print("="*80)
    
    log_path = os.path.join(logs_dir, f"raspberry_quantization_{timestamp}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("树莓派推理 - 量化对比日志\n")
        f.write("="*80 + "\n\n")
        f.write(f"实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备: CPU\n")
        f.write(f"数据目录: {data_dir}\n\n")
        
        f.write("="*80 + "\n")
        f.write("精度对比\n")
        f.write("="*80 + "\n\n")
        f.write(f"FP32 MSE: {mse_fp32:.6f}\n")
        f.write(f"INT8 MSE: {mse_int8:.6f}\n")
        f.write(f"精度下降: {((mse_int8 - mse_fp32) / mse_fp32 * 100):.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("生成文件\n")
        f.write("="*80 + "\n\n")
        f.write(f"预测结果图: {plot_path}\n")
        f.write(f"预测结果数据: {predictions_path}\n")
        f.write(f"本日志: {log_path}\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n✅ 量化对比日志已保存: {log_path}")
    
    # ==================== 总结 ====================
    print("\n" + "="*80)
    print("🏆 测试完成总结")
    print("="*80)
    
    print(f"\n📦 模型信息:")
    print(f"   - 输入: 24个时间点 × 3个特征")
    print(f"   - 输出: 6个时间点（未来6小时功耗）")
    print(f"   - 隐藏层: 64")
    print(f"   - 层数: 2")
    
    print(f"\n⚡ 性能总结:")
    print(f"   - FP32: MSE={mse_fp32:.6f}")
    print(f"   - INT8: MSE={mse_int8:.6f}")
    print(f"   - 精度下降: {((mse_int8 - mse_fp32) / mse_fp32 * 100):.2f}%")
    
    print(f"\n📁 生成文件:")
    print(f"   - 预测结果图: {plot_path}")
    print(f"   - 预测结果数据: {predictions_path}")
    print(f"   - 量化对比日志: {log_path}")
    
    print(f"\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
