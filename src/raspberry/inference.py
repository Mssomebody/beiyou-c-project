# -*- coding: utf-8 -*-
"""
推理主逻辑模块
树莓派推理引擎核心
"""

import os
import sys
import time
import numpy as np
import torch

# 添加项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from model import create_model
from data_loader import RaspberryDataLoader
from quantize import ModelQuantizer


class RaspberryInferenceEngine:
    """
    树莓派推理引擎
    整合模型加载、数据预处理、量化、推理等功能
    """
    
    def __init__(self, model_path, data_dir, device='cpu'):
        """
        初始化推理引擎
        
        参数:
            model_path: 预训练模型路径
            data_dir: 数据目录路径
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.data_dir = data_dir
        
        print(f"\n{'='*80}")
        print("🚀 初始化树莓派推理引擎")
        print(f"{'='*80}")
        print(f"   设备: {self.device}")
        print(f"   模型路径: {model_path}")
        print(f"   数据目录: {self.data_dir}")
        
        # 1. 初始化数据加载器
        print(f"\n📦 初始化数据加载器...")
        self.data_loader = RaspberryDataLoader(data_dir)
        
        # 2. 创建并加载模型
        print(f"\n🤖 初始化模型...")
        
        # 先尝试创建和checkpoint匹配的模型 (input_size=1, output_size=1)
        self.model_fp32 = create_model(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            output_size=6
        )
        
        # 尝试加载预训练模型
        model_loaded = False
        if os.path.exists(model_path):
            print(f"🔄 尝试加载预训练模型...")
            try:
                state_dict = torch.load(model_path, map_location=device)
                # 检查模型结构是否匹配
                self.model_fp32.load_state_dict(state_dict)
                print(f"   ✅ 模型加载成功！")
                model_loaded = True
            except Exception as e:
                print(f"   ⚠️  模型结构不匹配，使用随机权重初始化（用于演示）")
                print(f"   错误: {e}")
        
        if not model_loaded:
            print(f"⚠️  使用随机权重初始化模型（用于演示）")
            
        self.model_fp32.to(device)
        self.model_fp32.eval()
        
        # 3. 初始化量化器
        print(f"\n⚡ 初始化量化器...")
        self.quantizer = ModelQuantizer(self.model_fp32)
        self.model_int8 = self.quantizer.quantize_dynamic()
        
        # 加载测试数据
        print(f"\n📊 加载测试数据...")
        self.X_test, self.y_test = self.data_loader.load_test_data()
        
    def predict_single(self, X_input, use_int8=False):
        """
        单样本推理: 输入24点 → 输出6点预测
        
        参数:
            X_input: 输入数据, 形状 (24, 3)
            use_int8: 是否使用INT8量化模型
        
        返回:
            y_pred: 预测值, 形状 (6,) (原始尺度, kW)
            inference_time: 推理时间 (ms)
        """
        # 选择模型
        model = self.model_int8 if use_int8 else self.model_fp32
        
        # 归一化
        X_norm = self.data_loader.normalize(X_input)
        
        # 转换为张量并添加batch维度
        X_tensor = torch.FloatTensor(X_norm).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            start_time = time.time()
            y_pred_norm = model(X_tensor)
            inference_time = (time.time() - start_time) * 1000
            
        # 反归一化 (仅功耗)
        y_pred = self.data_loader.denormalize_power(y_pred_norm.cpu().numpy()[0])
        
        return y_pred, inference_time
        
    def predict_batch(self, X_batch, use_int8=False):
        """
        批量推理
        
        参数:
            X_batch: 批量输入, 形状 (batch_size, 24, 3)
            use_int8: 是否使用INT8量化模型
        
        返回:
            y_pred: 批量预测值, 形状 (batch_size, 6)
            inference_time: 总推理时间 (ms)
        """
        model = self.model_int8 if use_int8 else self.model_fp32
        
        # 归一化
        X_norm = self.data_loader.normalize(X_batch)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        # 推理
        with torch.no_grad():
            start_time = time.time()
            y_pred_norm = model(X_tensor)
            inference_time = (time.time() - start_time) * 1000
            
        # 反归一化
        y_pred = self.data_loader.denormalize_power(y_pred_norm.cpu().numpy())
        
        return y_pred, inference_time
        
    def evaluate_test_set(self, use_int8=False):
        """
        在测试集上评估模型
        
        参数:
            use_int8: 是否使用INT8量化模型
        
        返回:
            y_pred: 预测值, 形状 (n_samples, 6)
            y_true: 真实值, 形状 (n_samples,)
            mse: 均方误差
        """
        model_type = "INT8" if use_int8 else "FP32"
        print(f"\n🔍 测试集评估 ({model_type})...")
        
        # 批量推理
        y_pred, inference_time = self.predict_batch(self.X_test, use_int8=use_int8)
        
        # 反归一化真实值
        y_true = self.data_loader.denormalize_power(self.y_test)
        
        # 计算MSE (用第一个预测点和真实值比较)
        # 因为y_test是单步预测，我们取预测的第一个点
        mse = np.mean((y_pred[:, 0] - y_true) ** 2)
        
        print(f"   测试样本数: {len(self.X_test)}")
        print(f"   总推理时间: {inference_time:.2f} ms")
        print(f"   单样本平均: {inference_time / len(self.X_test):.4f} ms")
        print(f"   测试MSE: {mse:.6f}")
        
        return y_pred, y_true, mse
        
    def get_models(self):
        """
        获取FP32和INT8模型
        
        返回:
            model_fp32: FP32模型
            model_int8: INT8模型
        """
        return self.model_fp32, self.model_int8
