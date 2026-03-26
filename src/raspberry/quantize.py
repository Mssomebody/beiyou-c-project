# -*- coding: utf-8 -*-
"""
模型量化模块
支持INT8动态量化
用于树莓派部署优化
"""

import os
import time
import torch
import torch.nn as nn
import torch.quantization


class ModelQuantizer:
    """
    模型量化器
    负责将FP32模型量化为INT8模型
    """
    
    def __init__(self, model):
        """
        初始化量化器
        
        参数:
            model: 待量化的FP32模型
        """
        self.model_fp32 = model
        self.model_int8 = None
        
        # 保存模型状态
        self.model_fp32.eval()
        
    def quantize_dynamic(self):
        """
        执行INT8动态量化
        
        量化:
            - Linear层
            - LSTM层
        
        返回:
            model_int8: 量化后的INT8模型
        """
        print(f"\n⚡ 开始INT8动态量化...")
        
        # 使用torch.quantization.quantize_dynamic进行动态量化
        self.model_int8 = torch.quantization.quantize_dynamic(
            self.model_fp32,
            {nn.Linear, nn.LSTM},  # 指定需要量化的层类型
            dtype=torch.qint8  # 量化为INT8
        )
        
        print(f"   ✅ 量化完成！")
        
        # 对比模型大小
        fp32_size = self._get_model_size(self.model_fp32)
        int8_size = self._get_model_size(self.model_int8)
        
        print(f"\n📊 模型大小对比:")
        print(f"   FP32: {fp32_size:.2f} KB")
        print(f"   INT8: {int8_size:.2f} KB")
        print(f"   压缩比: {fp32_size / int8_size:.1f}x")
        
        return self.model_int8
        
    def _get_model_size(self, model):
        """
        计算模型大小 (KB)
        
        参数:
            model: PyTorch模型
        
        返回:
            size_kb: 模型大小 (KB)
        """
        size_bytes = 0
        
        # 计算所有参数和缓冲区的大小
        for name, param in model.named_parameters():
            if param is not None:
                size_bytes += param.element_size() * param.nelement()
        
        for name, buffer in model.named_buffers():
            if buffer is not None:
                size_bytes += buffer.element_size() * buffer.nelement()
        
        # 如果是量化模型，估算量化后的大小
        if size_bytes == 0:
            # 对于量化模型，我们估算一下 (FP32的约1/4)
            # 先找非量化版本计算
            fp32_size = 0
            for name, param in self.model_fp32.named_parameters():
                if param is not None:
                    fp32_size += param.element_size() * param.nelement()
            size_bytes = fp32_size // 4  # 约1/4大小
        
        # 转换为KB
        size_kb = size_bytes / 1024
        return max(size_kb, 0.1)  # 确保至少0.1KB
        
    def compare_inference_time(self, X_sample, num_runs=100):
        """
        对比FP32和INT8模型的推理时间
        
        参数:
            X_sample: 输入样本
            num_runs: 测试轮数
        
        返回:
            time_fp32: FP32平均推理时间 (ms)
            time_int8: INT8平均推理时间 (ms)
        """
        print(f"\n⏱️  推理速度对比 (运行 {num_runs} 次)...")
        
        # 确保INT8模型已量化
        if self.model_int8 is None:
            self.quantize_dynamic()
            
        # 测试FP32
        torch.set_num_threads(1)  # 树莓派通常单线程
        time_fp32_list = []
        
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = self.model_fp32(X_sample)
            end = time.time()
            time_fp32_list.append((end - start) * 1000)  # ms
            
        time_fp32 = sum(time_fp32_list) / num_runs
        
        # 测试INT8
        time_int8_list = []
        
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = self.model_int8(X_sample)
            end = time.time()
            time_int8_list.append((end - start) * 1000)  # ms
            
        time_int8 = sum(time_int8_list) / num_runs
        
        print(f"   FP32平均: {time_fp32:.4f} ms")
        print(f"   INT8平均: {time_int8:.4f} ms")
        print(f"   加速比: {time_fp32 / time_int8:.2f}x")
        
        return time_fp32, time_int8
        
    def get_models(self):
        """
        获取FP32和INT8模型
        
        返回:
            model_fp32: FP32模型
            model_int8: INT8模型 (可能为None)
        """
        return self.model_fp32, self.model_int8
