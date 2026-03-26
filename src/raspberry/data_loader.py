# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
用于树莓派推理的数据处理
"""

import os
import numpy as np


class RaspberryDataLoader:
    """
    树莓派数据加载器
    负责加载和预处理联邦学习数据
    """
    
    def __init__(self, data_dir):
        """
        初始化数据加载器
        
        参数:
            data_dir: 数据目录路径 (例如: 'fl_data/site_0/')
        """
        self.data_dir = data_dir
        
        # 加载归一化参数
        self.mean = None
        self.std = None
        self._load_normalization_params()
        
        # 存储加载的数据
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def _load_normalization_params(self):
        """
        加载归一化的均值和标准差
        从 mean.npy 和 std.npy 文件读取
        """
        mean_path = os.path.join(self.data_dir, 'mean.npy')
        std_path = os.path.join(self.data_dir, 'std.npy')
        
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            print(f"✅ 归一化参数已加载")
            print(f"   均值: {self.mean.round(4)}")
            print(f"   标准差: {self.std.round(4)}")
        else:
            raise FileNotFoundError(f"未找到归一化参数文件: {mean_path} 或 {std_path}")
        
    def load_test_data(self):
        """
        加载测试集数据
        
        返回:
            X_test: 测试集输入, 形状 (样本数, 24, 3)
            y_test: 测试集标签, 形状 (样本数,)
        """
        X_test_path = os.path.join(self.data_dir, 'X_test.npy')
        y_test_path = os.path.join(self.data_dir, 'y_test.npy')
        
        if os.path.exists(X_test_path) and os.path.exists(y_test_path):
            self.X_test = np.load(X_test_path)
            self.y_test = np.load(y_test_path)
            print(f"✅ 测试集已加载")
            print(f"   X_test形状: {self.X_test.shape}")
            print(f"   y_test形状: {self.y_test.shape}")
            return self.X_test, self.y_test
        else:
            raise FileNotFoundError(f"未找到测试集文件: {X_test_path} 或 {y_test_path}")
        
    def load_train_data(self):
        """
        加载训练集数据 (可选，用于参考)
        
        返回:
            X_train: 训练集输入, 形状 (样本数, 24, 3)
            y_train: 训练集标签, 形状 (样本数,)
        """
        X_train_path = os.path.join(self.data_dir, 'X_train.npy')
        y_train_path = os.path.join(self.data_dir, 'y_train.npy')
        
        if os.path.exists(X_train_path) and os.path.exists(y_train_path):
            self.X_train = np.load(X_train_path)
            self.y_train = np.load(y_train_path)
            print(f"✅ 训练集已加载")
            print(f"   X_train形状: {self.X_train.shape}")
            print(f"   y_train形状: {self.y_train.shape}")
            return self.X_train, self.y_train
        else:
            print(f"⚠️  未找到训练集文件 (可选)")
            return None, None
        
    def normalize(self, X):
        """
        归一化输入数据
        
        参数:
            X: 原始输入数据, 形状 (..., 3)
        
        返回:
            X_normalized: 归一化后的数据
        """
        X_normalized = (X - self.mean) / self.std
        return X_normalized
        
    def denormalize_power(self, y_normalized):
        """
        反归一化功耗预测值 (仅功耗特征)
        
        参数:
            y_normalized: 归一化后的预测值
        
        返回:
            y_original: 原始尺度的功耗值 (kW)
        """
        # 功耗是第一个特征 (index 0)
        y_original = y_normalized * self.std[0] + self.mean[0]
        return y_original
        
    def get_single_sample(self, sample_idx=0):
        """
        获取单个测试样本
        
        参数:
            sample_idx: 样本索引
        
        返回:
            X_sample: 单个样本输入, 形状 (24, 3)
            y_sample: 单个样本标签, 标量
        """
        if self.X_test is None:
            self.load_test_data()
            
        if sample_idx >= len(self.X_test):
            raise IndexError(f"样本索引 {sample_idx} 超出范围 (最大: {len(self.X_test)-1})")
            
        X_sample = self.X_test[sample_idx]
        y_sample = self.y_test[sample_idx]
        
        return X_sample, y_sample
