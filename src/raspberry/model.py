# -*- coding: utf-8 -*-
"""
LSTM模型定义
参考fedprox_comparison.py中的模型结构
用于树莓派推理
"""

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    LSTM预测模型
    输入: 过去24小时的3个特征
    输出: 未来6小时的功耗预测
    """
    
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=6):
        """
        初始化LSTM模型
        
        参数:
            input_size: 输入特征数 (默认3: 功耗,业务量,温度)
            hidden_size: LSTM隐藏层大小 (默认64)
            num_layers: LSTM层数 (默认2)
            output_size: 输出预测点数 (默认6: 未来6小时)
        """
        super(LSTMPredictor, self).__init__()
        
        # 保存模型参数
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入格式: (batch, seq_len, input_size)
        )
        
        # 全连接层: 将LSTM输出映射到预测值
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量, 形状 (batch_size, seq_len, input_size)
               batch_size: 批量大小
               seq_len: 时间序列长度 (24小时)
               input_size: 特征数 (3个特征)
        
        返回:
            out: 预测输出, 形状 (batch_size, output_size)
                 output_size: 预测点数 (6小时)
        """
        # 初始化LSTM隐藏状态和细胞状态
        # h0形状: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        # out形状: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出: (batch_size, hidden_size)
        # 并通过全连接层得到预测值
        out = self.fc(out[:, -1, :])
        
        return out


def create_model(input_size=3, hidden_size=64, num_layers=2, output_size=6):
    """
    工厂函数: 创建LSTM模型实例
    
    参数:
        同LSTMPredictor.__init__
    
    返回:
        model: LSTMPredictor模型实例
    """
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )
    return model
