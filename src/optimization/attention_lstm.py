"""
v4: 注意力机制 + 双向LSTM

功能：
- 双向 LSTM
- 自注意力机制
- 多头注意力
- 残差连接
- 层归一化

专业特性：
- 参数化所有关键配置
- 完整的输入验证
- 支持多种注意力类型
- 可配置的模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union
import logging

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    """
    自注意力机制
    
    参数:
        hidden_dim: int, 隐藏层维度
        dropout: float, dropout 比例
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        
        Returns:
            context: [batch, hidden_dim]
            weights: [batch, seq_len]
        """
        # 计算注意力权重
        weights = self.attention(x)  # [batch, seq_len, 1]
        weights = weights.squeeze(-1)  # [batch, seq_len]
        weights = torch.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        # 加权求和
        context = torch.sum(weights.unsqueeze(-1) * x, dim=1)  # [batch, hidden_dim]
        
        return context, weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    参数:
        hidden_dim: int, 隐藏层维度
        num_heads: int, 头数
        dropout: float, dropout 比例
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        
        Returns:
            out: [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = torch.matmul(attn, v)  # [batch, heads, seq, head_dim]
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)  # [batch, seq, hidden]
        
        return self.out(out)


class AttentionLSTM(nn.Module):
    """
    注意力增强 LSTM
    
    参数:
        input_dim: int, 输入特征维度
        hidden_dim: int, LSTM 隐藏层维度
        num_layers: int, LSTM 层数
        output_dim: int, 输出维度
        dropout: float, dropout 比例
        bidirectional: bool, 是否双向
        attention_type: str, 注意力类型 ('none', 'self', 'multihead')
        num_heads: int, 多头注意力头数（仅 multihead 时有效）
        use_residual: bool, 是否使用残差连接
        use_layer_norm: bool, 是否使用层归一化
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention_type: Literal['none', 'self', 'multihead'] = 'self',
        num_heads: int = 4,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        # 参数验证
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0,1], got {dropout}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.attention_type = attention_type
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 注意力层
        if attention_type == 'self':
            self.attention = SelfAttention(lstm_out_dim, dropout)
        elif attention_type == 'multihead':
            self.attention = MultiHeadAttention(lstm_out_dim, num_heads, dropout)
        else:
            self.attention = None
        
        # 层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_out_dim)
        
        # 输出层
        self.fc = nn.Linear(lstm_out_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, seq_len, input_dim]
        
        Returns:
            output: [batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_out_dim]
        
        # 残差连接（取最后一个时间步）
        residual = lstm_out[:, -1, :]
        
        # 注意力
        if self.attention_type != 'none':
            if self.attention_type == 'multihead':
                attended = self.attention(lstm_out)
                context = attended[:, -1, :]  # 取最后一个时间步
            else:
                context, _ = self.attention(lstm_out)
        else:
            context = residual
        
        # 层归一化 + 残差
        if self.use_layer_norm:
            if self.use_residual:
                out = self.layer_norm(context + residual)
            else:
                out = self.layer_norm(context)
        else:
            out = context + residual if self.use_residual else context
        
        out = self.dropout(out)
        
        return self.fc(out)
    
    def count_parameters(self) -> int:
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters())


def create_attention_model(
    input_dim: int,
    output_dim: int = 4,
    **kwargs
) -> AttentionLSTM:
    """
    创建注意力模型（工厂函数）
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        **kwargs: 传递给 AttentionLSTM 的参数
    
    Returns:
        AttentionLSTM 实例
    """
    default_config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'attention_type': 'self',
        'num_heads': 4,
        'use_residual': True,
        'use_layer_norm': True
    }
    
    default_config.update(kwargs)
    
    return AttentionLSTM(
        input_dim=input_dim,
        output_dim=output_dim,
        **default_config
    )


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 v4 注意力机制")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, seq_len, input_dim = 32, 28, 7
    
    # 测试1：自注意力
    print("\n测试1: 自注意力")
    model1 = create_attention_model(
        input_dim=input_dim,
        attention_type='self',
        hidden_dim=128
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    y = model1(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  参数量: {model1.count_parameters():,}")
    
    # 测试2：多头注意力
    print("\n测试2: 多头注意力")
    model2 = create_attention_model(
        input_dim=input_dim,
        attention_type='multihead',
        hidden_dim=128,
        num_heads=4
    ).to(device)
    
    y = model2(x)
    print(f"  输出: {y.shape}")
    print(f"  参数量: {model2.count_parameters():,}")
    
    # 测试3：无注意力
    print("\n测试3: 无注意力（基线）")
    model3 = create_attention_model(
        input_dim=input_dim,
        attention_type='none',
        hidden_dim=128
    ).to(device)
    
    y = model3(x)
    print(f"  输出: {y.shape}")
    print(f"  参数量: {model3.count_parameters():,}")
    
    # 测试4：错误处理
    print("\n测试4: 错误处理")
    try:
        create_attention_model(input_dim=0)
    except ValueError as e:
        print(f"  ✅ 输入验证正常: {e}")
    
    try:
        create_attention_model(input_dim=7, num_heads=3, hidden_dim=128)
    except AssertionError as e:
        print(f"  ✅ 维度验证正常: {e}")
    
    print("\n✅ 测试通过")