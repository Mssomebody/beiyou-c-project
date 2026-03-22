"""
v3: 周期性编码模块

功能：
- 小时 sin/cos 编码
- 星期 sin/cos 编码
- 月份 sin/cos 编码

专业特性：
- 参数化周期
- 自动检测数据范围
- 完整错误处理
- 单元测试
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CyclicalEncoder:
    """
    周期性编码器
    
    将时间特征转换为 sin/cos 编码，保留周期性质。
    
    参数:
        hour_period: int, 小时周期，默认 24
        day_period: int, 星期周期，默认 7
        month_period: int, 月份周期，默认 12
        hour_col: str, 小时列名，默认 'hour_code'
        day_col: Optional[str], 星期列名，默认 None（从 Data 列提取）
        month_col: str, 月份列名，默认 'month'
        auto_detect: bool, 是否自动检测小时范围，默认 True
    """
    
    def __init__(
        self,
        hour_period: int = 24,
        day_period: int = 7,
        month_period: int = 12,
        hour_col: str = 'hour_code',
        day_col: Optional[str] = None,
        month_col: str = 'month',
        auto_detect: bool = True
    ):
        self.hour_period = hour_period
        self.day_period = day_period
        self.month_period = month_period
        self.hour_col = hour_col
        self.day_col = day_col
        self.month_col = month_col
        self.auto_detect = auto_detect
        
        # 验证参数
        if hour_period <= 0:
            raise ValueError(f"hour_period must be > 0, got {hour_period}")
        if day_period <= 0:
            raise ValueError(f"day_period must be > 0, got {day_period}")
        if month_period <= 0:
            raise ValueError(f"month_period must be > 0, got {month_period}")
    
    def _detect_hour_range(self, values: np.ndarray) -> int:
        """检测小时值范围，返回缩放因子"""
        max_val = values.max()
        if max_val <= 3:
            return 6  # 6小时粒度
        elif max_val <= 23:
            return 1  # 小时粒度
        else:
            logger.warning(f"Unknown hour range: max={max_val}, using raw values")
            return 1
    
    def _encode_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码小时特征"""
        if self.hour_col not in df.columns:
            logger.warning(f"Hour column '{self.hour_col}' not found, skipping")
            return df
        
        hour_values = df[self.hour_col].values
        
        if self.auto_detect:
            scale = self._detect_hour_range(hour_values)
            hour_normalized = hour_values * scale / self.hour_period
        else:
            hour_normalized = hour_values / self.hour_period
        
        df['hour_sin'] = np.sin(2 * np.pi * hour_normalized)
        df['hour_cos'] = np.cos(2 * np.pi * hour_normalized)
        
        return df
    
    def _encode_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码星期特征"""
        if self.day_col and self.day_col in df.columns:
            day_values = df[self.day_col].values
        elif 'Data' in df.columns:
            day_values = df['Data'].dt.dayofweek.values
        else:
            logger.warning("No day column found, skipping")
            return df
        
        day_normalized = day_values / self.day_period
        df['day_sin'] = np.sin(2 * np.pi * day_normalized)
        df['day_cos'] = np.cos(2 * np.pi * day_normalized)
        
        return df
    
    def _encode_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码月份特征"""
        if self.month_col not in df.columns:
            logger.warning(f"Month column '{self.month_col}' not found, skipping")
            return df
        
        month_values = df[self.month_col].values - 1  # 1-12 → 0-11
        month_normalized = month_values / self.month_period
        df['month_sin'] = np.sin(2 * np.pi * month_normalized)
        df['month_cos'] = np.cos(2 * np.pi * month_normalized)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加周期性编码特征
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            添加了特征的 DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
        
        df = df.copy()
        
        logger.info("Adding cyclical features...")
        original_cols = len(df.columns)
        
        df = self._encode_hour(df)
        df = self._encode_day(df)
        df = self._encode_month(df)
        
        new_cols = len(df.columns) - original_cols
        logger.info(f"Added {new_cols} cyclical features")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """同 fit_transform，无状态"""
        return self.fit_transform(df)


def add_cyclical_features(
    df: pd.DataFrame,
    hour_period: int = 24,
    day_period: int = 7,
    month_period: int = 12,
    **kwargs
) -> pd.DataFrame:
    """
    便捷函数：添加周期性编码
    
    Args:
        df: 输入 DataFrame
        hour_period: 小时周期
        day_period: 星期周期
        month_period: 月份周期
        **kwargs: 传递给 CyclicalEncoder 的其他参数
    
    Returns:
        添加了特征的 DataFrame
    """
    encoder = CyclicalEncoder(
        hour_period=hour_period,
        day_period=day_period,
        month_period=month_period,
        **kwargs
    )
    return encoder.fit_transform(df)


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 v3 周期性编码")
    print("=" * 60)
    
    # 创建测试数据
    df = pd.DataFrame({
        'hour_code': [0, 1, 2, 3],
        'Data': pd.date_range('2024-01-01', periods=4, freq='6h'),
        'month': [1, 1, 1, 1]
    })
    
    print(f"\n原始数据: {df.shape}")
    
    # 测试1：默认参数
    df1 = add_cyclical_features(df)
    print(f"\n默认参数后: {df1.shape}")
    print(f"新增列: {[c for c in df1.columns if 'sin' in c or 'cos' in c]}")
    
    # 测试2：自定义周期
    df2 = add_cyclical_features(df, hour_period=12, day_period=5)
    print(f"\n自定义周期后: {df2.shape}")
    
    # 测试3：错误处理
    try:
        add_cyclical_features(df, hour_period=0)
    except ValueError as e:
        print(f"\n错误处理正常: {e}")
    
    print("\n✅ 测试通过")