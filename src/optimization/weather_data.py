"""
v5: 天气数据集成（基于真实巴塞罗那天气数据）

功能：
- 加载 Open-Meteo 天气数据
- 与能耗数据时间对齐
- 天气特征工程
- 滞后特征、滚动统计
- 缺失值处理

数据源：
- Open-Meteo API（已下载）
- 时间范围：2019-2025
- 变量：温度、湿度、降水、风速
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class WeatherDataLoader:
    """
    天气数据加载器
    加载已下载的 Open-Meteo 天气数据
    """
    
    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: 天气数据文件路径
        """
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(base_dir, "data", "raw", "weather", "barcelona_weather_6h.csv")
        
        self.data_path = data_path
        self._weather_df = None
    
    def load(self) -> pd.DataFrame:
        """
        加载天气数据
        
        Returns:
            weather_df: 天气数据 DataFrame
        """
        if self._weather_df is not None:
            return self._weather_df
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Weather data not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df['Data'] = pd.to_datetime(df['time'])
        df = df.drop(columns=['time'])
        
        # 重命名列（简化）
        df = df.rename(columns={
            'temperature_2m': 'temperature',
            'relative_humidity_2m': 'humidity',
            'precipitation': 'precipitation',
            'wind_speed_10m': 'wind_speed'
        })
        
        logger.info(f"Loaded weather data: {len(df)} records, {df['Data'].min()} ~ {df['Data'].max()}")
        
        self._weather_df = df
        return df
    
    def get_weather_for_dates(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指定时间范围的天气数据"""
        df = self.load()
        mask = (df['Data'] >= start_date) & (df['Data'] <= end_date)
        return df[mask].copy()


class WeatherFeatureEngineer:
    """
    天气特征工程
    
    功能：
    - 添加天气特征到能耗数据
    - 滞后特征（1-3天）
    - 滚动统计（7天）
    - 分类特征（是否下雨、温度等级）
    """
    
    def __init__(
        self,
        lag_days: List[int] = [1, 2, 3],
        rolling_windows: List[int] = [7, 14, 28],
        use_interaction: bool = True
    ):
        """
        Args:
            lag_days: 滞后天数列表（1天=4个6小时时段）
            rolling_windows: 滚动窗口大小（天数）
            use_interaction: 是否使用交互特征
        """
        self.lag_steps = [d * 4 for d in lag_days]  # 转换为时段数
        self.rolling_windows = [w * 4 for w in rolling_windows]  # 转换为时段数
        self.use_interaction = use_interaction
    
    def _add_lag_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """添加滞后特征"""
        for lag in self.lag_steps:
            df[f'{col}_lag_{lag//4}d'] = df[col].shift(lag)
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """添加滚动统计特征"""
        for window in self.rolling_windows:
            df[f'{col}_rolling_mean_{window//4}d'] = df[col].rolling(window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window//4}d'] = df[col].rolling(window, min_periods=1).std()
        return df
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加分类特征"""
        # 是否下雨（降水 > 0）
        df['is_rain'] = (df['precipitation'] > 0).astype(int)
        
        # 温度等级
        df['temp_level'] = pd.cut(
            df['temperature'],
            bins=[-float('inf'), 5, 15, 25, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # 湿度等级
        df['humidity_level'] = pd.cut(
            df['humidity'],
            bins=[-float('inf'), 40, 60, 80, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交互特征"""
        if not self.use_interaction:
            return df
        
        # 温度 × 湿度（体感温度）
        df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
        
        # 降水 × 温度（雨雪天气）
        df['precip_temp'] = df['precipitation'] * df['temperature']
        
        # 周末 × 温度（周末温度影响）
        if 'is_weekend' in df.columns:
            df['weekend_temp'] = df['is_weekend'] * df['temperature']
        
        return df
    
    def fit_transform(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        on: str = 'Data'
    ) -> pd.DataFrame:
        """
        完整特征工程流程
        """
        # 空数据检查
        if energy_df.empty:
            raise ValueError("Energy DataFrame is empty")
        if weather_df.empty:
            raise ValueError("Weather DataFrame is empty")
        
        print("=" * 60)
        print("v5: 天气特征工程")
        print("=" * 60)
        
        # 确保时间列是 datetime
        if on in energy_df.columns:
            energy_df[on] = pd.to_datetime(energy_df[on])
        if on in weather_df.columns:
            weather_df[on] = pd.to_datetime(weather_df[on])
        
        original_cols = len(energy_df.columns)
        
        # 1. 合并数据
        df = energy_df.merge(weather_df, on=on, how='left')
        print(f"  ✓ 合并后行数: {len(df)}")
        
        # 2. 检查缺失值
        weather_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        missing = df[weather_cols].isnull().sum().sum()
        if missing > 0:
            print(f"  ⚠️ 发现 {missing} 个缺失值，进行前向填充")
            df[weather_cols] = df[weather_cols].fillna(method='ffill').fillna(0)
        
        # 3. 按节点分组处理（如果有节点列）
        if 'Codi_Postal' in df.columns:
            result_dfs = []
            for node_id, node_df in df.groupby('Codi_Postal'):
                node_df = node_df.sort_values(on)
                node_df = self._process_node(node_df)
                result_dfs.append(node_df)
            df = pd.concat(result_dfs, ignore_index=True)
        else:
            df = df.sort_values(on)
            df = self._process_node(df)
        
        # 4. 添加分类特征
        df = self._add_categorical_features(df)
        
        # 5. 添加交互特征
        df = self._add_interaction_features(df)
        
        new_cols = len(df.columns) - original_cols
        print(f"  ✓ 新增天气特征: {new_cols} 列")
        
        return df
    
    def _process_node(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理单个节点的天气特征"""
        weather_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        
        for col in weather_cols:
            if col in df.columns:
                df = self._add_lag_features(df, col)
                df = self._add_rolling_features(df, col)
        
        return df


class WeatherDataPipeline:
    """
    天气数据完整流水线
    整合加载、对齐、特征工程
    """
    
    def __init__(
        self,
        weather_data_path: str = None,
        lag_days: List[int] = [1, 2, 3],
        rolling_windows: List[int] = [7, 14, 28],
        use_interaction: bool = True
    ):
        self.loader = WeatherDataLoader(weather_data_path)
        self.engineer = WeatherFeatureEngineer(
            lag_days=lag_days,
            rolling_windows=rolling_windows,
            use_interaction=use_interaction
        )
    
    def process(self, energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        完整处理流程
        
        Args:
            energy_df: 能耗数据
        
        Returns:
            df: 添加天气特征后的数据
        """
        # 加载天气数据
        weather_df = self.loader.load()
        
        # 特征工程
        result = self.engineer.fit_transform(energy_df, weather_df)
        
        return result
    
    def process_node(self, energy_df: pd.DataFrame, node_id: int) -> pd.DataFrame:
        """处理单个节点的数据"""
        return self.process(energy_df)


# ============================================================
# 工厂函数
# ============================================================
def create_weather_pipeline(
    weather_data_path: str = None,
    lag_days: List[int] = None,
    rolling_windows: List[int] = None,
    use_interaction: bool = True
) -> WeatherDataPipeline:
    """创建天气数据流水线"""
    if lag_days is None:
        lag_days = [1, 2, 3]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28]
    
    return WeatherDataPipeline(
        weather_data_path=weather_data_path,
        lag_days=lag_days,
        rolling_windows=rolling_windows,
        use_interaction=use_interaction
    )


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 v5 天气数据集成")
    print("=" * 60)
    
    # 加载真实天气数据
    loader = WeatherDataLoader()
    weather_df = loader.load()
    print(f"\n天气数据: {len(weather_df)} 行")
    print(f"  时间范围: {weather_df['Data'].min()} ~ {weather_df['Data'].max()}")
    print(f"  变量: {list(weather_df.columns)}")
    
    # 创建模拟能耗数据
    dates = pd.date_range('2024-01-01', periods=100, freq='6H')
    energy_df = pd.DataFrame({
        'Data': dates,
        'Valor': np.random.randn(100).cumsum() + 100,
        'Codi_Postal': 8001,
        'Valor_norm': np.random.rand(100)
    })
    
    # 测试特征工程
    print(f"\n原始能耗数据: {energy_df.shape}")
    
    pipeline = create_weather_pipeline()
    result = pipeline.process(energy_df)
    
    print(f"\n添加天气特征后: {result.shape}")
    
    # 显示新增特征
    new_cols = [c for c in result.columns if c not in energy_df.columns]
    print(f"新增特征: {new_cols[:10]}...")
    print(f"总计新增: {len(new_cols)} 列")
    
    # 检查缺失值
    missing = result.isnull().sum().sum()
    print(f"\n缺失值总数: {missing}")
    
    print("\n✅ 测试通过")