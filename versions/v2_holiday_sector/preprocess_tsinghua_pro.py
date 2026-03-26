
import os
import sys
import pickle
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# ============================================================
# 配置管理
# ============================================================

class Config:
    """集中配置管理"""
    
    # 项目路径
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    RAW_DIR = DATA_ROOT / "raw" / "tsinghua"
    PROCESSED_DIR = DATA_ROOT / "processed" / "tsinghua"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # 数据文件
    DATASETS = {
        '4g': {'file': 'Performance_4G_Weekday.txt', 'type': '4g'},
        '5g': {'file': 'Performance_5G_Weekday.txt', 'type': '5g'},
        '4g_weekend': {'file': 'Performance_4G_Weekend.txt', 'type': '4g'},
        '5g_weekend': {'file': 'Performance_5G_Weekend.txt', 'type': '5g'},
    }
    
    # 列名常量
    COL_STATION = 'Base Station ID'
    COL_CELL = 'Cell ID'
    COL_TIME = 'Timestamp'
    COL_PRB = 'PRB Usage Ratio (%)'
    COL_TRAFFIC = 'Traffic Volume (KByte)'
    COL_USERS = 'Number of Users'
    COL_BBU = 'BBU Energy (W)'
    COL_RRU = 'RRU Energy (W)'
    COL_CHANNEL = 'Channel Shutdown Time (Millisecond)'
    COL_DEEP = 'Deep Sleep Time (Millisecond)'
    
    # 特征列
    FEATURE_COLS = [COL_PRB, COL_TRAFFIC, COL_USERS]
    
    # 处理参数
    RANDOM_SEED = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    
    # 日志级别
    LOG_LEVEL = logging.INFO


# ============================================================
# 日志系统
# ============================================================

class Logger:
    """统一日志管理"""
    
    @staticmethod
    def setup(name: str = __name__) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(Config.LOG_LEVEL)
        
        if not logger.handlers:
            # 控制台
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(console)
            
            # 文件
            Config.LOG_DIR.mkdir(exist_ok=True)
            file = logging.FileHandler(
                Config.LOG_DIR / f'tsinghua_preprocess_{datetime.now():%Y%m%d_%H%M%S}.log',
                encoding='utf-8'
            )
            file.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file)
        
        return logger


logger = Logger.setup()


# ============================================================
# 数据验证
# ============================================================

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_schema(df: pd.DataFrame, dataset_type: str) -> bool:
        """验证数据模式"""
        expected_cols = [Config.COL_STATION, Config.COL_CELL, Config.COL_TIME,
                         Config.COL_PRB, Config.COL_TRAFFIC, Config.COL_USERS,
                         Config.COL_BBU, Config.COL_RRU]
        
        if dataset_type == '5g':
            expected_cols.extend([Config.COL_CHANNEL, Config.COL_DEEP])
        
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            logger.error(f"缺失列: {missing}")
            return False
        
        logger.info(f"✓ 模式验证通过: {len(df.columns)} 列")
        return True
    
    @staticmethod
    def validate_values(df: pd.DataFrame) -> pd.DataFrame:
        """验证并清洗数值"""
        # PRB范围 0-100
        if Config.COL_PRB in df.columns:
            invalid = (df[Config.COL_PRB] < 0) | (df[Config.COL_PRB] > 100)
            if invalid.any():
                logger.warning(f"修复 {invalid.sum()} 个PRB异常值")
                df.loc[invalid, Config.COL_PRB] = df[Config.COL_PRB].clip(0, 100)
        
        # 能耗非负
        for col in [Config.COL_BBU, Config.COL_RRU]:
            if col in df.columns:
                invalid = df[col] < 0
                if invalid.any():
                    logger.warning(f"修复 {invalid.sum()} 个{col}负值")
                    df.loc[invalid, col] = 0
        
        # 流量非负
        if Config.COL_TRAFFIC in df.columns:
            invalid = df[Config.COL_TRAFFIC] < 0
            if invalid.any():
                logger.warning(f"修复 {invalid.sum()} 个流量负值")
                df.loc[invalid, Config.COL_TRAFFIC] = 0
        
        return df


# ============================================================
# 特征工程
# ============================================================

class FeatureEngineer:
    """特征工程"""
    
    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        if Config.COL_TIME not in df.columns:
            return df
        
        # 解析时间
        time_parts = df[Config.COL_TIME].str.split(':', expand=True)
        hour = time_parts[0].astype(int)
        
        # 周期性编码
        df['hour'] = hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        logger.info(f"✓ 添加时间特征: hour, hour_sin, hour_cos")
        return df
    
    @staticmethod
    def add_energy_target(df: pd.DataFrame) -> pd.DataFrame:
        """添加能耗目标"""
        df['total_energy'] = df[Config.COL_BBU] + df[Config.COL_RRU]
        logger.info(f"✓ 添加目标: total_energy (范围: {df['total_energy'].min():.2f} - {df['total_energy'].max():.2f})")
        return df
    
    @staticmethod
    def add_5g_features(df: pd.DataFrame) -> pd.DataFrame:
        """添加5G特有特征"""
        if Config.COL_CHANNEL in df.columns:
            df['channel_shutdown'] = df[Config.COL_CHANNEL].fillna(0)
            df['deep_sleep'] = df[Config.COL_DEEP].fillna(0)
            logger.info(f"✓ 添加5G节能特征: channel_shutdown, deep_sleep")
        return df
    
    @staticmethod
    def get_feature_matrix(df: pd.DataFrame, is_5g: bool) -> np.ndarray:
        """构建特征矩阵"""
        features = [Config.COL_PRB, Config.COL_TRAFFIC, Config.COL_USERS]
        
        # 时间特征
        if 'hour_sin' in df.columns:
            features.extend(['hour_sin', 'hour_cos'])
        
        # 5G特征
        if is_5g and 'channel_shutdown' in df.columns:
            features.extend(['channel_shutdown', 'deep_sleep'])
        
        X = df[features].values.astype(np.float32)
        logger.info(f"✓ 特征矩阵: {X.shape}")
        return X


# ============================================================
# 归一化器
# ============================================================

class StationNormalizer:
    """基站独立归一化"""
    
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StationNormalizer':
        self.scaler_X.fit(X)
        self.scaler_y.fit(y)
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("必须先调用fit")
        return self.scaler_X.transform(X), self.scaler_y.transform(y)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)
    
    def save(self, path: Path):
        with open(path / 'scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open(path / 'scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
    
    @classmethod
    def load(cls, path: Path) -> 'StationNormalizer':
        norm = cls()
        with open(path / 'scaler_X.pkl', 'rb') as f:
            norm.scaler_X = pickle.load(f)
        with open(path / 'scaler_y.pkl', 'rb') as f:
            norm.scaler_y = pickle.load(f)
        norm.fitted = True
        return norm


# ============================================================
# 基站处理器
# ============================================================

class StationProcessor:
    """单基站处理器"""
    
    def __init__(self, station_id: int, df: pd.DataFrame, is_5g: bool):
        self.station_id = station_id
        self.df = df.copy()
        self.is_5g = is_5g
        self.n_samples = len(df)
    
    def process(self, output_dir: Path) -> Dict:
        """处理单基站"""
        try:
            # 排序
            self.df = self.df.sort_values(Config.COL_TIME)
            
            # 特征工程
            self.df = FeatureEngineer.add_time_features(self.df)
            self.df = FeatureEngineer.add_energy_target(self.df)
            if self.is_5g:
                self.df = FeatureEngineer.add_5g_features(self.df)
            
            # 构建特征和目标
            X = FeatureEngineer.get_feature_matrix(self.df, self.is_5g)
            y = self.df['total_energy'].values.reshape(-1, 1).astype(np.float32)
            
            # 数据划分
            indices = np.arange(len(X))
            train_idx, temp_idx = train_test_split(
                indices, train_size=Config.TRAIN_RATIO, random_state=Config.RANDOM_SEED
            )
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=Config.VAL_RATIO / (1 - Config.TRAIN_RATIO),
                random_state=Config.RANDOM_SEED
            )
            
            # 归一化
            normalizer = StationNormalizer()
            X_norm, y_norm = normalizer.fit_transform(X, y)
            
            # 保存
            station_dir = output_dir / f"station_{self.station_id}"
            station_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            data = {
                'station_id': self.station_id,
                'n_samples': self.n_samples,
                'features': X_norm,
                'target': y_norm,
                'features_raw': X,
                'target_raw': y,
                'indices': {
                    'train': train_idx,
                    'val': val_idx,
                    'test': test_idx
                },
                'timestamps': self.df[Config.COL_TIME].values,
                'total_energy_raw': self.df['total_energy'].values
            }
            
            with open(station_dir / 'data.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            normalizer.save(station_dir)
            
            return {
                'station_id': self.station_id,
                'n_samples': self.n_samples,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'test_samples': len(test_idx),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"基站 {self.station_id} 处理失败: {e}")
            return {
                'station_id': self.station_id,
                'status': 'failed',
                'error': str(e)
            }


# ============================================================
# 主流程
# ============================================================

class TsinghuaPreprocessor:
    """清华数据预处理器"""
    
    def __init__(self):
        self.results = {}
    
    def process_dataset(self, name: str, file_name: str, net_type: str) -> Dict:
        """处理单个数据集"""
        logger.info(f"\n{'='*60}")
        logger.info(f"处理: {name} ({net_type})")
        logger.info(f"{'='*60}")
        
        # 创建输出目录
        output_dir = Config.PROCESSED_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        file_path = Config.RAW_DIR / file_name
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return {'status': 'failed', 'error': 'file not found'}
        
        df = pd.read_csv(file_path)
        logger.info(f"✓ 加载数据: {len(df):,} 行, {len(df.columns)} 列")
        
        # 验证
        if not DataValidator.validate_schema(df, net_type):
            return {'status': 'failed', 'error': 'schema validation failed'}
        
        df = DataValidator.validate_values(df)
        
        # 获取所有基站
        stations = df[Config.COL_STATION].unique()
        logger.info(f"✓ 发现基站: {len(stations):,} 个")
        
        # 处理每个基站
        results = []
        is_5g = net_type == '5g'
        
        for i, sid in enumerate(stations):
            if (i + 1) % 1000 == 0:
                logger.info(f"  进度: {i+1:,}/{len(stations):,}")
            
            station_df = df[df[Config.COL_STATION] == sid]
            processor = StationProcessor(sid, station_df, is_5g)
            result = processor.process(output_dir)
            results.append(result)
        
        # 统计
        success = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - success
        
        logger.info(f"\n✓ 完成: 成功 {success:,}, 失败 {failed}")
        
        # 保存汇总
        summary = {
            'dataset': name,
            'net_type': net_type,
            'total_stations': len(stations),
            'success': success,
            'failed': failed,
            'results': results,
            'processed_at': datetime.now().isoformat()
        }
        
        with open(output_dir / 'summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        # 保存JSON版本（便于查看）
        json_summary = {k: v for k, v in summary.items() if k != 'results'}
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        
        return summary
    
    def run(self):
        """运行所有处理"""
        logger.info("="*60)
        logger.info("清华数据预处理 - 五星级专业版")
        logger.info("="*60)
        
        # 创建目录
        Config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # 处理每个数据集
        for name, info in Config.DATASETS.items():
            summary = self.process_dataset(name, info['file'], info['type'])
            self.results[name] = summary
        
        # 最终报告
        logger.info("\n" + "="*60)
        logger.info("处理完成")
        logger.info("="*60)
        for name, summary in self.results.items():
            if summary and summary.get('status') != 'failed':
                logger.info(f"{name}: {summary['success']:,}/{summary['total_stations']:,}")
        
        # 保存全局汇总
        with open(Config.PROCESSED_DIR / 'all_summaries.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        return self.results


# ============================================================
# 入口
# ============================================================

def main():
    np.random.seed(Config.RANDOM_SEED)
    preprocessor = TsinghuaPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()
EOF

# 运行
