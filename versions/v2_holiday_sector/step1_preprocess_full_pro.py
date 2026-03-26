#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤1：全量数据预处理 - 五星级专业版
- 4G: 全部 12,162 个基站
- 5G: 全部 5,165 个基站
- 支持断点续传
- 完整日志
"""

import os
import sys
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 配置管理
# ============================================================

@dataclass
class Config:
    """集中配置管理"""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    raw_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    
    # 数据文件
    files: Dict[str, str] = field(default_factory=lambda: {
        '4g': 'Performance_4G_Weekday.txt',
        '5g': 'Performance_5G_Weekday.txt'
    })
    
    # 公共特征
    common_features: List[str] = field(default_factory=lambda: [
        'PRB Usage Ratio (%)',
        'Traffic Volume (KByte)',
        'Number of Users'
    ])
    
    # 训练配置
    train_ratio: float = 0.8
    seq_len: int = 24
    
    # 随机种子
    seed: int = 42
    
    # 断点续传
    checkpoint_file: str = 'checkpoint.pkl'
    
    def __post_init__(self):
        if self.raw_dir is None:
            self.raw_dir = self.project_root / "data" / "raw" / "tsinghua"
        if self.output_dir is None:
            self.output_dir = self.project_root / "data" / "processed" / "tsinghua_full"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(self.seed)


# ============================================================
# 日志系统
# ============================================================

def setup_logger(name: str = __name__, config: Config = None) -> logging.Logger:
    """配置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    # 控制台
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)
    
    # 文件
    if config:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = config.logs_dir / f"preprocess_full_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# 数据处理器
# ============================================================

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict:
        """加载断点"""
        checkpoint_path = self.config.output_dir / self.config.checkpoint_file
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_checkpoint(self, data_type: str, station_id: int):
        """保存断点"""
        checkpoint_path = self.config.output_dir / self.config.checkpoint_file
        self.checkpoint[f"{data_type}_{station_id}"] = True
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.checkpoint, f)
    
    def process_station(self, df: pd.DataFrame, station_id: int, data_type: str) -> Optional[Dict]:
        """处理单个基站"""
        try:
            station_df = df[df['Base Station ID'] == station_id].copy()
            station_df = station_df.sort_values('Timestamp')
            
            # 特征提取
            X = station_df[self.config.common_features].values.astype(np.float32)
            y = station_df['Total_Energy'].values.astype(np.float32).reshape(-1, 1)
            
            # 时间特征
            hour = station_df['Timestamp'].str.split(':').str[0].astype(int).values
            hour_sin = np.sin(2 * np.pi * hour / 24).reshape(-1, 1)
            hour_cos = np.cos(2 * np.pi * hour / 24).reshape(-1, 1)
            X = np.hstack([X, hour_sin, hour_cos])
            
            # 划分训练/测试
            n = len(X)
            train_end = int(n * self.config.train_ratio)
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[train_end:]
            y_test = y[train_end:]
            y_test_raw = y_test.copy()
            
            # 归一化
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train_norm = scaler_X.fit_transform(X_train)
            y_train_norm = scaler_y.fit_transform(y_train)
            X_test_norm = scaler_X.transform(X_test)
            y_test_norm = scaler_y.transform(y_test)
            
            # 保存
            station_dir = self.config.output_dir / data_type / f"station_{station_id}"
            station_dir.mkdir(parents=True, exist_ok=True)
            
            data = {
                'station_id': station_id,
                'n_samples': n,
                'n_train': train_end,
                'n_test': n - train_end,
                'X_train_norm': X_train_norm,
                'y_train_norm': y_train_norm,
                'X_test_norm': X_test_norm,
                'y_test_norm': y_test_norm,
                'y_test_raw': y_test_raw,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
            }
            
            with open(station_dir / 'data.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            self._save_checkpoint(data_type, station_id)
            
            return {
                'station_id': station_id,
                'n_samples': n,
                'n_train': train_end,
                'n_test': n - train_end,
            }
            
        except Exception as e:
            self.logger.error(f"处理基站 {station_id} 失败: {e}")
            return None
    
    def process_dataset(self, data_type: str) -> List[Dict]:
        """处理完整数据集"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"处理 {data_type} 数据")
        self.logger.info(f"{'='*60}")
        
        file_path = self.config.raw_dir / self.config.files[data_type]
        if not file_path.exists():
            self.logger.error(f"文件不存在: {file_path}")
            return []
        
        df = pd.read_csv(file_path)
        df['Total_Energy'] = df['BBU Energy (W)'] + df['RRU Energy (W)']
        
        stations = df['Base Station ID'].unique()
        self.logger.info(f"基站总数: {len(stations):,}")
        
        results = []
        failed = []
        
        for station_id in tqdm(stations, desc=f"处理 {data_type}"):
            # 检查断点
            if self.checkpoint.get(f"{data_type}_{station_id}"):
                continue
            
            result = self.process_station(df, station_id, data_type)
            if result:
                results.append(result)
            else:
                failed.append(station_id)
        
        # 统计
        total_samples = sum(r['n_samples'] for r in results)
        self.logger.info(f"\n  成功: {len(results):,} 个基站")
        self.logger.info(f"  失败: {len(failed):,} 个基站")
        self.logger.info(f"  总样本数: {total_samples:,}")
        
        if failed:
            self.logger.warning(f"失败基站: {failed[:10]}...")
        
        return results


# ============================================================
# 主函数
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='全量数据预处理')
    parser.add_argument('--data_type', type=str, default='all', 
                        choices=['all', '4g', '5g'],
                        help='数据类型')
    parser.add_argument('--resume', action='store_true',
                        help='断点续传')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    config = Config()
    logger = setup_logger(__name__, config)
    
    logger.info("="*60)
    logger.info("步骤1：全量数据预处理 - 五星级专业版")
    logger.info("="*60)
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"断点续传: {args.resume}")
    
    processor = DataProcessor(config, logger)
    
    data_types = []
    if args.data_type == 'all':
        data_types = ['4g', '5g']
    else:
        data_types = [args.data_type]
    
    for data_type in data_types:
        processor.process_dataset(data_type)
    
    logger.info("\n✅ 预处理完成")


if __name__ == "__main__":
    main()
