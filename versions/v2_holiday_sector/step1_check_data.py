#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4G+5G代际协同 - 第1步：数据格式检查
"""

import os
import pickle
import logging
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置
# ============================================================

class Config:
    """配置类"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed" / "tsinghua"
    
    # 日志配置
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# ============================================================
# 日志
# ============================================================

def setup_logger(name: str = __name__) -> logging.Logger:
    """配置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        logger.addHandler(console)
    
    return logger


logger = setup_logger()


# ============================================================
# 数据检查
# ============================================================

def check_data_structure(data_dir: Path) -> dict:
    """检查数据结构"""
    result = {}
    
    for data_type in ['4g', '5g', '4g_weekend', '5g_weekend']:
        type_dir = data_dir / data_type
        if not type_dir.exists():
            logger.warning(f"目录不存在: {type_dir}")
            continue
        
        # 统计基站数量
        stations = [d for d in type_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
        n_stations = len(stations)
        
        # 检查第一个基站的数据格式
        if n_stations > 0:
            sample_station = stations[0]
            data_path = sample_station / 'data.pkl'
            
            try:
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                
                result[data_type] = {
                    'n_stations': n_stations,
                    'sample_station': sample_station.name,
                    'n_samples': data.get('n_samples', 'N/A'),
                    'features_shape': data.get('features', np.array([])).shape if 'features' in data else 'N/A',
                    'target_shape': data.get('target', np.array([])).shape if 'target' in data else 'N/A',
                }
            except Exception as e:
                logger.error(f"读取失败 {data_path}: {e}")
                result[data_type] = {'n_stations': n_stations, 'error': str(e)}
    
    return result


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("4G+5G代际协同 - 数据格式检查")
    logger.info("="*60)
    logger.info(f"数据目录: {Config.DATA_DIR}")
    
    if not Config.DATA_DIR.exists():
        logger.error(f"数据目录不存在: {Config.DATA_DIR}")
        return
    
    # 检查数据
    result = check_data_structure(Config.DATA_DIR)
    
    # 打印结果
    logger.info("\n数据统计:")
    for data_type, info in result.items():
        if 'error' in info:
            logger.info(f"  {data_type}: 错误 - {info['error']}")
        else:
            logger.info(f"  {data_type}: {info['n_stations']} 个基站")
            logger.info(f"      样本基站: {info['sample_station']}")
            logger.info(f"      样本数: {info['n_samples']}")
            logger.info(f"      特征形状: {info['features_shape']}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Config.PROJECT_ROOT / "results" / f"data_check_{timestamp}.pkl"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"\n结果保存: {output_path}")
    
    return result


if __name__ == "__main__":
    import numpy as np
    main()
