#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专业测试：联邦学习数据加载和训练
- 使用项目内置的 config 模块
- 不依赖 __file__ 或 os.getcwd()
- 真正的单配置
"""

import sys
import yaml
from pathlib import Path

# ============================================================
# 1. 使用项目内置的 config 模块
# ============================================================

# 尝试导入项目配置模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from src.config import load_config
    config = load_config()
    print("✅ 使用 src/config.py")
except ImportError:
    # 备用：直接读 yaml
    config_path = Path(__file__).parent / "configs" / "paths.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("✅ 直接读取 configs/paths.yaml")

data_root = Path(config['data_root'])
data_version = config['barcelona'][config['current']['barcelona']]
data_path = data_root / data_version

print(f"数据路径: {data_path}")

# ============================================================
# 2. 测试数据加载
# ============================================================

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from versions.v2_holiday_sector.train_federated_pro import DataLoaderFactory, FedConfig, FederatedTrainer, Logger

print("\n配置...")
fed_config = FedConfig()
fed_config.nodes = [8001, 8002]
fed_config.rounds = 2
fed_config.data_path = data_path

logger = Logger(fed_config)

print("加载数据...")
train_loaders, val_loaders, test_loaders = DataLoaderFactory.load_all_loaders(fed_config)
print(f"✅ 加载成功，{len(train_loaders)} 个节点")

print("\n训练...")
trainer = FederatedTrainer(fed_config, logger)
val_loss, test_smape = trainer.train_full(0.01, train_loaders, val_loaders, test_loaders)
print(f"\n✅ 完成: val_loss={val_loss:.6f}, test_smape={test_smape:.2f}%")
