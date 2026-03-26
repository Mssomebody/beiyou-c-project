"""
测试数据加载器 - 使用单配置，不写死路径
"""

import sys
import os
import yaml
from pathlib import Path

# 自动获取项目根目录（基于当前脚本位置）
def get_project_root():
    return Path(__file__).parent.parent.parent

PROJECT_ROOT = get_project_root()
print(f"项目根目录: {PROJECT_ROOT}")

# 加载配置
config_path = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "configs" / "paths.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    paths = yaml.safe_load(f)

data_root = Path(paths['data_root'])
data_version = paths['barcelona'][paths['current']['barcelona']]
data_path = data_root / data_version

print(f"数据版本: {data_version}")
print(f"数据路径: {data_path}")

# 添加项目路径
sys.path.insert(0, str(PROJECT_ROOT))

print("1. 导入模块...")
from versions.v2_holiday_sector.train_federated_pro import DataLoaderFactory, FedConfig

print("2. 配置...")
config = FedConfig()
config.nodes = [8001, 8002]
config.data_path = data_path

print("3. 开始加载数据...")
try:
    train_loaders, val_loaders, test_loaders = DataLoaderFactory.load_all_loaders(config)
    print(f"4. 加载成功，{len(train_loaders)} 个节点")
    for node, loader in train_loaders.items():
        print(f"   节点 {node}: {len(loader.dataset)} 样本")
except Exception as e:
    import traceback
    print(f"4. 失败: {e}")
    traceback.print_exc()
