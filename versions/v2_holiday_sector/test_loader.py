"""
专业测试脚本：使用配置文件 + 正确的数据加载器
"""

import sys
import os
import yaml
from pathlib import Path

# 获取项目根目录（自动，不写死）
def get_project_root():
    return Path(__file__).parent.parent.parent

PROJECT_ROOT = get_project_root()

# 加载配置
config_path = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "configs" / "paths.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    paths = yaml.safe_load(f)

data_root = Path(paths['data_root'])
data_version = paths['barcelona'][paths['current']['barcelona']]
data_path = data_root / data_version

print(f"项目根目录: {PROJECT_ROOT}")
print(f"数据版本: {data_version}")
print(f"数据路径: {data_path}")

# 添加项目路径
sys.path.insert(0, str(PROJECT_ROOT))

# 使用支持 data_version 的加载器（如果有）
try:
    from src.data_loader.barcelona_dataset_v1 import get_node_data_loader
except ImportError:
    print("导入失败，使用备用加载器")
    sys.exit(1)

print("测试节点 8001...")

try:
    loader, scaler, _ = get_node_data_loader(
        node_id=8001,
        split='train',
        batch_size=64,
        shuffle=True,
        window_size=28,
        predict_size=4,
        sector_feature=True,
        holiday_feature=True,
        weekend_feature=True
    )
    print(f"✅ 加载成功，数据量: {len(loader.dataset)}")
except Exception as e:
    print(f"❌ 失败: {e}")
