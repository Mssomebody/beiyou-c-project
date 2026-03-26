"""
调试联邦训练 - 找出卡住的位置
"""

import sys
import torch
from pathlib import Path
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("1. 导入模块...")
from versions.v2_holiday_sector.train_federated_pro import DataLoaderFactory, FedConfig, FederatedTrainer, Logger

print("2. 加载配置...")
config_path = Path(__file__).parent / "configs" / "paths.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    paths = yaml.safe_load(f)

data_path = Path(paths['data_root']) / paths['barcelona'][paths['current']['barcelona']]
print(f"   数据路径: {data_path}")

print("3. 创建配置...")
fed_config = FedConfig()
fed_config.nodes = [8001]
fed_config.rounds = 1
fed_config.data_path = data_path

logger = Logger(fed_config)

print("4. 加载数据...")
train_loaders, val_loaders, test_loaders = DataLoaderFactory.load_all_loaders(fed_config)
print(f"   加载成功，节点: {list(train_loaders.keys())}")

print("5. 获取一个 batch...")
loader = train_loaders[8001]
batch = next(iter(loader))
x, y = batch
print(f"   x 形状: {x.shape}, y 形状: {y.shape}")

print("6. 创建模型...")
trainer = FederatedTrainer(fed_config, logger)
model = trainer._create_model()

print("7. 测试前向传播...")
model.eval()
with torch.no_grad():
    output = model(x)
print(f"   输出形状: {output.shape}")

print("8. 测试单轮训练...")
mu = 0.01
try:
    val_loss, test_smape = trainer.train_full(mu, train_loaders, val_loaders, test_loaders)
    print(f"   完成: val_loss={val_loss:.6f}, test_smape={test_smape:.2f}%")
except Exception as e:
    import traceback
    print(f"   训练失败: {e}")
    traceback.print_exc()

print("\n✅ 调试完成")
