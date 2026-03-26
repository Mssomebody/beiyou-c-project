"""
调试 train_full 函数
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

print("3. 创建配置...")
fed_config = FedConfig()
fed_config.nodes = [8001]
fed_config.rounds = 2
fed_config.data_path = data_path

logger = Logger(fed_config)

print("4. 加载数据...")
train_loaders, val_loaders, test_loaders = DataLoaderFactory.load_all_loaders(fed_config)

print("5. 创建训练器...")
trainer = FederatedTrainer(fed_config, logger)

mu = 0.01
print(f"6. 开始训练 (μ={mu}, rounds={fed_config.rounds})...")

# 手动模拟训练过程，看哪里卡住
model = trainer._create_model()

print("   6.1 创建模型完成")
print(f"   6.2 训练数据节点: {list(train_loaders.keys())}")

for round_num in range(1, fed_config.rounds + 1):
    print(f"   6.3 Round {round_num} 开始")
    
    client_weights = []
    client_sizes = []
    
    for client_id, loader in train_loaders.items():
        print(f"       客户端 {client_id} 开始本地训练...")
        
        # 本地模型
        local_model = trainer._create_model()
        local_model.load_state_dict(model.state_dict())
        
        optimizer = torch.optim.Adam(local_model.parameters(), lr=fed_config.learning_rate)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(fed_config.local_epochs):
            print(f"           Epoch {epoch+1}/{fed_config.local_epochs}")
            batch_count = 0
            for x, y in loader:
                batch_count += 1
                if batch_count == 1:
                    print(f"             第一个 batch 开始处理...")
                x, y = x.to(trainer.device), y.to(trainer.device)
                optimizer.zero_grad()
                output = local_model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                if batch_count == 1:
                    print(f"             第一个 batch 完成")
            print(f"           完成 {batch_count} 个 batch")
        
        print(f"       客户端 {client_id} 训练完成")
        client_weights.append(local_model.state_dict())
        client_sizes.append(len(loader.dataset))
    
    print(f"   6.4 Round {round_num} 聚合...")
    total = sum(client_sizes)
    global_weights = {}
    for key in client_weights[0].keys():
        global_weights[key] = torch.zeros_like(client_weights[0][key])
        for w, size in zip(client_weights, client_sizes):
            global_weights[key] += w[key] * (size / total)
    model.load_state_dict(global_weights)
    print(f"   6.5 Round {round_num} 完成")

print("7. 训练完成")
