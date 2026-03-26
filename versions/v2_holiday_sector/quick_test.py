"""
快速测试分口径数据（用Trial 3的最佳参数）
"""
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor, train_epoch, evaluate

# 修改这里选择口径
DATA_PATH = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"
# DATA_PATH = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"

NODE_ID = 8001
BATCH_SIZE = 64
HIDDEN = 160
LAYERS = 4
LR = 0.00572
DROPOUT = 0.16
EPOCHS = 20

def get_loaders(node_id, batch_size, data_path):
    # 尝试两种节点名格式
    data_dir1 = os.path.join(data_path, f"node_{node_id}")
    data_dir2 = os.path.join(data_path, f"node_{node_id}.0")
    
    if os.path.exists(data_dir1):
        data_dir = data_dir1
    elif os.path.exists(data_dir2):
        data_dir = data_dir2
    else:
        raise FileNotFoundError(f"找不到节点目录: {data_dir1} 或 {data_dir2}")
    
    train_dataset = BarcelonaDataset(os.path.join(data_dir, "train.pkl"))
    val_dataset = BarcelonaDataset(os.path.join(data_dir, "val.pkl"))
    test_dataset = BarcelonaDataset(os.path.join(data_dir, "test.pkl"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader

print("="*50)
print(f"数据路径: {DATA_PATH}")
print("="*50)

train_loader, val_loader, test_loader = get_loaders(NODE_ID, BATCH_SIZE, DATA_PATH)

# 获取输入维度
sample_x, _ = train_loader.dataset[0]
input_dim = sample_x.shape[1]
print(f"输入维度: {input_dim}")

model = LSTMPredictor(
    input_dim=input_dim,
    hidden_dim=HIDDEN,
    num_layers=LAYERS,
    output_dim=4,
    dropout=DROPOUT
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"\n开始训练 {EPOCHS} 轮...")
print("-"*50)

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

test_loss = evaluate(model, test_loader, criterion, device)
print("-"*50)
print(f"测试损失: {test_loss:.6f}")
