import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
from experiments.beautified.train_single_node import LSTMPredictor, train_epoch, evaluate

def get_loaders(node_id, batch_size, data_path):
    data_dir = os.path.join(data_path, f"node_{node_id}")
    train_dataset = BarcelonaDataset(os.path.join(data_dir, "train.pkl"))
    val_dataset = BarcelonaDataset(os.path.join(data_dir, "val.pkl"))
    test_dataset = BarcelonaDataset(os.path.join(data_dir, "test.pkl"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader

def compute_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

def evaluate_with_smape(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    smape = compute_smape(all_targets, all_preds)
    return total_loss / len(dataloader), smape

HIDDEN = 160
LAYERS = 4
LR = 0.00572
DROPOUT = 0.16
BATCH_SIZE = 64
EPOCHS = 30

print("="*50)
print("测试旧口径 (2019-2022)")
print("="*50)

data_path_old = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2019_2022"
train_loader, val_loader, test_loader = get_loaders(8001, BATCH_SIZE, data_path_old)

sample_x, _ = train_loader.dataset[0]
input_dim = sample_x.shape[1]

model = LSTMPredictor(input_dim=input_dim, hidden_dim=HIDDEN, num_layers=LAYERS, output_dim=4, dropout=DROPOUT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_smape = evaluate_with_smape(model, val_loader, criterion, device)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_smape={val_smape:.2f}%")

test_loss, test_smape = evaluate_with_smape(model, test_loader, criterion, device)
print(f"\n旧口径: 测试损失={test_loss:.6f}, 测试sMAPE={test_smape:.2f}%")

print("\n" + "="*50)
print("测试新口径 (2023-2025)")
print("="*50)

data_path_new = r"D:\Desk\desk\beiyou_c_project\data\processed\barcelona_ready_2023_2025"
train_loader, val_loader, test_loader = get_loaders(8001, BATCH_SIZE, data_path_new)

model = LSTMPredictor(input_dim=input_dim, hidden_dim=HIDDEN, num_layers=LAYERS, output_dim=4, dropout=DROPOUT)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_smape = evaluate_with_smape(model, val_loader, criterion, device)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_smape={val_smape:.2f}%")

test_loss, test_smape = evaluate_with_smape(model, test_loader, criterion, device)
print(f"\n新口径: 测试损失={test_loss:.6f}, 测试sMAPE={test_smape:.2f}%")