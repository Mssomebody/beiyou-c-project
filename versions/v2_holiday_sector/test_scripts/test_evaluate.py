import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn

# 模型定义（与训练脚本一致）
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 使用一天窗口预训练模型（已存在）
model_path = PROJECT_ROOT / "results" / "two_stage" / "model_fed_pretrain_1day.pth"
if not model_path.exists():
    print(f"模型文件不存在: {model_path}")
    sys.exit(1)

# 加载 MinMax 参数
minmax_path = PROJECT_ROOT / "versions" / "v2_holiday_sector" / "node_minmax.pkl"
with open(minmax_path, 'rb') as f:
    node_minmax = pickle.load(f)

# 创建模型并加载权重
device = torch.device('cpu')
model = LSTMPredictor(input_dim=7, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 测试节点 8001 的测试集（一天窗口，window_size=4）
from test_dataset import MinMaxBarcelonaDataset  # 复用之前的数据集类

node_id = 8001
test_file = PROJECT_ROOT / "data" / "processed" / "barcelona_ready_v1" / f"node_{node_id}" / "test.pkl"
dataset = MinMaxBarcelonaDataset(test_file, node_id, node_minmax, window_size=4, predict_size=4)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

data_min, data_max = node_minmax[node_id]
all_preds_real = []
all_targets_real = []
with torch.no_grad():
    for x, y in loader:
        pred_norm = model(x).cpu().numpy()
        target_norm = y.cpu().numpy()
        pred_real = pred_norm * (data_max - data_min) + data_min
        target_real = target_norm * (data_max - data_min) + data_min
        all_preds_real.append(pred_real)
        all_targets_real.append(target_real)

all_preds_real = np.concatenate(all_preds_real)
all_targets_real = np.concatenate(all_targets_real)
denominator = (np.abs(all_targets_real) + np.abs(all_preds_real)) / 2
denominator = np.where(denominator == 0, 1e-8, denominator)
smape = np.mean(np.abs(all_targets_real - all_preds_real) / denominator) * 100
print(f"节点 {node_id} 真实 sMAPE: {smape:.2f}%")
