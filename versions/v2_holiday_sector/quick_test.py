import sys
sys.path.insert(0, '.')
from train_federated_pretrain import FederatedTrainer, load_node_loaders, MinMaxBarcelonaDataset
import pickle
from pathlib import Path
import torch

# 只取2个节点
node_ids = [8001, 8002]
data_dir = Path('data/processed/barcelona_ready_v1')
with open('versions/v2_holiday_sector/node_minmax.pkl', 'rb') as f:
    node_minmax = pickle.load(f)

# 自定义加载函数，只取每个数据集的前200个样本
def load_small_loaders(node_ids, data_dir, node_minmax, split, batch_size=8):
    loaders = {}
    for node_id in node_ids:
        pkl_file = data_dir / f'node_{node_id}' / f'{split}.pkl'
        if not pkl_file.exists():
            continue
        full_ds = MinMaxBarcelonaDataset(pkl_file, node_id, node_minmax)
        # 切片取前200个样本
        small_ds = torch.utils.data.Subset(full_ds, range(min(200, len(full_ds))))
        loader = torch.utils.data.DataLoader(small_ds, batch_size=batch_size, shuffle=(split=='train'))
        loaders[node_id] = loader
    return loaders

train_loaders = load_small_loaders(node_ids, data_dir, node_minmax, 'train')
val_loaders = load_small_loaders(node_ids, data_dir, node_minmax, 'val')

class Args:
    device = 'cpu'
    lr = 0.002
    local_epochs = 2
    rounds = 3
    output_model = 'quick_test.pth'
args = Args()
trainer = FederatedTrainer(args)
model, best_smape = trainer.train(train_loaders, val_loaders, node_minmax, rounds=3, mu=0.01)
print(f"快速测试完成，最佳验证sMAPE: {best_smape:.2f}%")
