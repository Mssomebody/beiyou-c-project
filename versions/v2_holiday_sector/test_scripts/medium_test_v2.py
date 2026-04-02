import sys
sys.path.insert(0, '.')
from train_federated_pretrain import FederatedTrainer, load_node_loaders, MinMaxBarcelonaDataset
import pickle
from pathlib import Path

# 只取5个节点
node_ids = [8001, 8002, 8004, 8006, 8012]
data_dir = Path('data/processed/barcelona_ready_v1')
with open('versions/v2_holiday_sector/node_minmax.pkl', 'rb') as f:
    node_minmax = pickle.load(f)

train_loaders = load_node_loaders(node_ids, data_dir, node_minmax, 'train', batch_size=64, shuffle=True)
val_loaders = load_node_loaders(node_ids, data_dir, node_minmax, 'val', batch_size=64, shuffle=False)

class Args:
    device = 'cpu'
    lr = 0.001
    local_epochs = 15
    rounds = 20
    output_model = 'medium_test_v2_model.pth'
args = Args()
trainer = FederatedTrainer(args)
model, best_smape = trainer.train(train_loaders, val_loaders, node_minmax, rounds=20, mu=0.005)
print(f"测试完成，最佳验证sMAPE: {best_smape:.2f}%")
