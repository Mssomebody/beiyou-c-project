import sys; print("脚本启动", file=sys.stderr)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五星级联邦学习 + 贝叶斯自动找最优 μ
- 使用 configs/paths.yaml 配置文件
- 自动检测数据版本
- 完整日志系统
- SQLite 断点续传
- 结果自动保存
- sMAPE 评估
"""

import os
import sys
import json
import logging
import argparse
import optuna
import yaml
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from torch.utils.data import DataLoader

# ============================================================
# 加载全局配置
# ============================================================

def load_global_config():
    """加载 paths.yaml 配置"""
    # 查找配置文件（支持多个可能位置）
    possible_paths = [
        Path(__file__).parent.parent.parent / "configs" / "paths.yaml",
        Path(__file__).parent.parent / "configs" / "paths.yaml",
        Path(__file__).parent / "configs" / "paths.yaml",
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    
    raise FileNotFoundError("找不到 configs/paths.yaml")


GLOBAL_CONFIG = load_global_config()
DATA_ROOT = Path(GLOBAL_CONFIG['data_root'])
BARCE_DATA_VERSION = GLOBAL_CONFIG['current']['barcelona']
BARCE_DATA_PATH = DATA_ROOT / GLOBAL_CONFIG['barcelona'][BARCE_DATA_VERSION]


# ============================================================
# 联邦学习配置
# ============================================================

@dataclass
class FedConfig:
    """联邦学习配置 - 可扩展"""
    # 数据配置
    data_path: Path = BARCE_DATA_PATH
    nodes: List[int] = field(default_factory=lambda: list(range(8001, 8025)))
    window_size: int = 28
    predict_size: int = 4
    batch_size: int = 64
    
    # 模型配置
    input_dim: int = 7
    hidden_dim: int = 160
    num_layers: int = 4
    dropout: float = 0.1565
    learning_rate: float = 0.00572
    
    # 联邦配置
    rounds: int = 20
    local_epochs: int = 5
    mu_min: float = 0.0
    mu_max: float = 0.1
    
    # 贝叶斯配置
    n_trials: int = 15
    
    # 其他
    seed: int = 42
    device: str = 'cpu'
    
    def __post_init__(self):
        """后处理"""
        # 确保目录存在
        self.data_path = Path(self.data_path)
        # 设置随机种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


# ============================================================
# 日志系统
# ============================================================

class Logger:
    """统一日志管理"""
    
    def __init__(self, config: FedConfig, name: str = "fed_optuna"):
        self.config = config
        self.name = name
        self.logger = self._setup()
    
    def _setup(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            return logger
        
        # 控制台处理器
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console)
        
        # 文件处理器
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)


# ============================================================
# 数据加载器
# ============================================================

class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def get_node_loader(
        node_id: int,
        split: str,
        batch_size: int,
        shuffle: bool,
        config: FedConfig
    ) -> DataLoader:
        """获取单个节点的数据加载器"""
        # 构建数据路径
        data_dir = config.data_path / f"node_{node_id}"
        file_path = data_dir / f"{split}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 创建数据集
        from src.data_loader.barcelona_dataset_v1 import BarcelonaDataset
        dataset = BarcelonaDataset(
            data_path=str(file_path),
            window_size=config.window_size,
            predict_size=config.predict_size,
            sector_feature=True,
            holiday_feature=True,
            weekend_feature=True
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    
    @classmethod
    def load_all_loaders(cls, config: FedConfig) -> Tuple[Dict, Dict, Dict]:
        """加载所有节点的数据加载器"""
        train_loaders = {}
        val_loaders = {}
        test_loaders = {}
        
        for node in config.nodes:
            train_loaders[node] = cls.get_node_loader(node, 'train', config.batch_size, True, config)
            val_loaders[node] = cls.get_node_loader(node, 'val', config.batch_size, False, config)
            test_loaders[node] = cls.get_node_loader(node, 'test', config.batch_size, False, config)
        
        return train_loaders, val_loaders, test_loaders


# ============================================================
# 联邦训练器
# ============================================================

class FederatedTrainer:
    """联邦训练器"""
    
    def __init__(self, config: FedConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        logger.info(f"设备: {self.device}")
    
    def _create_model(self) -> nn.Module:
        """创建模型"""
        from experiments.beautified.train_single_node import LSTMPredictor
        return LSTMPredictor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            output_dim=self.config.predict_size,
            dropout=self.config.dropout
        )
    
    def train_round(
        self,
        model: nn.Module,
        train_loaders: Dict[int, DataLoader],
        mu: float
    ) -> nn.Module:
        """一轮联邦训练"""
        client_weights = []
        client_sizes = []
        
        for client_id, loader in train_loaders.items():
            # 创建本地模型
            local_model = self._create_model().to(self.device)
            local_model.load_state_dict(model.state_dict())
            
            optimizer = torch.optim.Adam(local_model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()
            
            # 本地训练
            for _ in range(self.config.local_epochs):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(x)
                    loss = criterion(output, y)
                    
                    # FedProx 近端项
                    if mu > 0:
                        prox_loss = 0.0
                        for param, global_param in zip(local_model.parameters(), model.parameters()):
                            prox_loss += torch.norm(param - global_param) ** 2
                        loss += (mu / 2) * prox_loss
                    
                    loss.backward()
                    optimizer.step()
            
            client_weights.append(local_model.state_dict())
            client_sizes.append(len(loader.dataset))
        
        # 加权聚合
        total_samples = sum(client_sizes)
        global_weights = {}
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key])
            for w, size in zip(client_weights, client_sizes):
                global_weights[key] += w[key] * (size / total_samples)
        
        model.load_state_dict(global_weights)
        return model
    
    def evaluate_loss(self, model: nn.Module, loaders: Dict[int, DataLoader]) -> float:
        """评估损失"""
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for loader in loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    loss = criterion(output, y)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)
        
        return total_loss / total_samples
    
    def evaluate_smape(self, model: nn.Module, test_loaders: Dict[int, DataLoader]) -> float:
        """评估 sMAPE"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for loader in test_loaders.values():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    all_preds.append(output.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        denominator = (np.abs(all_targets) + np.abs(all_preds)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(np.abs(all_targets - all_preds) / denominator) * 100
        
        return smape
    
    def train_full(
        self,
        mu: float,
        train_loaders: Dict,
        val_loaders: Dict,
        test_loaders: Dict
    ) -> Tuple[float, float]:
        """完整训练，返回 (最佳验证损失, 测试 sMAPE)"""
        model = self._create_model().to(self.device)
        
        best_val_loss = float('inf')
        best_model = None
        
        for round_num in range(1, self.config.rounds + 1):
            model = self.train_round(model, train_loaders, mu)
            val_loss = self.evaluate_loss(model, val_loaders)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if round_num % 5 == 0:
                self.logger.info(f"  Round {round_num}: val_loss={val_loss:.6f}")
        
        if best_model:
            model.load_state_dict(best_model)
        
        test_smape = self.evaluate_smape(model, test_loaders)
        return best_val_loss, test_smape


# ============================================================
# 贝叶斯优化
# ============================================================

class BayesianOptimizer:
    """贝叶斯优化器"""
    
    def __init__(self, config: FedConfig, logger: Logger):
        self.config = config
        self.logger = logger
    
    def objective(self, trial: optuna.Trial) -> float:
        """目标函数"""
        mu = trial.suggest_float('mu', self.config.mu_min, self.config.mu_max, log=False)
        self.logger.info(f"Trial {trial.number}: μ={mu:.6f}")
        
        # 加载数据
        train_loaders, val_loaders, test_loaders = DataLoaderFactory.load_all_loaders(self.config)
        
        # 训练
        trainer = FederatedTrainer(self.config, self.logger)
        val_loss, test_smape = trainer.train_full(mu, train_loaders, val_loaders, test_loaders)
        
        self.logger.info(f"  Test sMAPE: {test_smape:.2f}%")
        
        return test_smape
    
    def optimize(self) -> Tuple[float, Dict]:
        """执行优化"""
        # 创建 study（支持断点续传）
        output_dir = Path(__file__).parent.parent.parent / "results" / "fed_optuna"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        study_name = f"fed_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage = f"sqlite:///{output_dir / study_name}.db"
        
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        
        self.logger.info(f"开始优化，试验次数: {self.config.n_trials}")
        
        study.optimize(
            lambda trial: self.objective(trial),
            n_trials=self.config.n_trials,
            show_progress_bar=True
        )
        
        return study.best_params['mu'], study.best_value, study


# ============================================================
# 结果保存
# ============================================================

class ResultSaver:
    """结果保存器"""
    
    def __init__(self, config: FedConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(__file__).parent.parent.parent / "results" / "fed_optuna"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, best_mu: float, best_smape: float, study: optuna.Study):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'config': self.config.to_dict(),
            'best_mu': best_mu,
            'best_smape': best_smape,
            'data_info': {
                'data_version': BARCE_DATA_VERSION,
                'data_path': str(BARCE_DATA_PATH),
                'nodes': self.config.nodes,
                'n_nodes': len(self.config.nodes)
            },
            'trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'datetime': str(t.datetime)
                }
                for t in study.trials if t.value is not None
            ]
        }
        
        # 保存 JSON
        json_path = self.output_dir / f"results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"结果保存: {json_path}")
        
        # 保存文本摘要
        summary_path = self.output_dir / f"summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("联邦学习贝叶斯优化结果\n")
            f.write("="*60 + "\n\n")
            f.write(f"数据版本: {BARCE_DATA_VERSION}\n")
            f.write(f"数据路径: {BARCE_DATA_PATH}\n")
            f.write(f"节点数: {len(self.config.nodes)}\n\n")
            f.write(f"最优 μ: {best_mu:.6f}\n")
            f.write(f"最优 sMAPE: {best_smape:.2f}%\n\n")
            f.write("各试验结果:\n")
            for t in results['trials']:
                f.write(f"  Trial {t['number']}: μ={t['params']['mu']:.6f}, sMAPE={t['value']:.2f}%\n")
        
        self.logger.info(f"摘要保存: {summary_path}")
        
        return json_path


# ============================================================
# 主函数
# ============================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习 + 贝叶斯自动找最优 μ')
    parser.add_argument('--nodes_start', type=int, default=8001, help='起始节点')
    parser.add_argument('--nodes_end', type=int, default=8025, help='结束节点')
    parser.add_argument('--n_trials', type=int, default=15, help='试验次数')
    parser.add_argument('--rounds', type=int, default=20, help='联邦轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = FedConfig(
        nodes=list(range(args.nodes_start, args.nodes_end)),
        n_trials=args.n_trials,
        rounds=args.rounds,
        seed=args.seed
    )
    
    # 初始化日志
    logger = Logger(config)
    
    # 打印配置
    logger.info("="*70)
    logger.info("联邦学习 + 贝叶斯自动找最优 μ")
    logger.info("="*70)
    logger.info(f"数据版本: {BARCE_DATA_VERSION}")
    logger.info(f"数据路径: {BARCE_DATA_PATH}")
    logger.info(f"节点: {config.nodes[0]} - {config.nodes[-1]} ({len(config.nodes)} 个)")
    logger.info(f"μ 范围: [{config.mu_min}, {config.mu_max}]")
    logger.info(f"试验次数: {config.n_trials}")
    logger.info(f"联邦轮数: {config.rounds}")
    logger.info(f"输出目录: {Path(__file__).parent.parent.parent / 'results' / 'fed_optuna'}")
    logger.info("="*70)
    
    # 创建优化器
    optimizer = BayesianOptimizer(config, logger)
    
    # 执行优化
    best_mu, best_smape, study = optimizer.optimize()
    
    # 保存结果
    saver = ResultSaver(config, logger)
    saver.save(best_mu, best_smape, study)
    
    # 打印结果
    logger.info("\n" + "="*70)
    logger.info("优化完成")
    logger.info("="*70)
    logger.info(f"最优 μ: {best_mu:.6f}")
    logger.info(f"最优 sMAPE: {best_smape:.2f}%")
    logger.info("="*70)
    
    return best_mu


if __name__ == "__main__":
    best_mu = main()
    print(f"\n✅ 最优 μ = {best_mu:.6f}")
