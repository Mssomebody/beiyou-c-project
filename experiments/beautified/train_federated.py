import torch
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Federated Training Script with FedProx
Supports: config file (YAML/JSON) OR command-line arguments
Auto logging to experiments/logs/
"""

import sys
import os
import logging
import json
import yaml
import pickle
import argparse
from datetime import datetime

# ============================================================
# 配置日志（自动保存到 experiments/logs/）
# ============================================================
def setup_logging():
    """配置日志，自动保存到 experiments/logs/"""
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(base_dir, "experiments", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_federated_{timestamp}.log")
    
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"日志文件: {log_file}")
    return log_file

# 初始化日志
LOG_FILE = setup_logging()

# 重定向 print 到 logging（可选）
import builtins
original_print = builtins.print
def print_with_log(*args, **kwargs):
    original_print(*args, **kwargs)
    logging.info(' '.join(str(arg) for arg in args))
builtins.print = print_with_log


# ============================================================
# 导入模块
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.barcelona_dataset_v1 import get_node_data_loader
from src.federated.fedprox_client import create_clients
from src.federated.fedprox_server import FedProxServer
from experiments.beautified.train_single_node import LSTMPredictor, plot_loss_curve


def load_config(config_path):
    """Load configuration from YAML or JSON file"""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")


def load_node_data(node_ids, batch_size=64, window_size=28, predict_size=4):
    """Load data for multiple nodes"""
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    scalers = {}
    
    for node_id in node_ids:
        train_loader, scaler_path, _ = get_node_data_loader(
            node_id=node_id, split='train', batch_size=batch_size, shuffle=True,
            window_size=window_size, predict_size=predict_size,
            sector_feature=True, holiday_feature=True, weekend_feature=True
        )
        val_loader, _, _ = get_node_data_loader(
            node_id=node_id, split='val', batch_size=batch_size, shuffle=False,
            window_size=window_size, predict_size=predict_size,
            sector_feature=True, holiday_feature=True, weekend_feature=True
        )
        test_loader, _, _ = get_node_data_loader(
            node_id=node_id, split='test', batch_size=batch_size, shuffle=False,
            window_size=window_size, predict_size=predict_size,
            sector_feature=True, holiday_feature=True, weekend_feature=True
        )
        
        train_loaders[node_id] = train_loader
        val_loaders[node_id] = val_loader
        test_loaders[node_id] = test_loader
        
        with open(scaler_path, 'rb') as f:
            scalers[node_id] = pickle.load(f)
        
        print(f"Node {node_id}: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    
    return train_loaders, val_loaders, test_loaders, scalers


def train_federated(config):
    """
    Federated training with FedProx using config dict
    """
    # Extract config
    node_ids = config['data']['nodes']
    window_size = config['data']['window_size']
    predict_size = config['data']['predict_size']
    batch_size = config['data']['batch_size']
    
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    lr = config['model']['lr']
    dropout = config['model'].get('dropout', 0.2)
    
    rounds = config['federated']['rounds']
    local_epochs = config['federated']['local_epochs']
    mu_values = config['federated']['mu_values']
    
    save_dir = config['output'].get('save_dir', 'results/beautified')
    save_plots = config['output'].get('save_plots', True)
    show_plots = config['output'].get('show_plots', True)
    
    # Create save directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(base_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("FedGreen-C Federated Learning Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Nodes: {node_ids} ({len(node_ids)} nodes)")
    print(f"  Rounds: {rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Mu values: {mu_values}")
    print(f"  Window: {window_size} → {predict_size}")
    print(f"  Device: {device}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_loaders, val_loaders, test_loaders, scalers = load_node_data(
        node_ids, batch_size, window_size, predict_size
    )
    
    # Get input dimension
    sample_x, _ = next(iter(train_loaders[node_ids[0]]))
    input_dim = sample_x.shape[-1]
    print(f"Input dimension: {input_dim}")
    
    # Model parameters
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'output_dim': predict_size,
        'dropout': dropout
    }
    
    # Train for each mu value
    all_results = []
    
    for mu in mu_values:
        print(f"\n{'#'*60}")
        print(f"Training with mu = {mu}")
        print(f"{'#'*60}")
        
        # Create global model
        global_model = LSTMPredictor(**model_params).to(device)
        
        # Create clients
        clients = create_clients(
            node_ids, train_loaders, val_loaders, test_loaders,
            LSTMPredictor, model_params, device, mu
        )
        
        # Create server
        server = FedProxServer(global_model, clients, device, aggregation='weighted')
        
        # Federated training
        print("\nStarting federated training...")
        round_losses = []
        
        for round_num in range(1, rounds + 1):
            print(f"\n  Round {round_num}/{rounds}")
            round_loss, client_losses = server.federated_round(local_epochs, lr)
            round_losses.append(round_loss)
            print(f"    Avg Loss = {round_loss:.6f}")
            
            if round_num % 5 == 0:
                test_loss = server.evaluate_global(test_loaders)
                print(f"    Test Loss = {test_loss:.6f}")
        
        # Final evaluation
        test_loss = server.evaluate_global(test_loaders)
        print(f"\nFinal Test Loss: {test_loss:.6f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'node_ids': node_ids,
            'num_nodes': len(node_ids),
            'rounds': rounds,
            'local_epochs': local_epochs,
            'mu': mu,
            'batch_size': batch_size,
            'window_size': window_size,
            'predict_size': predict_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'lr': lr,
            'round_losses': round_losses,
            'final_test_loss': test_loss,
            'timestamp': timestamp
        }
        
        # Save loss curve
        if save_plots:
            loss_plot_path = os.path.join(
                save_dir, 
                f"federated_nodes{len(node_ids)}_rounds{rounds}_mu{mu}_loss_{timestamp}.png"
            )
            plot_loss_curve(round_losses, None, 
                           f"Federated (nodes={len(node_ids)}, mu={mu}, rounds={rounds})", 
                           loss_plot_path if save_plots else None)
            if not show_plots:
                import matplotlib.pyplot as plt
                plt.close()
        
        # Save results
        results_path = os.path.join(
            save_dir,
            f"federated_nodes{len(node_ids)}_rounds{rounds}_mu{mu}_results_{timestamp}.pkl"
        )
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save model
        model_path = os.path.join(
            save_dir,
            f"federated_nodes{len(node_ids)}_rounds{rounds}_mu{mu}_model_{timestamp}.pth"
        )
        server.save_checkpoint(model_path)
        
        all_results.append(results)
        print(f"✅ Results saved for mu={mu}")
    
    # Plot comparison if multiple mu values
    if len(mu_values) > 1 and save_plots:
        print("\n" + "=" * 60)
        print("Generating comparison plot...")
        print("=" * 60)
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        for results in all_results:
            mu = results['mu']
            losses = results['round_losses']
            plt.plot(range(1, len(losses) + 1), losses, 
                     label=f"mu = {mu}", linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Communication Round', fontsize=12, fontweight='bold')
        plt.ylabel('Average Loss (MSE)', fontsize=12, fontweight='bold')
        plt.title(f'Federated Learning: FedProx vs FedAvg\n({len(node_ids)} nodes, {rounds} rounds)', 
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(save_dir, f"federated_comparison_{timestamp}.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved: {comparison_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    print("\n" + "=" * 60)
    print("All training complete!")
    print("=" * 60)
    print("\nSummary:")
    for results in all_results:
        print(f"  mu = {results['mu']}: Final Test Loss = {results['final_test_loss']:.6f}")
    
    logging.info("Training completed successfully")
    
    return all_results


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedGreen-C Federated Learning Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (YAML or JSON)')
    parser.add_argument('--generate_config', action='store_true',
                        help='Generate example config file')
    
    # 如果没有配置文件，也可以使用命令行参数（向后兼容）
    parser.add_argument('--nodes', type=str, default=None,
                        help='Node IDs, comma separated (overrides config)')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Number of rounds (overrides config)')
    parser.add_argument('--mu', type=str, default=None,
                        help='Mu values, comma separated (overrides config)')
    
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("FedGreen-C Federated Learning Training Started")
    logging.info("=" * 60)
    
    # Generate example config file
    if args.generate_config:
        example_config = {
            'data': {
                'nodes': [8001, 8002, 8003],
                'window_size': 28,
                'predict_size': 4,
                'batch_size': 64
            },
            'model': {
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'lr': 0.001
            },
            'federated': {
                'rounds': 10,
                'local_epochs': 5,
                'mu_values': [0.0, 0.01]
            },
            'output': {
                'save_dir': 'results/beautified',
                'save_plots': True,
                'show_plots': True
            }
        }
        
        # 保存 YAML
        os.makedirs('configs', exist_ok=True)
        with open('configs/federated_config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(example_config, f, default_flow_style=False, allow_unicode=True)
        logging.info("✅ Example config saved: configs/federated_config.yaml")
        
        # 保存 JSON
        with open('configs/federated_config.json', 'w', encoding='utf-8') as f:
            json.dump(example_config, f, indent=2, ensure_ascii=False)
        logging.info("✅ Example config saved: configs/federated_config.json")
        
        sys.exit(0)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logging.info(f"Loaded config from: {args.config}")
    else:
        # 默认配置文件路径
        default_config_path = 'configs/federated_config.yaml'
        if os.path.exists(default_config_path):
            config = load_config(default_config_path)
            logging.info(f"Loaded default config from: {default_config_path}")
        else:
            logging.warning("No config file found. Using default values.")
            config = {
                'data': {'nodes': [8001, 8002, 8003], 'window_size': 28, 'predict_size': 4, 'batch_size': 64},
                'model': {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001},
                'federated': {'rounds': 10, 'local_epochs': 5, 'mu_values': [0.0, 0.01]},
                'output': {'save_dir': 'results/beautified', 'save_plots': True, 'show_plots': True}
            }
    
    # 命令行参数覆盖
    if args.nodes:
        config['data']['nodes'] = [int(x.strip()) for x in args.nodes.split(',')]
        logging.info(f"Override nodes: {config['data']['nodes']}")
    if args.rounds:
        config['federated']['rounds'] = args.rounds
        logging.info(f"Override rounds: {config['federated']['rounds']}")
    if args.mu:
        config['federated']['mu_values'] = [float(x.strip()) for x in args.mu.split(',')]
        logging.info(f"Override mu: {config['federated']['mu_values']}")
    
    # Run training
    try:
        results = train_federated(config)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise