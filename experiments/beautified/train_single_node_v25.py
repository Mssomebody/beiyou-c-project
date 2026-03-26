#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single Node LSTM Training Script - Professional Edition
Supports both v1 and v2.5 features via --use_v25 flag
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import sys
import pickle
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.barcelona_dataset import get_node_data_loader


# ============================================================
# Unified Plot Style Configuration (Professional)
# ============================================================

# Set seaborn style
sns.set_theme(style="darkgrid", context="notebook")

# Color scheme (FedGreen-C theme: green + blue)
COLORS = {
    'primary': '#2E8B57',      # Sea Green
    'secondary': '#2E86AB',    # Ocean Blue
    'accent': '#F18F01',       # Warm Orange
    'warning': '#E76F51',      # Coral
    'success': '#6AAB9E',      # Success Green
    'background': '#F5F5F5',   # Light Gray
    'text': '#2C3E50',         # Dark Gray
    'grid': '#D3D3D3'          # Grid Gray
}

# Set color palette
sns.set_palette([COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning']])

# Matplotlib configuration
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'white',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.color': COLORS['grid'],
    'savefig.format': 'png'
})


# ============================================================
# Model Definition
# ============================================================
class LSTMPredictor(nn.Module):
    """
    LSTM Energy Prediction Model
    Input: [batch, window_size, features]
    Output: [batch, predict_size]
    """
    
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model (normalized scale)"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)
    
    return total_loss / len(dataloader.dataset)


def inverse_normalize(predictions, scaler):
    """Inverse normalize predictions back to original scale"""
    orig_shape = predictions.shape
    pred_flat = predictions.reshape(-1, 1)
    pred_orig = scaler.inverse_transform(pred_flat)
    return pred_orig.reshape(orig_shape)


def calculate_smape(y_true, y_pred):
    """Calculate sMAPE (Symmetric Mean Absolute Percentage Error)"""
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 0
    if mask.sum() > 0:
        return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
    return float('inf')


def calculate_mape_filtered(y_true, y_pred, percentile=5):
    """Calculate filtered MAPE (exclude bottom percentile)"""
    threshold = np.percentile(y_true, percentile)
    mask = y_true > threshold
    if mask.sum() > 0:
        return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100, threshold
    return float('inf'), threshold


def evaluate_original_scale(model, dataloader, scaler, device, percentile=5):
    """Evaluate model on original scale"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Inverse normalize
    all_preds_orig = inverse_normalize(all_preds, scaler)
    all_targets_orig = inverse_normalize(all_targets, scaler)
    
    # 1. RMSE
    rmse = np.sqrt(np.mean((all_preds_orig - all_targets_orig) ** 2))
    
    # 2. MAE
    mae = np.mean(np.abs(all_preds_orig - all_targets_orig))
    
    # 3. sMAPE
    smape = calculate_smape(all_targets_orig.flatten(), all_preds_orig.flatten())
    
    # 4. MAPE with percentile filtering
    mape_filtered, threshold = calculate_mape_filtered(
        all_targets_orig.flatten(), 
        all_preds_orig.flatten(), 
        percentile
    )
    
    return {
        'rmse': rmse,
        'mae': mae,
        'smape': smape,
        'mape_filtered': mape_filtered,
        'mape_threshold': threshold,
        'predictions': all_preds_orig,
        'targets': all_targets_orig,
        'percentile': percentile
    }


def plot_predictions(all_preds, all_targets, node_id, save_path=None):
    """Plot prediction comparison (English labels)"""
    n_samples = min(200, len(all_preds))
    n_steps = 4
    
    global_max = max(
        all_targets[:n_samples].max(),
        all_preds[:n_samples].max()
    ) * 1.05
    
    step_names = [
        'Step 1: Next 0-6 hours',
        'Step 2: Next 6-12 hours', 
        'Step 3: Next 12-18 hours',
        'Step 4: Next 18-24 hours'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    for i, step in enumerate(range(n_steps)):
        ax = axes[i // 2, i % 2]
        step_preds = all_preds[:n_samples, step]
        step_targets = all_targets[:n_samples, step]
        
        ax.plot(step_targets, label='Actual', color=COLORS['primary'], linewidth=1.8)
        ax.plot(step_preds, label='Predicted', color=COLORS['secondary'], linewidth=1.5, linestyle='--', alpha=0.9)
        
        ax.set_title(step_names[i], fontsize=12, fontweight='bold', color=COLORS['text'])
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Energy (kWh)', fontsize=10)
        ax.set_ylim(0, global_max)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    fig.suptitle(f'Node {node_id} - LSTM Energy Prediction Results\n(Unified Y-axis: 0-{global_max:.0f} kWh)', 
                 fontsize=14, fontweight='bold', color=COLORS['primary'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Prediction plot saved: {save_path}")
    
    plt.show()


def plot_loss_curve(train_losses, val_losses, node_id, save_path=None):
    """Plot loss curve (English labels)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='Train Loss', color=COLORS['primary'], linewidth=2, marker='o', markersize=4)
    if val_losses:
        ax.plot(epochs, val_losses, label='Validation Loss', color=COLORS['secondary'], linewidth=2, marker='s', markersize=4)
    
    if isinstance(node_id, str):
        title = node_id
    else:
        title = f'Node {node_id} - Training Curve'
    
    ax.set_xlabel('Epoch' if val_losses else 'Communication Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Loss curve saved: {save_path}")
    
    plt.show()


# ============================================================
# Main Training Function
# ============================================================
def train_single_node(node_id=8001, epochs=20, batch_size=64, 
                      window_size=28, predict_size=4,
                      hidden_dim=64, num_layers=2, lr=0.001,
                      dropout=0.2, percentile=5, success_threshold=30,
                      sector_feature=True, holiday_feature=True, weekend_feature=True,
                      use_v25=False):
    """
    Train single node (supports both v1 and v2.5 features)
    
    Args:
        use_v25: Use v2.5 features (lag, rolling, interaction) with feature selection
    """
    print("=" * 60)
    print(f"Single Node Training: Node {node_id}")
    if use_v25:
        print("Mode: v2.5 (精选特征)")
    else:
        print("Mode: v1 (原版特征)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create save directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(base_dir, "results", "beautified")
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ============================================================
    # Load data (v1 or v2.5)
    # ============================================================
    if use_v25:
        # v2.5: 使用精选特征
        print("\nLoading data with v2.5 features...")
        train_loader, scaler_path, train_dataset = get_node_data_loader(
            node_id=node_id,
            split='train',
            batch_size=batch_size,
            shuffle=True,
            window_size=window_size,
            predict_size=predict_size,
            sector_feature=sector_feature
        )
        val_loader, _, _ = get_node_data_loader(
            node_id=node_id,
            split='val',
            batch_size=batch_size,
            shuffle=False,
            window_size=window_size,
            predict_size=predict_size,
            sector_feature=sector_feature
        )
        test_loader, _, _ = get_node_data_loader(
            node_id=node_id,
            split='test',
            batch_size=batch_size,
            shuffle=False,
            window_size=window_size,
            predict_size=predict_size,
            sector_feature=sector_feature
        )
    else:
        # v1: 原版特征
        print("\nLoading data with v1 features...")
        train_loader, scaler_path, train_dataset = get_node_data_loader(
            node_id=node_id,
            split='train',
            batch_size=batch_size,
            shuffle=True,
            window_size=window_size,
            predict_size=predict_size,
            sector_feature=sector_feature,
        )
        val_loader, _, _ = get_node_data_loader(
            node_id=node_id,
            split='val',
            batch_size=batch_size,
            shuffle=False,
            window_size=window_size,
            predict_size=predict_size,
            sector_feature=sector_feature,
        )
        test_loader, _, _ = get_node_data_loader(
            node_id=node_id,
            split='test',
            batch_size=batch_size,
            shuffle=False,
            window_size=window_size,
            predict_size=predict_size,
            sector_feature=sector_feature,
        )
    
    # Load scaler (valor scaler for inverse normalization)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Get input dimension
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]
    
    print(f"Input dimension: {input_dim}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=predict_size,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # Normalized scale evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nNormalized Scale:")
    print(f"  Test Loss (MSE): {test_loss:.6f}")
    
    # Original scale evaluation
    eval_results = evaluate_original_scale(
        model, test_loader, scaler, device, percentile
    )
    
    print(f"\nOriginal Scale (kWh):")
    print(f"  RMSE: {eval_results['rmse']:.2f} kWh")
    print(f"  MAE:  {eval_results['mae']:.2f} kWh")
    print(f"  sMAPE: {eval_results['smape']:.2f}%")
    print(f"  MAPE (filtered < {eval_results['mape_threshold']:.0f} kWh, i.e., {percentile}th percentile): {eval_results['mape_filtered']:.2f}%")
    
    # Success check
    if eval_results['mape_filtered'] < success_threshold:
        print(f"\n✅ Training successful! MAPE {eval_results['mape_filtered']:.2f}% < {success_threshold}%")
    else:
        print(f"\n⚠️ MAPE {eval_results['mape_filtered']:.2f}% > {success_threshold}%, consider tuning hyperparameters")
    
    # Plot prediction comparison
    pred_plot_path = os.path.join(save_dir, f"node_{node_id}_predictions_{timestamp}.png")
    plot_predictions(eval_results['predictions'], eval_results['targets'], node_id, pred_plot_path)
    
    # Plot loss curve
    loss_plot_path = os.path.join(save_dir, f"node_{node_id}_loss_curve_{timestamp}.png")
    plot_loss_curve(train_losses, val_losses, node_id, loss_plot_path)
    
    # Save results
    results = {
        'version': 'v2.5' if use_v25 else 'v1',
        'node_id': node_id,
        'timestamp': timestamp,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'lr': lr,
        'dropout': dropout,
        'batch_size': batch_size,
        'percentile': percentile,
        'rmse': eval_results['rmse'],
        'mae': eval_results['mae'],
        'smape': eval_results['smape'],
        'mape_filtered': eval_results['mape_filtered'],
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    results_path = os.path.join(save_dir, f"node_{node_id}_results_{timestamp}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ Results saved: {results_path}")
    
    return model, results


# ============================================================
# Main Entry
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FedGreen-C Single Node LSTM Training")
    print("=" * 60)
    print("Evaluation Metrics:")
    print("  - RMSE: Root Mean Square Error (sensitive to outliers)")
    print("  - MAE:  Mean Absolute Error (intuitive)")
    print("  - sMAPE: Symmetric Mean Absolute Percentage Error (avoids denominator issues)")
    print("  - MAPE: Traditional MAPE (excludes bottom 5th percentile samples)")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=int, default=8001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--percentile', type=int, default=5)
    parser.add_argument('--success_threshold', type=float, default=30)
    parser.add_argument('--no_sector', action='store_true', help='Disable sector feature')
    parser.add_argument('--no_holiday', action='store_true', help='Disable holiday feature')
    parser.add_argument('--no_weekend', action='store_true', help='Disable weekend feature')
    parser.add_argument('--use_v25', action='store_true', help='Use v2.5 features (精选特征)')
    
    args = parser.parse_args()
    
    # Train
    model, results = train_single_node(
        node_id=args.node,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        dropout=args.dropout,
        percentile=args.percentile,
        success_threshold=args.success_threshold,
        sector_feature=not args.no_sector,
        holiday_feature=not args.no_holiday,
        weekend_feature=not args.no_weekend,
        use_v25=args.use_v25
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Metrics:")
    print(f"  RMSE: {results['rmse']:.2f} kWh")
    print(f"  MAE:  {results['mae']:.2f} kWh")
    print(f"  sMAPE: {results['smape']:.2f}%")
    print(f"  MAPE (filtered <{results.get('mape_threshold', 0):.0f} kWh): {results['mape_filtered']:.2f}%")
    print("=" * 60)