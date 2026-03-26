import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import pickle
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data" / "processed" / "tsinghua_v2"
        self.output_dir = self.project_root / "results" / "shap_analysis"
        
        self.data_type = '4g'
        self.max_stations = 50
        self.samples_per_station = 100
        self.seq_len = 24
        self.input_dim = 5
        self.hidden_dim = 64
        self.num_layers = 2
        self.batch_size = 64
        self.epochs = 20
        self.lr = 0.001
        
        self.shap_background = 50
        self.shap_test = 30
        
        self.feature_names = ['PRB', 'Traffic', 'Users', 'Hour_sin', 'Hour_cos']
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(42)
        tf.random.set_seed(42)


def build_model(input_dim=5, hidden_dim=64, num_layers=2):
    inputs = tf.keras.Input(shape=(24, input_dim))
    x = inputs
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = tf.keras.layers.LSTM(hidden_dim, return_sequences=return_sequences)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def load_data(config):
    data_dir = config.data_dir / config.data_type
    station_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('station_')]
    station_dirs = station_dirs[:config.max_stations]
    
    all_features = []
    for s_dir in tqdm(station_dirs, desc=f"加载 {config.data_type} 数据"):
        with open(s_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        features = data.get('X_train_norm', data.get('features_norm', None))
        if features is not None:
            if config.samples_per_station:
                features = features[:config.samples_per_station]
            all_features.append(features)
    
    X = np.concatenate(all_features, axis=0)
    
    X_seq = []
    for i in range(len(X) - config.seq_len):
        X_seq.append(X[i:i+config.seq_len])
    X_seq = np.array(X_seq)
    
    y = X_seq[:, -1, 0:1]
    
    return X_seq, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='4g', choices=['4g', '5g'])
    parser.add_argument('--max_stations', type=int, default=50)
    parser.add_argument('--samples_per_station', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    config = Config()
    config.data_type = args.data_type
    config.max_stations = args.max_stations
    config.samples_per_station = args.samples_per_station
    config.epochs = args.epochs
    
    logger.info("="*60)
    logger.info(f"SHAP DeepExplainer 分析 - {config.data_type.upper()} (TF静态图模式)")
    logger.info("="*60)
    
    X_seq, y = load_data(config)
    logger.info(f"序列样本数: {len(X_seq):,}")
    logger.info(f"序列形状: {X_seq.shape}")
    
    model = build_model(config.input_dim, config.hidden_dim, config.num_layers)
    model.summary(print_fn=lambda x: logger.info(x))
    
    logger.info("训练模型...")
    history = model.fit(X_seq, y, batch_size=config.batch_size, epochs=config.epochs, verbose=0)
    for epoch in [10, 20]:
        if epoch <= config.epochs:
            logger.info(f"  Epoch {epoch}: loss={history.history['loss'][epoch-1]:.4f}")
    
    logger.info("计算 SHAP 值...")
    background = X_seq[:config.shap_background]
    test = X_seq[config.shap_background:config.shap_background + config.shap_test]
    logger.info(f"  背景样本: {background.shape}")
    logger.info(f"  测试样本: {test.shape}")
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test)
    
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    logger.info(f"  SHAP 值形状: {shap_values.shape}")
    
    importance = np.abs(shap_values).mean(axis=(0, 1))
    
    logger.info("\n特征重要性排序:")
    sorted_idx = np.argsort(importance)[::-1]
    for idx in sorted_idx:
        logger.info(f"  {config.feature_names[idx]}: {importance[idx]:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(config.feature_names, importance, color='#2E8B57')
    ax.set_xlabel('平均 SHAP 重要性')
    ax.set_title(f'{config.data_type.upper()} 基站特征重要性 (DeepExplainer)')
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
    plt.tight_layout()
    bar_path = config.output_dir / f"feature_importance_{config.data_type}_{timestamp}.png"
    plt.savefig(bar_path, dpi=150)
    logger.info(f"图片保存: {bar_path}")
    
    results = {
        'data_type': config.data_type,
        'timestamp': timestamp,
        'feature_importance': {name: float(imp) for name, imp in zip(config.feature_names, importance)},
        'sorted': [{config.feature_names[i]: float(importance[i])} for i in sorted_idx]
    }
    result_path = config.output_dir / f"shap_results_{config.data_type}_{timestamp}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果保存: {result_path}")


if __name__ == "__main__":
    main()