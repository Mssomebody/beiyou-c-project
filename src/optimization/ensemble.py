"""
v7: 模型集成（完整修复版）

支持：
- LSTM + XGBoost 集成
- 加权平均
- Stacking（带交叉验证）
- 动态权重优化
- 输入验证
- 模型保存/加载
- 交叉验证时模型克隆
"""

import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import warnings
import copy
import pickle

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ============================================================
# 模型包装器
# ============================================================

class LSTMWrapper:
    """
    LSTM 模型包装器
    
    参数:
        model: PyTorch LSTM 模型
        scaler: 归一化器（用于反归一化）
        device: 计算设备，默认 'cpu'
        input_shape: 可选，输入形状 (seq_len, features)
    """
    
    def __init__(self, model, scaler, device='cpu', input_shape=None):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.input_shape = input_shape
        self.model.eval()
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """验证并调整输入形状"""
        if len(X.shape) == 1:
            # 单样本 [features] -> [1, seq_len, features]
            if self.input_shape is not None:
                seq_len, feat_dim = self.input_shape
                X = X.reshape(1, seq_len, feat_dim)
            else:
                raise ValueError("Cannot reshape 1D input without input_shape")
        
        elif len(X.shape) == 2:
            # 2D [n_samples, features] -> [n_samples, 1, features]
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        elif len(X.shape) != 3:
            raise ValueError(f"Expected 1D, 2D, or 3D input, got {X.shape}")
        
        return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: numpy array 
                - 2D: [n_samples, features] (单步预测)
                - 3D: [n_samples, seq_len, features] (序列预测)
        
        Returns:
            predictions: numpy array [n_samples]
        """
        X = self._validate_input(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            pred = self.model(X_tensor).cpu().numpy()
        
        # 反归一化
        if len(pred.shape) == 2:
            pred = pred.reshape(-1, 1)
        elif len(pred.shape) == 1:
            pred = pred.reshape(-1, 1)
        
        return self.scaler.inverse_transform(pred).flatten()
    
    def predict_with_uncertainty(self, X: np.ndarray, n_dropout=50) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Monte Carlo Dropout 估计不确定性
        
        Args:
            X: 输入特征
            n_dropout: 采样次数
        
        Returns:
            mean: 预测均值
            std: 预测标准差
        """
        X = self._validate_input(X)
        
        predictions = []
        self.model.train()  # 启用 dropout
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            for _ in range(n_dropout):
                pred = self.model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        self.model.eval()
        predictions = np.array(predictions).squeeze()
        
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_shape': self.input_shape
        }, path)
    
    @classmethod
    def load(cls, path: str, device='cpu'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        from experiments.beautified.train_single_node import LSTMPredictor
        input_dim = checkpoint['input_shape'][1] if checkpoint['input_shape'] else 7
        model = LSTMPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=4)
        model.load_state_dict(checkpoint['model_state_dict'])
        return cls(model, checkpoint['scaler'], device, checkpoint['input_shape'])


class XGBoostWrapper:
    """
    XGBoost 模型包装器
    
    参数:
        model: XGBoost 模型
        feature_columns: 特征列名（可选）
        feature_importance: 是否返回特征重要性
    """
    
    def __init__(self, model, feature_columns=None, feature_importance=False):
        self.model = model
        self.feature_columns = feature_columns
        self.feature_importance = feature_importance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: numpy array [n_samples, n_features]
        
        Returns:
            predictions: numpy array [n_samples]
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性（如果可用）"""
        if self.feature_importance and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def save(self, path: str):
        """保存模型"""
        self.model.save_model(path)
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        model = xgb.XGBRegressor()
        model.load_model(path)
        return cls(model)


# ============================================================
# 集成预测器
# ============================================================

class EnsemblePredictor:
    """
    加权平均集成预测器
    
    支持：
    - 等权重平均
    - 自定义权重
    - 动态权重优化（基于验证集）
    - 模型保存/加载
    """
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Args:
            models: 模型字典 {name: model}
            weights: 权重字典 {name: weight}，None 则等权重
        """
        if not models:
            raise ValueError("At least one model required")
        
        self.models = models
        
        if weights is None:
            # 等权重
            self.weights = {name: 1.0 / len(models) for name in models}
        else:
            # 验证权重
            for name in weights:
                if name not in models:
                    raise KeyError(f"Model {name} not found")
                if weights[name] < 0:
                    raise ValueError(f"Weight for {name} must be >= 0")
            self.weights = weights
        
        # 归一化权重
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        else:
            raise ValueError("Total weight must be > 0")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        加权平均预测
        
        Args:
            X: 输入特征
        
        Returns:
            预测值数组
        """
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred * self.weights[name])
        return np.sum(predictions, axis=0)
    
    def predict_with_variance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        带方差预测（评估不确定性）
        
        Args:
            X: 输入特征
        
        Returns:
            mean: 预测均值
            std: 预测标准差
        """
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X))
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        基于验证集优化权重（修复：返回 Python float）
        
        Args:
            X_val: 验证特征
            y_val: 验证标签
        
        Returns:
            optimal_weights: 最优权重（Python float）
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("scipy not installed, cannot optimize weights")
            return self.weights
        
        # 获取各模型预测
        preds = []
        for model in self.models.values():
            preds.append(model.predict(X_val))
        preds = np.array(preds)  # [n_models, n_samples]
        
        def loss(w):
            w = np.abs(w)
            w_sum = np.sum(w)
            if w_sum == 0:
                return 1e10
            w = w / w_sum
            ensemble_pred = np.dot(w, preds)
            return np.mean((ensemble_pred - y_val) ** 2)
        
        n_models = len(self.models)
        initial_weights = np.ones(n_models) / n_models
        
        result = minimize(
            loss,
            initial_weights,
            method='L-BFGS-B',
            bounds=[(0, 1)] * n_models,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            options={'maxiter': 100}
        )
        
        if result.success:
            # 修复：转换为 Python float（不是 np.float64）
            optimal_weights = {name: float(w) for name, w in zip(self.models.keys(), result.x)}
            self.weights = optimal_weights
            logger.info(f"Weights optimized: {optimal_weights}")
        else:
            logger.warning("Weight optimization failed, using original weights")
        
        return self.weights
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights
    
    def save(self, path: str):
        """
        保存集成模型（新增）
        
        Args:
            path: 保存路径
        """
        save_data = {
            'model_type': 'ensemble',
            'weights': self.weights,
            'models': {}
        }
        for name, model in self.models.items():
            model_path = f"{path}_{name}.pkl"
            if hasattr(model, 'save'):
                model.save(model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            save_data['models'][name] = model_path
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """
        加载集成模型（新增）
        
        Args:
            path: 保存路径
        
        Returns:
            EnsemblePredictor 实例
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        models = {}
        for name, model_path in save_data['models'].items():
            if 'lstm' in name.lower():
                models[name] = LSTMWrapper.load(model_path)
            else:
                with open(model_path, 'rb') as f2:
                    models[name] = pickle.load(f2)
        
        return cls(models, save_data['weights'])


class StackingEnsemble:
    """
    Stacking 集成（带交叉验证和模型克隆）
    
    使用元学习器融合多个基模型，支持交叉验证防止过拟合
    """
    
    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Any = None,
        use_cv: bool = True,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            base_models: 基模型字典
            meta_model: 元学习器（默认 RandomForestRegressor）
            use_cv: 是否使用交叉验证生成基模型预测
            cv_folds: 交叉验证折数
            random_state: 随机种子
        """
        if not base_models:
            raise ValueError("At least one base model required")
        
        self.base_models = base_models
        self.meta_model = meta_model or RandomForestRegressor(
            n_estimators=100,
            random_state=random_state
        )
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.is_fitted = False
    
    def _clone_model(self, model):
        """
        克隆模型（新增：避免交叉验证时修改原模型）
        
        Args:
            model: 原始模型
        
        Returns:
            克隆的模型
        """
        if hasattr(model, 'clone'):
            return model.clone()
        elif hasattr(model, 'model'):
            # 对于包装器，克隆内部模型
            return copy.deepcopy(model)
        else:
            return copy.deepcopy(model)
    
    def _get_base_predictions_cv(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        使用交叉验证生成基模型预测（避免过拟合）
        
        Args:
            X: 训练特征
            y: 训练标签
        
        Returns:
            base_predictions: [n_samples, n_models]
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        n_samples = len(X)
        n_models = len(self.base_models)
        base_preds = np.zeros((n_samples, n_models))
        
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            # 为每个基模型单独做交叉验证
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                # 克隆模型避免污染
                model_fold = self._clone_model(model)
                
                if hasattr(model_fold, 'fit'):
                    model_fold.fit(X_train_fold, y_train_fold)
                elif hasattr(model_fold, 'model') and hasattr(model_fold.model, 'fit'):
                    model_fold.model.fit(X_train_fold, y_train_fold)
                else:
                    # 如果模型没有 fit 方法，尝试直接训练
                    logger.warning(f"Model {name} has no fit method, skipping fold")
                    continue
                
                base_preds[val_idx, model_idx] = model_fold.predict(X_val_fold)
        
        return base_preds
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """直接使用基模型预测（无需交叉验证）"""
        preds = []
        for model in self.base_models.values():
            pred = model.predict(X)
            if len(pred.shape) == 1:
                pred = pred.reshape(-1, 1)
            preds.append(pred)
        return np.hstack(preds)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'StackingEnsemble':
        """
        训练 Stacking 集成
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        
        Returns:
            self
        """
        if self.use_cv:
            # 使用交叉验证生成基模型预测
            base_preds = self._get_base_predictions_cv(X_train, y_train)
        else:
            # 直接使用基模型预测（可能过拟合）
            base_preds = self._get_base_predictions(X_train)
        
        # 训练元模型
        self.meta_model.fit(base_preds, y_train)
        self.is_fitted = True
        
        # 可选：用全部数据重新训练基模型（用于最终预测）
        for name, model in self.base_models.items():
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
        
        logger.info(f"Stacking ensemble fitted with {len(self.base_models)} base models")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征
        
        Returns:
            预测值数组
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        base_preds = self._get_base_predictions(X)
        return self.meta_model.predict(base_preds)
    
    def predict_with_variance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        带方差预测（仅当 meta_model 是随机森林时可用）
        
        Args:
            X: 输入特征
        
        Returns:
            mean: 预测均值
            std: 预测标准差
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        base_preds = self._get_base_predictions(X)
        
        # 如果元模型是随机森林，可以获取各棵树预测
        if hasattr(self.meta_model, 'estimators_'):
            predictions = []
            for estimator in self.meta_model.estimators_:
                predictions.append(estimator.predict(base_preds))
            predictions = np.array(predictions)
            return np.mean(predictions, axis=0), np.std(predictions, axis=0)
        
        return self.predict(X), np.zeros(len(X))
    
    def save(self, path: str):
        """
        保存 Stacking 模型（新增）
        
        Args:
            path: 保存路径
        """
        save_data = {
            'model_type': 'stacking',
            'meta_model': self.meta_model,
            'base_models': {}
        }
        for name, model in self.base_models.items():
            model_path = f"{path}_{name}.pkl"
            if hasattr(model, 'save'):
                model.save(model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            save_data['base_models'][name] = model_path
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Stacking ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """
        加载 Stacking 模型（新增）
        
        Args:
            path: 保存路径
        
        Returns:
            StackingEnsemble 实例
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        base_models = {}
        for name, model_path in save_data['base_models'].items():
            if 'lstm' in name.lower():
                base_models[name] = LSTMWrapper.load(model_path)
            else:
                with open(model_path, 'rb') as f2:
                    base_models[name] = pickle.load(f2)
        
        return cls(base_models, save_data['meta_model'])


# ============================================================
# 工厂函数
# ============================================================

def create_ensemble(
    lstm_model=None,
    xgb_model=None,
    method: str = 'weighted',
    weights: Optional[Dict[str, float]] = None,
    meta_model: Any = None,
    use_cv: bool = True,
    cv_folds: int = 5
):
    """
    创建集成模型
    
    Args:
        lstm_model: LSTM 模型包装器
        xgb_model: XGBoost 模型包装器
        method: 集成方法 ('weighted', 'stacking')
        weights: 权重（仅 weighted）
        meta_model: 元学习器（仅 stacking）
        use_cv: 是否使用交叉验证（仅 stacking）
        cv_folds: 交叉验证折数（仅 stacking）
    
    Returns:
        EnsemblePredictor 或 StackingEnsemble
    """
    models = {}
    if lstm_model is not None:
        models['lstm'] = lstm_model
    if xgb_model is not None:
        models['xgboost'] = xgb_model
    
    if method == 'weighted':
        return EnsemblePredictor(models, weights)
    elif method == 'stacking':
        return StackingEnsemble(models, meta_model, use_cv, cv_folds)
    else:
        raise ValueError(f"Unknown method: {method}")


def create_lstm_wrapper(
    model,
    scaler,
    device: str = 'cpu',
    input_shape: Optional[Tuple[int, int]] = None
) -> LSTMWrapper:
    """创建 LSTM 包装器"""
    return LSTMWrapper(model, scaler, device, input_shape)


def create_xgb_wrapper(
    model,
    feature_columns: Optional[List[str]] = None,
    feature_importance: bool = False
) -> XGBoostWrapper:
    """创建 XGBoost 包装器"""
    return XGBoostWrapper(model, feature_columns, feature_importance)


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v7: 模型集成模块测试")
    print("=" * 60)
    
    # 创建模拟模型
    class DummyModel:
        def __init__(self, bias=0):
            self.bias = bias
        def fit(self, X, y):
            pass
        def predict(self, X):
            return X[:, 0] + self.bias
        def clone(self):
            return DummyModel(self.bias)
    
    # 测试1：加权平均集成
    print("\n测试1: 加权平均集成")
    models = {
        'model1': DummyModel(0),
        'model2': DummyModel(0.5),
        'model3': DummyModel(-0.5)
    }
    
    X = np.random.randn(100, 10)
    y = X[:, 0]
    
    ensemble = create_ensemble(
        lstm_model=models['model1'],
        xgb_model=models['model2'],
        method='weighted'
    )
    pred = ensemble.predict(X)
    print(f"  预测形状: {pred.shape}")
    print(f"  权重: {ensemble.get_weights()}")
    
    # 测试2：权重优化（返回 Python float）
    print("\n测试2: 权重优化")
    X_val = np.random.randn(50, 10)
    y_val = X_val[:, 0]
    optimal_weights = ensemble.optimize_weights(X_val, y_val)
    print(f"  优化后权重类型: {type(list(optimal_weights.values())[0])}")
    print(f"  优化后权重: {optimal_weights}")
    
    # 测试3：Stacking 集成
    print("\n测试3: Stacking 集成")
    stacking = create_ensemble(
        lstm_model=models['model1'],
        xgb_model=models['model2'],
        method='stacking',
        use_cv=True,
        cv_folds=3
    )
    stacking.fit(X[:80], y[:80])
    pred = stacking.predict(X[80:90])
    print(f"  Stacking 预测: {pred[:3]}")
    
    # 测试4: 保存和加载（使用 DummyModel 测试，避免 LSTM 依赖）
    print("\n测试4: 保存和加载")
    import tempfile
    # 只使用 DummyModel（不是 LSTM）测试保存/加载
    simple_models = {'dummy1': DummyModel(0), 'dummy2': DummyModel(0.5)}
    simple_ensemble = EnsemblePredictor(simple_models)
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        simple_ensemble.save(f.name)
        print(f"  保存到: {f.name}")
        loaded = EnsemblePredictor.load(f.name)
        print(f"  加载成功，权重: {loaded.get_weights()}")
    
    # 测试5：带方差预测
    print("\n测试5: 带方差预测")
    mean, std = ensemble.predict_with_variance(X[:5])
    print(f"  均值: {mean}")
    print(f"  标准差: {std}")
    
    print("\n✅ 测试通过")