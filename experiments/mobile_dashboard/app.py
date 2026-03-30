# -*- coding: utf-8 -*-
"""
Mobile Dashboard - 手机端仪表盘
基于树莓派推理引擎的实时能耗预测
"""

from flask import Flask, render_template, jsonify
import sys
import os
import numpy as np
import time
import random

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(experiments_dir)
sys.path.append(experiments_dir)

# 导入树莓派推理引擎
from raspberry_inference import RaspberryInferenceEngine, LSTMPredictor

app = Flask(__name__)

# ==================== 配置 ====================
MODEL_PATH = os.path.join(project_root, "checkpoints", "best_fedavg_barcelona.pth")
DATA_DIR = os.path.join(experiments_dir, "fl_data", "site_0")

# ==================== 全局变量 ====================
inference_engine = None

def initialize_inference():
    """初始化推理引擎"""
    global inference_engine
    try:
        print("📦 初始化推理引擎...")
        inference_engine = RaspberryInferenceEngine(
            model_path=MODEL_PATH,
            data_dir=DATA_DIR,
            device='cpu'
        )
        print("✅ 推理引擎初始化成功！")
        return True
    except Exception as e:
        print(f"⚠️  推理引擎初始化失败: {e}")
        print("   使用模拟数据演示...")
        return False

# 尝试初始化
inference_initialized = initialize_inference()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """获取实时状态API"""
    try:
        # 1. 获取当前时间
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 2. 模拟手机电池电量 (70-100%)
        battery_level = random.randint(70, 100)
        
        # 3. 获取推理结果
        if inference_initialized and inference_engine:
            try:
                # 使用真实推理
                # 读取测试数据
                X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
                y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
                mean = np.load(os.path.join(DATA_DIR, 'mean.npy'))
                std = np.load(os.path.join(DATA_DIR, 'std.npy'))
                
                # 随机选择一个样本
                sample_idx = random.randint(0, len(X_test) - 1)
                sample_input = X_test[sample_idx]
                
                # 推理（返回6个点的预测）
                next_6h, inference_time = inference_engine.predict_single(sample_input)
                
                # 当前功耗（真实值）
                current_power = float(y_test[sample_idx] * std[0] + mean[0])
                
                # 历史数据（最近24点）- 从原始数据读取
                history_data = []
                start_idx = max(0, sample_idx - 23)
                for i in range(start_idx, sample_idx + 1):
                    val = float(y_test[i] * std[0] + mean[0])
                    history_data.append(val)
                
                # 确保next_6h是list格式
                next_6h = next_6h.tolist() if hasattr(next_6h, 'tolist') else list(next_6h)
                
            except Exception as e:
                print(f"⚠️  推理失败，使用模拟数据: {e}")
                import traceback
                traceback.print_exc()
                # 模拟数据
                current_power = 4.5 + random.uniform(-1.0, 1.0)
                next_6h = [current_power + random.uniform(-0.5, 0.5) for _ in range(6)]
                history_data = [4.0 + random.uniform(-1.0, 1.0) for _ in range(24)]
        else:
            # 使用模拟数据
            current_power = 4.5 + random.uniform(-1.0, 1.0)
            next_6h = [current_power + random.uniform(-0.5, 0.5) for _ in range(6)]
            history_data = [4.0 + random.uniform(-1.0, 1.0) for _ in range(24)]
        
        # 4. 状态判断（根据真实数据范围调整）
        if current_power < 3.5:
            status = "Low"
            status_color = "#7c3aed"
        elif current_power < 5.5:
            status = "Normal"
            status_color = "#059669"
        else:
            status = "High"
            status_color = "#dc2626"
        
        # 5. 返回JSON
        return jsonify({
            'success': True,
            'data': {
                'current_time': current_time,
                'battery_level': battery_level,
                'current_power': round(current_power, 2),
                'status': status,
                'status_color': status_color,
                'history_24h': [round(x, 2) for x in history_data],
                'prediction_6h': [round(x, 2) for x in next_6h]
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    print("=" * 80)
    print("📱 Mobile Dashboard - 手机端能耗预测仪表盘")
    print("=" * 80)
    print(f"\n📂 项目根目录: {project_root}")
    print(f"📂 数据目录: {DATA_DIR}")
    print(f"📂 模型路径: {MODEL_PATH}")
    print(f"\n🚀 启动服务器...")
    print(f"🌐 访问地址: http://127.0.0.1:5000")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
