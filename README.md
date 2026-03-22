# FedGreen-C: 面向5G基站能耗预测的联邦学习系统

> 基于联邦学习的5G基站能耗预测系统 | PyTorch | LSTM | FedProx | Optuna | 手机部署

---

## 📌 项目概述

本项目实现了一套完整的5G基站能耗预测系统，核心创新点：

- **联邦学习框架**：FedProx算法，支持42个基站节点协同训练，保护数据隐私
- **自适应早停**：基于统计检验(t-test)和运行平均值检测，无需手动设置patience
- **贝叶斯超参数优化**：8个超参数自动搜索，sMAPE从69.93%降至62.64%
- **v1 最优基线**：特征选择后达到 **61.73%**，超越所有 v2.5 特征工程尝试
- **边缘部署**：INT8量化压缩比3.12x，树莓派推理19.3ms/样本，手机端部署
- **实时监控**：Flask + ECharts手机仪表盘，10秒自动刷新

**技术栈**：PyTorch 2.10 | LSTM | FedProx | Optuna 4.8 | Flask | ECharts | 手机 (VIVO Y50)

---

## 📊 数据集

| 维度 | 数据 |
|:---|:---|
| 来源 | 巴塞罗那开放数据 (Open Data BCN) |
| 时间范围 | 2019-2025 (7年) |
| 数据量 | 1,665,130 行 |
| 节点数 | 42个邮编区 (模拟基站) |
| 时间粒度 | 6小时/时段 |
| 特征 | 能耗值 + 节假日 + 周末 + 基站类型(4类) |

**预处理流程**：
1. 过滤"No consta"无效时段 (减少333,026行)
2. 按邮编分组 → 42个独立节点
3. 时序划分 (70%训练 / 15%验证 / 15%测试)
4. 每个节点独立MinMax归一化
5. 构建滑动窗口 (28时段输入 → 4时段输出)

---

## 📊 核心成果

### 单节点优化路线

| 阶段 | 方法 | sMAPE | 累计提升 |
|:---|:---|:---:|:---:|
| Step 1 | 阈值优化 (15%分位数) | 69.93% | 基线 |
| Step 2 | 自适应早停 (统计检验+运行平均值) | 68.15% | +1.78% |
| Step 3 | 贝叶斯优化 (8个超参数) | 62.64% | +7.29% |
| **v1 最佳** | **特征选择 + 超参数调优** | **61.73%** | **+8.20%** |

**最新验证** (2026-03-22): sMAPE **62.40%**，RMSE 10,830 kWh，MAE 7,029 kWh

**最佳超参数**：
```bash
hidden_dim=192, lr=0.002, dropout=0.45, batch_size=48
```

### 贝叶斯优化最佳超参数

| 参数 | 值 | 搜索范围 | 说明 |
|:---|:---|:---|:---|
| hidden_dim | 256 | 32-256 | LSTM隐藏层维度 |
| num_layers | 2 | 1-4 | LSTM层数 |
| learning_rate | 0.00363 | 1e-4 ~ 1e-2 (log空间) | Adam优化器 |
| dropout | 0.46 | 0.1-0.5 | Dropout比例 |
| batch_size | 32 | 32, 64, 128 | 批次大小 |
| optimizer | Adam | Adam, AdamW | 优化器类型 |
| scheduler | plateau | plateau, cosine | 学习率调度器 |
| grad_clip | 0.34 | 0.1-5.0 (log空间) | 梯度裁剪阈值 |

### 特征工程对比（v1 vs v2.5）

| 版本 | 特征数 | 参数量 | sMAPE | 结论 |
|:---|:---:|:---:|:---:|:---|
| v2.5 原版 | 19 | 460k | 70.34% | 过拟合 |
| v2.5 精选 | 16 | 118k | 67.21% | 仍过拟合 |
| v2.5 超级精选 | 12 | 54k | 68.17% | 仍无效 |
| **v1 最佳** | **7** | **451k** | **61.73%** | **最优** |

### 联邦学习结果

| 模型 | 节点数 | 特征 | 轮数 | sMAPE | 状态 |
|:---|:---:|:---:|:---:|:---:|:---:|
| FedAvg | 42 | 仅能耗 | 10 | 65-70% | ✅ 已完成 |
| FedProx (mu=0.001) | 24 | v1 特征 | 20 | 运行中 | ▶️ |
| FedProx (mu=0.01) | 24 | v1 特征 | 20 | 运行中 | ▶️ |

### 贝叶斯优化（进行中）

| Trial | hidden | layers | lr | dropout | bs | optimizer | scheduler | grad_clip | Val Loss | Test sMAPE |
|-------|--------|--------|-----|---------|-----|-----------|-----------|-----------|----------|------------|
| 0 | 192 | 2 | 2.76e-03 | 0.18 | 32 | Adam | cosine | 0.93 | 0.001756 | **62.59%** |
| 3 | 224 | 2 | 7.29e-04 | 0.39 | 32 | Adam | plateau | 0.20 | 0.002160 | 63.97% |
| 6 | 96 | 4 | 9.21e-03 | 0.18 | 128 | AdamW | plateau | 0.40 | 0.002008 | 64.07% |

**状态**：20次试验，已运行 7/20 次，预计 1-2 小时完成

### 边缘部署性能

| 指标 | FP32 | INT8 | 提升 |
|:---|:---:|:---:|:---:|
| 模型大小 | 3.8 MB | 1.2 MB | 压缩比 **3.12x** |
| 推理时间 (树莓派4B) | 15.6 ms | 19.3 ms | +24% |
| MSE | 0.33304 | 0.33305 | 损失 **0.01%** |
| 特征工程 | 1个原始特征 | 11个工程特征 | 滚动统计+差分+时间 |

---

## 📈 训练曲线

### 1. 自适应早停对比图
![自适应早停](results/beautified/adaptive_comparison_20260320_123038.png)
*统计检验早停 vs 固定早停，18轮自动停止，验证损失0.003931*

### 2. v1 单节点预测结果（最佳）
![v1 预测图](results/beautified/node_8001_v1_predictions_20260322_000255.png)
*6小时粒度基站能耗预测，sMAPE 62.40%，趋势一致*

### 3. v1 单节点损失曲线
![v1 损失曲线](results/beautified/node_8001_v1_loss_20260322_000255.png)
*20轮训练，训练损失0.001387，验证损失0.002003，无过拟合*

### 4. 42节点 FedAvg 联邦学习
![42节点 FedAvg](results/beautified/federated_nodes42_rounds10_mu0.0_loss_20260321_154304.png)
*10轮联邦训练，测试损失0.032457*

### 5. 树莓派手机仪表盘
![手机仪表盘](results/raspberry_prediction_20260319_131514.png)
*Flask后端 + ECharts前端，实时监控6小时预测*

---

## 🚀 优化模块

| 模块 | 功能 | 状态 | 说明 |
|:---|:---|:---:|:---|
| v3 周期性编码 | sin/cos 小时/星期/月份编码 | ✅ | 新增6列特征 |
| v4 注意力机制 | 自注意力 + 多头注意力 | ✅ | 自注意力537k参数 |
| v5 天气数据 | 温度/湿度/降水/风速 + 滞后/滚动 | ✅ | 新增45列特征 |
| v6 个性化联邦 | 自适应 mu + 个性化参数 | ✅ | 支持自适应正则化 |
| v7 模型集成 | 加权平均 + Stacking | ✅ | 支持保存/加载 |

---

## 📱 边缘部署 - 手机方案

### 手机配置 (VIVO Y50)

| 配置 | 值 |
|:---|:---|
| 型号 | VIVO Y50 (V1965A) |
| 处理器 | 骁龙 665 (2.0GHz 8核) |
| 内存 | 8GB |
| 存储 | 128GB |
| Android | 10 |

### 可实现功能

| 功能 | 实现方式 | 状态 |
|:---|:---|:---:|
| 模型推理 | PyTorch CPU 推理 | ✅ |
| INT8量化 | torch.quantization | ✅ |
| API服务 | Flask REST API | ✅ |
| 实时监控 | 手机浏览器界面 | ✅ |
| 远程访问 | 4G/WiFi 网络 | ✅ |
| ADB调试 | 项目内置 tools/ | ✅ |

---

## 📁 项目结构

```
beiyou_c_project/
├── data/processed/
│   ├── barcelona_ready/           # v2.5 数据
│   └── barcelona_ready_v1/        # v1 数据（最优）
├── src/
│   ├── data_loader/
│   │   ├── barcelona_preprocess_v1.py
│   │   ├── barcelona_dataset_v1.py
│   │   └── barcelona_preprocess.py
│   ├── federated/
│   │   ├── fedprox_client.py
│   │   └── fedprox_server.py
│   └── optimization/
│       ├── cyclical_encoder.py    # v3
│       ├── attention_lstm.py      # v4
│       ├── weather_data.py        # v5
│       ├── personalized_fed.py    # v6
│       └── ensemble.py            # v7
├── experiments/beautified/
│   ├── train_single_node_v1.py    # v1 训练
│   ├── train_federated.py         # 联邦训练
│   └── bayes_pro.py               # 贝叶斯优化
├── configs/
│   ├── federated_v1_24nodes_optimized.json
│   └── optimization/optimized_config.yaml
├── results/beautified/
│   ├── node_8001_v1_predictions_20260322_000255.png
│   ├── node_8001_v1_loss_20260322_000255.png
│   ├── federated_nodes42_rounds10_mu0.0_loss_20260321_154304.png
│   └── federated_nodes3_rounds10_mu0.0_*.pkl
├── tools/
│   └── platform-tools/            # ADB 手机调试
├── scripts/
│   ├── run_federated.sh
│   ├── run_single_node.sh
│   └── run_bayes.sh
├── docs/daily_logs/
│   ├── 2026-03-15_day1.md
│   ├── ...
│   └── 2026-03-22_day14.md
└── versions/
    └── v2_holiday_sector/         # 优化版本
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. v1 单节点训练（最佳配置）
python experiments/beautified/train_single_node_v1.py \
    --node 8001 --epochs 20 --percentile 15 \
    --hidden_dim 192 --lr 0.002 --dropout 0.45 --batch_size 48

# 3. 联邦学习（24节点）
python experiments/beautified/train_federated.py \
    --config configs/federated_v1_24nodes_optimized.json

# 4. 贝叶斯优化
python versions/v2_holiday_sector/experiments/beautified/bayes_pro.py --trials 20

# 5. 手机部署
./tools/platform-tools/adb.exe push models/ /sdcard/

# 6. 树莓派推理
cd experiments && python raspberry_inference.py

# 7. 手机仪表盘
python experiments/mobile_dashboard/app.py
# 访问: http://127.0.0.1:5000
```

---

## 📊 模型性能对比

| 模型 | 数据集 | 指标 | 结果 |
|:---|:---|:---|:---:|
| MLP | XOR | 准确率 | 100% |
| CNN | MNIST | 准确率 | 99.15% |
| LSTM | 时序 | MAE | 0.4059 |
| GCN | Cora | 准确率 | 79.9% |
| GAT (默认) | Cora | 准确率 | 82.4% |
| GAT (调优) | Cora | 准确率 | **84.2%** |
| LSTM (基线) | 基站能耗 | sMAPE | 69.93% |
| LSTM (早停) | 基站能耗 | sMAPE | 68.15% |
| LSTM (贝叶斯) | 基站能耗 | sMAPE | 62.64% |
| **LSTM (v1 最佳)** | **基站能耗** | **sMAPE** | **61.73%** 🏆 |

---

## 📝 技术要点

### 1. 自适应早停原理
- **统计检验 (t-test)**：比较最近10轮与之前10轮损失，p>0.05表示无显著改善
- **运行平均值检测**：近期平均损失 > 前期平均损失，表示开始恶化
- **改善检验**：相对改善 < 0.5% 时停止

### 2. 贝叶斯优化原理
- **高斯过程代理模型**：拟合超参数与损失的关系
- **采集函数 (EI)**：平衡探索与利用，选择下一个试验点
- **SQLite存储**：支持中断恢复，可随时查看中间结果

### 3. FedProx原理
```
Loss_local = MSE(y_pred, y_true) + (μ/2) * ||w - w_global||²
```
- μ=0：退化为FedAvg
- μ=0.01：中等约束，平衡全局与局部
- μ=0.1：强约束，趋近全局模型

### 4. 特征选择结论
- 6小时粒度数据下，滞后/滚动特征无效
- 基础特征（能耗+部门+节假日+周末）已足够
- 更多特征引入噪声，导致过拟合

---

## 📝 下一步

| 优先级 | 任务 | 预期提升 | 状态 |
|:---:|:---|:---:|:---:|
| P0 | v1 单节点最优 (61.73%) | - | ✅ |
| P1 | 24节点联邦学习 | 58-62% | ▶️ 过夜跑 |
| P2 | 贝叶斯优化完成 | 61-63% | ▶️ 进行中 |
| P3 | 手机端部署 | - | 📋 |
| P4 | 1小时粒度数据获取 | 30-40% | 📋 |

---

## 🔗 每日日志
- [Day 1: MLP与反向传播](docs/daily_logs/2026-03-15_day1.md)
- [Day 2-8: LSTM + FedAvg + GCN + GAT](docs/daily_logs/2026-03-16_day2-8.md)
- [Day 9: 真实数据集 + FedProx](docs/daily_logs/2026-03-17_day9.md)
- [Day 10: 树莓派部署 + 手机仪表盘](docs/daily_logs/2026-03-19_day11.md)
- [Day 11: 数据预处理 + 单节点基线](docs/daily_logs/2026-03-20_day12.md)
- [Day 12: 自适应早停 + 贝叶斯优化](docs/daily_logs/2026-03-21_day13.md)
- [Day 13: v1 最优 + 联邦学习过夜跑](docs/daily_logs/2026-03-22_day14.md)

---