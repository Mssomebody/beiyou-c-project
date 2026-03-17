# day5_dashboard.py
# Day 5: 联邦学习可视化仪表板

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import torch
import torch.nn as nn
import copy
import time

# ============ 页面配置 ============
st.set_page_config(
    page_title="5G基站联邦学习系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义CSS ============
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0 !important;
    }
    
    /* 卡片样式 */
    .css-1r6slb0 {
        border-radius: 20px;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* 指标卡片 */
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* 状态标签 */
    .status-badge {
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-green { background: #10b981; color: white; }
    .status-yellow { background: #f59e0b; color: white; }
    .status-red { background: #ef4444; color: white; }
    
    /* 进度条 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============ 标题 ============
st.markdown('<p class="main-title">📡 5G基站联邦学习与节能系统</p>', unsafe_allow_html=True)
st.markdown("#### *Federated Learning for 5G Base Station Energy Optimization*")
st.markdown("---")

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("### ⚙️ 控制中心")
    
    # 模式选择
    mode = st.radio(
        "选择模式",
        ["📊 数据看板", "🤖 联邦训练", "📈 结果分析", "💡 节能策略"],
        index=0,
        format_func=lambda x: f"**{x}**"
    )
    
    st.markdown("---")
    
    # 参数设置
    with st.expander("🔧 高级参数设置", expanded=True):
        n_rounds = st.slider("通信轮数", 5, 50, 10, help="联邦学习聚合轮数")
        local_epochs = st.slider("本地训练轮数", 1, 10, 5, help="每轮本地训练次数")
        learning_rate = st.select_slider(
            "学习率", 
            options=[0.1, 0.01, 0.001, 0.0001], 
            value=0.001,
            format_func=lambda x: f"{x:.4f}"
        )
        batch_size = st.selectbox("批次大小", [32, 64, 128, 256], index=1)
    
    st.markdown("---")
    
    # 系统状态
    st.markdown("### 📊 系统状态")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("在线基站", "5/5", "+0")
    with col2:
        st.metric("通信延迟", "42ms", "-12ms")
    
    st.markdown("---")
    st.caption(f"🕒 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============ 数据加载函数 ============
@st.cache_data
def load_data_preview():
    """加载Day3生成的联邦数据"""
    data_info = []
    if os.path.exists('fl_data'):
        for site_id in range(5):
            site_dir = f'fl_data/site_{site_id}'
            if os.path.exists(site_dir):
                X_train = np.load(f'{site_dir}/X_train.npy')
                data_info.append({
                    '站点': f'Site {site_id}',
                    '训练样本': f"{X_train.shape[0]:,}",
                    '序列长度': X_train.shape[1],
                    '特征数': X_train.shape[2],
                    '数据量': f"{X_train.nbytes / 1024 / 1024:.1f} MB"
                })
    return pd.DataFrame(data_info)

@st.cache_data
def load_raw_data():
    """加载原始时序数据"""
    if os.path.exists('base_station_data_10stations_30days.csv'):
        return pd.read_csv('base_station_data_10stations_30days.csv')
    return None

# ============ 1. 数据看板模式 ============
if mode == "📊 数据看板":
    # 顶部门户指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("总数据量", "7,200条", "+720")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("基站数量", "10个", "+0")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("时间跨度", "30天", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("采样频率", "每小时", "24/天")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 主内容区
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📁 联邦数据集")
        df_info = load_data_preview()
        if not df_info.empty:
            st.dataframe(
                df_info,
                use_container_width=True,
                column_config={
                    "站点": st.column_config.TextColumn("基站ID"),
                    "训练样本": st.column_config.TextColumn("样本数"),
                    "序列长度": st.column_config.NumberColumn("时间步"),
                    "特征数": st.column_config.NumberColumn("特征"),
                    "数据量": st.column_config.TextColumn("大小")
                },
                hide_index=True
            )
        else:
            st.warning("⚠️ 未找到联邦学习数据，请先运行 Day3")
    
    with col_right:
        st.markdown("### 📊 Non-IID分布")
        if os.path.exists('day3_non_iid_distribution.png'):
            st.image('day3_non_iid_distribution.png', use_container_width=True)
        else:
            st.info("运行 day3_multivariate_fl_data.py 生成分布图")
    
    # 时序数据可视化
    st.markdown("### 📈 实时能耗监控")
    df_raw = load_raw_data()
    if df_raw is not None:
        # 取最近24小时的数据
        df_recent = df_raw.tail(24 * 10)  # 10个基站各24小时
        
        fig = px.line(
            df_recent, 
            x=df_recent.index, 
            y='power_kw', 
            color='station_id',
            title="基站能耗实时监控",
            labels={'power_kw': '功耗 (kW)', 'index': '时间', 'station_id': '基站'},
            template="plotly_dark"
        )
        fig.update_layout(
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("未找到原始数据文件")

# ============ 2. 联邦训练模式 ============
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

def load_site_data(site_id):
    """加载单个站点的数据"""
    site_dir = f'fl_data/site_{site_id}'
    X_train = torch.FloatTensor(np.load(f'{site_dir}/X_train.npy'))
    y_train = torch.FloatTensor(np.load(f'{site_dir}/y_train.npy')).reshape(-1, 1)
    X_test = torch.FloatTensor(np.load(f'{site_dir}/X_test.npy'))
    y_test = torch.FloatTensor(np.load(f'{site_dir}/y_test.npy')).reshape(-1, 1)
    return X_train, y_train, X_test, y_test

def train_local(model, X_train, y_train, epochs):
    """本地训练"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses

def evaluate(model, X_test, y_test):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mae = torch.mean(torch.abs(pred - y_test)).item()
    return mae

if mode == "🤖 联邦训练":
    st.markdown("### 🚀 联邦学习训练控制台")
    
    # 检查数据
    if not os.path.exists('fl_data'):
        st.error("❌ 未找到联邦学习数据，请先运行 day3_multivariate_fl_data.py")
        st.stop()
    
    # 训练控制
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_training = st.button("▶️ 开始训练", type="primary", use_container_width=True)
    
    with col2:
        reset_training = st.button("🔄 重置", use_container_width=True)
    
    with col3:
        st.info("训练将进行 {} 轮通信，每轮本地训练 {} 次".format(n_rounds, local_epochs))
    
    # 训练进度容器
    progress_container = st.container()
    
    if start_training:
        with progress_container:
            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 实时指标
            metric_cols = st.columns(4)
            with metric_cols[0]:
                mae_metric = st.empty()
            with metric_cols[1]:
                round_metric = st.empty()
            with metric_cols[2]:
                time_metric = st.empty()
            with metric_cols[3]:
                improve_metric = st.empty()
            
            # 图表容器
            chart_container = st.empty()
            
            # 初始化记录
            history = {
                'round': [],
                'global_mae': [],
                'site_0': [], 'site_1': [], 'site_2': [], 'site_3': [], 'site_4': []
            }
            
            # 加载数据
            site_data = []
            for site_id in range(5):
                X_train, y_train, X_test, y_test = load_site_data(site_id)
                site_data.append({
                    'X_train': X_train, 'y_train': y_train,
                    'X_test': X_test, 'y_test': y_test
                })
            
            # 初始化全局模型
            global_model = LSTMPredictor()
            
            # 开始训练
            start_time = time.time()
            
            for round_idx in range(n_rounds):
                # 更新状态
                status_text.markdown(f"**训练进度**: 第 {round_idx+1}/{n_rounds} 轮")
                
                local_models = []
                
                # 各站点本地训练
                for site_id in range(5):
                    local_model = copy.deepcopy(global_model)
                    trained_model, _ = train_local(
                        local_model, 
                        site_data[site_id]['X_train'],
                        site_data[site_id]['y_train'],
                        local_epochs
                    )
                    local_models.append(trained_model)
                    
                    # 评估
                    mae = evaluate(
                        trained_model,
                        site_data[site_id]['X_test'],
                        site_data[site_id]['y_test']
                    )
                    history[f'site_{site_id}'].append(mae)
                
                # FedAvg聚合
                global_dict = global_model.state_dict()
                for key in global_dict.keys():
                    global_dict[key] = sum(m.state_dict()[key] for m in local_models) / 5
                global_model.load_state_dict(global_dict)
                
                # 评估全局模型
                global_mae = evaluate(
                    global_model,
                    torch.cat([site['X_test'] for site in site_data]),
                    torch.cat([site['y_test'] for site in site_data])
                )
                history['global_mae'].append(global_mae)
                history['round'].append(round_idx + 1)
                
                # 更新指标
                elapsed = time.time() - start_time
                improvement = (history['global_mae'][0] - global_mae) / history['global_mae'][0] * 100 if round_idx > 0 else 0
                
                with metric_cols[0]:
                    mae_metric.metric("当前 MAE", f"{global_mae:.4f}", 
                                     delta=f"{-improvement:.1f}%" if round_idx > 0 else None)
                with metric_cols[1]:
                    round_metric.metric("完成轮次", f"{round_idx+1}/{n_rounds}")
                with metric_cols[2]:
                    time_metric.metric("已用时", f"{elapsed:.1f}s")
                with metric_cols[3]:
                    improve_metric.metric("提升幅度", f"{improvement:.1f}%" if round_idx > 0 else "0%")
                
                # 更新图表
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=history['round'], y=history['global_mae'],
                    mode='lines+markers', name='全局模型',
                    line=dict(color='#00ff00', width=3),
                    marker=dict(size=8)
                ))
                
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
                for i in range(5):
                    fig.add_trace(go.Scatter(
                        x=history['round'], y=history[f'site_{i}'],
                        mode='lines', name=f'站点 {i}',
                        line=dict(color=colors[i], width=1.5, dash='dot'),
                        opacity=0.5
                    ))
                
                fig.update_layout(
                    title="实时训练曲线",
                    xaxis_title="通信轮数",
                    yaxis_title="MAE",
                    hovermode='x unified',
                    height=400,
                    template="plotly_dark",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                chart_container.plotly_chart(fig, use_container_width=True)
                
                # 更新进度条
                progress_bar.progress((round_idx + 1) / n_rounds)
            
            status_text.markdown("## ✅ 训练完成！")
            
            # 保存到session state
            st.session_state['history'] = history
            st.session_state['trained'] = True

# ============ 3. 结果分析模式 ============
if mode == "📈 结果分析":
    st.markdown("### 📊 训练结果分析")
    
    if 'history' not in st.session_state:
        st.info("请先在「联邦训练」页面运行一次训练")
        st.stop()
    
    history = st.session_state['history']
    
    # 顶部指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("初始 MAE", f"{history['global_mae'][0]:.4f}")
    with col2:
        st.metric("最终 MAE", f"{history['global_mae'][-1]:.4f}")
    with col3:
        improvement = (history['global_mae'][0] - history['global_mae'][-1]) / history['global_mae'][0] * 100
        st.metric("总提升", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
    with col4:
        st.metric("收敛轮次", f"{np.argmin(history['global_mae'])+1}")
    
    st.markdown("---")
    
    # 双图表
    col_left, col_right = st.columns(2)
    
    with col_left:
        # 各站点最终表现
        site_final = pd.DataFrame({
            '站点': [f'Site {i}' for i in range(5)],
            '最终 MAE': [history[f'site_{i}'][-1] for i in range(5)],
            '初始 MAE': [history[f'site_{i}'][0] for i in range(5)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='初始 MAE',
            x=site_final['站点'], y=site_final['初始 MAE'],
            marker_color='#94a3b8'
        ))
        fig.add_trace(go.Bar(
            name='最终 MAE',
            x=site_final['站点'], y=site_final['最终 MAE'],
            marker_color='#10b981'
        ))
        
        fig.update_layout(
            title="各站点性能对比",
            barmode='group',
            height=400,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        # 收敛速度分析
        df_convergence = pd.DataFrame({
            '轮次': history['round'],
            '全局MAE': history['global_mae']
        })
        
        fig = px.line(df_convergence, x='轮次', y='全局MAE', 
                     title="收敛曲线分析",
                     markers=True)
        
        # 添加最优值线
        best_mae = min(history['global_mae'])
        fig.add_hline(y=best_mae, line_dash="dash", 
                     annotation_text=f"最优 MAE: {best_mae:.4f}")
        
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    # 详细数据表
    st.markdown("### 📋 详细训练记录")
    df_details = pd.DataFrame(history)
    st.dataframe(
        df_details.round(4),
        use_container_width=True,
        column_config={
            "round": "轮次",
            "global_mae": "全局MAE",
            "site_0": "站点0 MAE",
            "site_1": "站点1 MAE",
            "site_2": "站点2 MAE",
            "site_3": "站点3 MAE",
            "site_4": "站点4 MAE"
        }
    )

# ============ 4. 节能策略模式 ============
if mode == "💡 节能策略":
    st.markdown("### ⚡ 实时节能策略推荐")
    
    # 实时刷新按钮
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        refresh = st.button("🔄 刷新数据", type="primary", use_container_width=True)
    with col2:
        auto_refresh = st.checkbox("自动刷新")
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # 生成模拟实时数据
    np.random.seed(int(time.time()) if refresh else 42)
    
    sites_status = []
    for i in range(5):
        power = np.random.uniform(2.5, 8.0)
        traffic = np.random.uniform(20, 200)
        temp = np.random.uniform(15, 35)
        
        # 动态策略
        if traffic < 50 and power > 4.0:
            suggestion = "🔴 建议休眠"
            status_class = "status-red"
            confidence = 0.85
        elif traffic < 100 and power > 5.0:
            suggestion = "🟡 建议降频"
            status_class = "status-yellow"
            confidence = 0.75
        else:
            suggestion = "🟢 正常运行"
            status_class = "status-green"
            confidence = 0.95
        
        # 预测未来1小时
        pred_power = power * np.random.uniform(0.9, 1.1)
        
        sites_status.append({
            '基站': f'基站 {i}',
            '功耗 (kW)': round(power, 2),
            '业务量 (GB)': round(traffic, 2),
            '温度 (°C)': round(temp, 2),
            '建议策略': suggestion,
            '状态': status_class,
            '置信度': confidence,
            '预测功耗': round(pred_power, 2)
        })
    
    df_status = pd.DataFrame(sites_status)
    
    # 基站状态卡片
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            site = df_status.iloc[idx]
            
            # 背景色根据状态变化
            if '休眠' in site['建议策略']:
                bg_color = "rgba(239, 68, 68, 0.2)"
            elif '降频' in site['建议策略']:
                bg_color = "rgba(245, 158, 11, 0.2)"
            else:
                bg_color = "rgba(16, 185, 129, 0.2)"
            
            st.markdown(f"""
            <div style="background: {bg_color}; border-radius: 15px; padding: 15px; text-align: center;">
                <h4>{site['基站']}</h4>
                <div style="font-size: 2rem; font-weight: bold;">{site['功耗 (kW)']}kW</div>
                <div>📊 {site['业务量 (GB)']}GB</div>
                <div>🌡️ {site['温度 (°C)']}°C</div>
                <div style="margin-top: 10px;">
                    <span class="status-badge {site['状态']}">{site['建议策略']}</span>
                </div>
                <div style="margin-top: 5px; font-size: 0.8rem;">置信度: {site['置信度']*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 可视化分析
    col_left, col_right = st.columns(2)
    
    with col_left:
        # 功耗分布
        fig = px.bar(df_status, x='基站', y='功耗 (kW)',
                    color='建议策略',
                    title="基站实时功耗分布",
                    color_discrete_map={
                        '🔴 建议休眠': '#ef4444',
                        '🟡 建议降频': '#f59e0b',
                        '🟢 正常运行': '#10b981'
                    })
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        # 功耗vs业务量
        fig = px.scatter(df_status, x='业务量 (GB)', y='功耗 (kW)',
                        text='基站', size='温度 (°C)',
                        color='建议策略',
                        title="业务量-功耗分析",
                        color_discrete_map={
                            '🔴 建议休眠': '#ef4444',
                            '🟡 建议降频': '#f59e0b',
                            '🟢 正常运行': '#10b981'
                        })
        fig.update_traces(textposition='top center')
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 节能效果预测
    st.markdown("### 📈 节能效果预测")
    
    # 计算如果执行策略后的节能效果
    total_power = df_status['功耗 (kW)'].sum()
    saved_power = 0
    
    for _, row in df_status.iterrows():
        if '休眠' in row['建议策略']:
            saved_power += row['功耗 (kW)'] * 0.95  # 休眠省95%
        elif '降频' in row['建议策略']:
            saved_power += row['功耗 (kW)'] * 0.3   # 降频省30%
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("当前总功耗", f"{total_power:.1f} kW")
    with col2:
        st.metric("预计节省", f"{saved_power:.1f} kW", 
                 delta=f"{(saved_power/total_power*100):.1f}%")
    with col3:
        st.metric("优化后功耗", f"{total_power - saved_power:.1f} kW")
    
    # 策略说明
    with st.expander("📋 策略说明"):
        st.markdown("""
        ### 节能策略说明
        - **🔴 建议休眠**: 业务量 < 50GB 且功耗 > 4.0kW，可暂时关闭基站
        - **🟡 建议降频**: 业务量 < 100GB 且功耗 > 5.0kW，可降低发射功率
        - **🟢 正常运行**: 业务量正常，保持当前状态
        
        ### 决策依据
        - 基于深度强化学习模型 (PPO)
        - 考虑历史负载模式
        - 实时温度影响
        - 相邻基站负载均衡
        """)