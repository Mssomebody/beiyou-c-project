"""
巴塞罗那基站能耗数据预处理
功能：
1. 合并2019-2025年所有数据
2. 过滤无效时段（No consta）
3. 按邮编划分42个联邦节点
4. 时序划分（70%训练/15%验证/15%测试）
5. 每个节点独立归一化
6. 保存预处理数据
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 输入输出路径
DATA_INPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "barcelona_ready")

# 参数配置
WINDOW_SIZE = 28      # 输入窗口：过去28个时段（7天）
PREDICT_SIZE = 4      # 输出窗口：未来4个时段（1天）
TRAIN_RATIO = 0.7     # 训练集比例
VAL_RATIO = 0.15      # 验证集比例
TEST_RATIO = 0.15     # 测试集比例

# 创建输出目录
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 数据加载与合并
# ============================================================
def load_and_merge_data():
    """加载2019-2025年所有数据并合并"""
    print("=" * 60)
    print("阶段1.1: 加载数据")
    print("=" * 60)
    
    # 获取所有数据文件
    files = [f for f in os.listdir(DATA_INPUT_DIR) 
             if f.endswith('.csv') and 'consum' in f]
    
    all_dfs = []
    for file in sorted(files):
        file_path = os.path.join(DATA_INPUT_DIR, file)
        df = pd.read_csv(file_path)
        print(f"  ✓ {file}: {len(df):,} 行")
        all_dfs.append(df)
    
    # 合并
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  ✅ 合并完成: {len(df):,} 行")
    
    # 转换日期格式
    df['Data'] = pd.to_datetime(df['Data'])
    
    # 按日期排序
    df = df.sort_values(['Data', 'Codi_Postal']).reset_index(drop=True)
    print(f"  时间范围: {df['Data'].min().date()} ~ {df['Data'].max().date()}")
    
    return df


# ============================================================
# 2. 数据清洗（过滤无效时段）
# ============================================================
def clean_data(df):
    """过滤无效数据"""
    print("\n" + "=" * 60)
    print("阶段1.2: 数据清洗")
    print("=" * 60)
    
    original_len = len(df)
    
    # 过滤"No consta"时段
    df = df[df['Tram_Horari'] != 'No consta']
    print(f"  过滤'No consta'时段: {original_len:,} → {len(df):,} 行 (减少 {original_len - len(df):,})")
    
    # 过滤能耗为0的数据（可选，保留以测试鲁棒性）
    # df = df[df['Valor'] > 0]
    # print(f"  过滤能耗为0: {len(df):,} 行")
    
    # 时段编码：转换为小时范围
    hour_mapping = {
        'De 00:00:00 a 05:59:59 h': 0,   # 00:00-05:59
        'De 06:00:00 a 11:59:59 h': 1,   # 06:00-11:59
        'De 12:00:00 a 17:59:59 h': 2,   # 12:00-17:59
        'De 18:00:00 a 23:59:59 h': 3    # 18:00-23:59
    }
    df['hour_code'] = df['Tram_Horari'].map(hour_mapping)
    
    # 经济部门编码
    sector_mapping = {
        'Indústria': 0,
        'Residencial': 1,
        'Serveis': 2,
        'No especificat': 3
    }
    df['sector_code'] = df['Sector_Economic'].map(sector_mapping)
    
    print(f"  ✅ 清洗完成: {len(df):,} 行")
    print(f"  时段分布:\n{df['hour_code'].value_counts().sort_index()}")
    
    return df


# ============================================================
# 3. 按邮编分组
# ============================================================
def group_by_postal_code(df):
    """按邮编分组，返回每个邮编的数据"""
    print("\n" + "=" * 60)
    print("阶段1.3: 按邮编分组")
    print("=" * 60)
    
    postal_codes = sorted(df['Codi_Postal'].unique())
    print(f"  总邮编数: {len(postal_codes)}")
    print(f"  邮编列表: {postal_codes}")
    
    # 统计每个邮编的数据量
    postal_data = {}
    for code in postal_codes:
        postal_df = df[df['Codi_Postal'] == code].copy()
        postal_data[code] = postal_df
        print(f"  {code}: {len(postal_df):,} 行")
    
    print(f"\n  ✅ 分组完成: {len(postal_data)} 个节点")
    
    return postal_data


# ============================================================
# 4. 时序划分
# ============================================================
def time_split(data_dict):
    """按时间顺序划分训练/验证/测试集"""
    print("\n" + "=" * 60)
    print("阶段1.4: 时序划分")
    print("=" * 60)
    
    split_data = {}
    
    for code, df in data_dict.items():
        # 按日期排序
        df = df.sort_values('Data').reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        split_data[code] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        print(f"  {code}: 训练={len(train_df):,} | 验证={len(val_df):,} | 测试={len(test_df):,}")
    
    print(f"\n  ✅ 时序划分完成")
    
    return split_data


# ============================================================
# 5. 归一化（每个节点独立）
# ============================================================
def normalize_node_data(split_data):
    """对每个节点的训练/验证/测试数据进行归一化"""
    print("\n" + "=" * 60)
    print("阶段1.5: 归一化处理")
    print("=" * 60)
    
    normalized_data = {}
    scalers = {}
    
    for code, splits in split_data.items():
        # 只从训练集计算scaler
        train_values = splits['train']['Valor'].values.reshape(-1, 1)
        
        # 创建scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_values)
        scalers[code] = scaler
        
        # 归一化所有集
        normalized_splits = {}
        for split_name in ['train', 'val', 'test']:
            df = splits[split_name].copy()
            values = df['Valor'].values.reshape(-1, 1)
            df['Valor_norm'] = scaler.transform(values).flatten()
            normalized_splits[split_name] = df
        
        normalized_data[code] = normalized_splits
        
        print(f"  {code}: 能耗范围 {splits['train']['Valor'].min():.0f}~{splits['train']['Valor'].max():.0f} → [0,1]")
    
    print(f"\n  ✅ 归一化完成")
    
    return normalized_data, scalers


# ============================================================
# 6. 保存预处理数据
# ============================================================
def save_preprocessed_data(normalized_data, scalers, metadata):
    """保存预处理后的数据"""
    print("\n" + "=" * 60)
    print("阶段1.6: 保存数据")
    print("=" * 60)
    
    # 保存每个节点的数据
    for code, splits in normalized_data.items():
        node_path = os.path.join(DATA_OUTPUT_DIR, f"node_{code}")
        os.makedirs(node_path, exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            file_path = os.path.join(node_path, f"{split_name}.pkl")
            splits[split_name].to_pickle(file_path)
        
        # 保存scaler
        with open(os.path.join(node_path, "scaler.pkl"), 'wb') as f:
            pickle.dump(scalers[code], f)
    
    # 保存元数据
    metadata_path = os.path.join(DATA_OUTPUT_DIR, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"  ✅ 数据保存到: {DATA_OUTPUT_DIR}")
    print(f"  - {len(normalized_data)} 个节点")
    print(f"  - 每个节点包含 train/val/test 三个pkl文件和scaler.pkl")
    print(f"  - 元数据: {metadata_path}")
    
    return


# ============================================================
# 7. 验证函数
# ============================================================
def verify_preprocessed_data():
    """验证预处理后的数据"""
    print("\n" + "=" * 60)
    print("验证预处理数据")
    print("=" * 60)
    
    # 加载元数据
    metadata_path = os.path.join(DATA_OUTPUT_DIR, "metadata.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"\n元数据:")
    print(f"  节点数量: {metadata['num_nodes']}")
    print(f"  窗口大小: {metadata['window_size']}")
    print(f"  预测大小: {metadata['predict_size']}")
    
    # 验证第一个节点
    first_node = metadata['postal_codes'][0]
    node_path = os.path.join(DATA_OUTPUT_DIR, f"node_{first_node}")
    
    print(f"\n验证节点 {first_node}:")
    
    for split_name in ['train', 'val', 'test']:
        file_path = os.path.join(node_path, f"{split_name}.pkl")
        df = pd.read_pickle(file_path)
        print(f"  {split_name}: {len(df)} 行, 能耗范围 {df['Valor'].min():.0f}~{df['Valor'].max():.0f}")
    
    # 验证scaler
    with open(os.path.join(node_path, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    
    # 取第一个测试样本，验证反归一化
    test_df = pd.read_pickle(os.path.join(node_path, "test.pkl"))
    sample_norm = test_df['Valor_norm'].iloc[0]
    sample_orig = test_df['Valor'].iloc[0]
    sample_inv = scaler.inverse_transform([[sample_norm]])[0][0]
    
    print(f"\n反归一化验证:")
    print(f"  归一化值: {sample_norm:.4f}")
    print(f"  原始值: {sample_orig:.0f}")
    print(f"  反归一化: {sample_inv:.0f}")
    print(f"  ✅ 验证通过" if abs(sample_orig - sample_inv) < 0.1 else "  ❌ 验证失败")
    
    return


# ============================================================
# 8. 主函数
# ============================================================
def preprocess_barcelona():
    """主预处理流程"""
    print("\n" + "=" * 60)
    print("巴塞罗那基站能耗数据预处理")
    print("=" * 60)
    print(f"输入路径: {DATA_INPUT_DIR}")
    print(f"输出路径: {DATA_OUTPUT_DIR}")
    print(f"参数: 窗口={WINDOW_SIZE}, 预测={PREDICT_SIZE}")
    print("=" * 60 + "\n")
    
    # 1. 加载数据
    df = load_and_merge_data()
    
    # 2. 清洗数据
    df = clean_data(df)
    
    # 3. 按邮编分组
    postal_data = group_by_postal_code(df)
    
    # 4. 时序划分
    split_data = time_split(postal_data)
    
    # 5. 归一化
    normalized_data, scalers = normalize_node_data(split_data)
    
    # 6. 保存
    metadata = {
        'num_nodes': len(normalized_data),
        'postal_codes': list(normalized_data.keys()),
        'window_size': WINDOW_SIZE,
        'predict_size': PREDICT_SIZE,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'date_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_preprocessed_data(normalized_data, scalers, metadata)
    
    print("\n" + "=" * 60)
    print("✅ 预处理完成！")
    print("=" * 60)
    
    return normalized_data, scalers, metadata


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    # 运行预处理
    normalized_data, scalers, metadata = preprocess_barcelona()
    
    # 验证
    verify_preprocessed_data()