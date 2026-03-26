"""
巴塞罗那能耗数据加载器 - 完美复现版
- 加载 2019-2025 数据
- 按邮编区划分节点
- 时间顺序 70/10/20 划分
- 用训练集的 mean/std 归一化所有数据
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

class BarcelonaEnergyLoader:
    def __init__(self, data_path, years=None, num_nodes=5, seq_length=24, pred_length=6,
                 filter_sectors=None, random_seed=42):
        self.data_path = data_path
        self.years = years or ['2019', '2020', '2021', '2022', '2023', '2024', '2025']
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.filter_sectors = filter_sectors or ['Residencial', 'Serveis']
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    def load_all_data(self):
        """加载所有年份数据并合并"""
        all_dfs = []
        
        for year in self.years:
            patterns = [
                f'{year}_consum_electricitat_bcn.csv',
                f'{year}_consum_electricitat_BCN.csv'
            ]
            
            loaded = False
            for pattern in patterns:
                file_path = os.path.join(self.data_path, pattern)
                if os.path.exists(file_path):
                    print(f"  加载 {year}年数据: {pattern}")
                    df = pd.read_csv(file_path)
                    
                    # 列名统一
                    df.columns = [c.strip() for c in df.columns]
                    if 'time' not in df.columns and 'datetime' in df.columns:
                        df['time'] = df['datetime']
                    
                    all_dfs.append(df)
                    loaded = True
                    break
            
            if not loaded:
                print(f"  ⚠️ 未找到 {year}年数据，跳过")
        
        if not all_dfs:
            raise FileNotFoundError(f"在 {self.data_path} 中未找到任何数据文件")
        
        # 合并所有年份
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ 总数据量: {len(df):,} 行")
        
        # 筛选部门
        if self.filter_sectors and 'sector' in df.columns:
            df = df[df['sector'].isin(self.filter_sectors)]
            print(f"  筛选后部门: {self.filter_sectors}")
        
        # 按时间排序
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            
        print(f"✅ 预处理完成，剩余 {len(df):,} 行")
        print(f"  邮编区数量: {df['postcode'].nunique() if 'postcode' in df.columns else '未知'}")
        
        return df
    
    def prepare_federated_data(self):
        """
        准备联邦学习数据
        返回: dict
            node_id: {
                'X_train': np.array (samples, seq_len, 1),
                'y_train': np.array (samples, pred_len),
                'X_val': ...,
                'y_val': ...,
                'X_test': ...,
                'y_test': ...,
                'mean': float,
                'std': float
            }
        """
        df = self.load_all_data()
        
        # 获取所有邮编区
        if 'postcode' in df.columns:
            postcodes = df['postcode'].unique()
            np.random.shuffle(postcodes)
            selected_postcodes = postcodes[:self.num_nodes]
            print(f"🔀 切分成 {self.num_nodes} 个节点...")
            for i, pc in enumerate(selected_postcodes):
                print(f"  节点 {i}: 邮编 {pc}, {len(df[df['postcode']==pc]):,} 行")
        else:
            # 如果没有邮编列，随机切分
            indices = np.arange(len(df))  # ✅ 保持时间顺序
            split_size = len(df) // self.num_nodes
            selected_postcodes = [f"node_{i}" for i in range(self.num_nodes)]
            print(f"🔀 随机切分成 {self.num_nodes} 个节点...")
        
        fed_data = {}
        
        for node_idx, pc in enumerate(selected_postcodes):
            node_id = f"node_{node_idx}"
            
            # 获取该节点数据
            if 'postcode' in df.columns:
                node_df = df[df['postcode'] == pc].copy()
            else:
                start = node_idx * split_size
                end = (node_idx + 1) * split_size if node_idx < self.num_nodes - 1 else len(df)
                node_df = df.iloc[start:end].copy()
            
            # 取功耗列
            value_col = 'Valor'  # 直接用西班牙语列名
            if value_col not in node_df.columns:
                value_col = node_df.select_dtypes(include=[np.number]).columns[0]
            
            values = node_df[value_col].values.astype(np.float32)
            
            # 时间顺序划分 70/10/20
            n = len(values)
            train_end = int(n * 0.7)
            val_end = int(n * 0.8)
            
            train_raw = values[:train_end]
            val_raw = values[train_end:val_end]
            test_raw = values[val_end:]
            
            # 计算训练集的 mean/std
            train_mean = train_raw.mean()
            train_std = train_raw.std() + 1e-8
            
            # 用训练集的 mean/std 归一化所有数据
            train_norm = (train_raw - train_mean) / train_std
            val_norm = (val_raw - train_mean) / train_std
            test_norm = (test_raw - train_mean) / train_std
            
            # 创建序列
            def create_sequences(data):
                X, y = [], []
                for i in range(len(data) - self.seq_length - self.pred_length + 1):
                    X.append(data[i:i+self.seq_length])
                    y.append(data[i+self.seq_length:i+self.seq_length+self.pred_length])
                return np.array(X).reshape(-1, self.seq_length, 1), np.array(y)
            
            X_train, y_train = create_sequences(train_norm)
            X_val, y_val = create_sequences(val_norm)
            X_test, y_test = create_sequences(test_norm)
            
            print(f"✅ 节点 {node_id} 准备完成: "
                  f"训练 {len(X_train)} 序列, "
                  f"验证 {len(X_val)} 序列, "
                  f"测试 {len(X_test)} 序列")
            
            fed_data[node_id] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'mean': train_mean,
                'std': train_std
            }
        
        return fed_data