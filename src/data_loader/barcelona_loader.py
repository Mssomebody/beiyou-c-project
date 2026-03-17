"""
巴塞罗那能耗数据加载器 - 适配FedLSTM
从 processed/ 读取2019-2025能耗数据
按邮编区(Codi_Postal)切分成多个节点
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BarcelonaEnergyLoader:
    """巴塞罗那按邮编区能耗数据加载器"""
    
    def __init__(self, 
                 data_path="data/processed",
                 years=None,
                 num_nodes=10,
                 seq_length=24,
                 pred_length=1,
                 val_ratio=0.1,
                 test_ratio=0.2,
                 filter_sectors=None):
        """
        Args:
            data_path: processed数据路径
            years: 要加载的年份，默认2019-2025全部
            num_nodes: 模拟的基站数量（取前num_nodes个邮编区）
            seq_length: 输入序列长度（小时）
            pred_length: 预测长度（小时）
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            filter_sectors: 筛选经济部门，如['Residencial', 'Serveis']，None表示全部
        """
        self.data_path = data_path
        self.years = years or ['2019','2020','2021','2022','2023','2024','2025']
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.filter_sectors = filter_sectors
        
        # 归一化器（每个节点独立）
        self.scalers = {}
        
        # 数据集元信息
        self.metadata = {
            'name': 'Barcelona Electricity Consumption by Postal Code',
            'source': 'Open Data BCN / Datadis',
            'years': self.years,
            'num_nodes': num_nodes,
            'seq_length': seq_length,
            'pred_length': pred_length,
            'features': ['Any', 'Data', 'Codi_Postal', 'Sector_Economic', 'Tram_Horari', 'Valor'],
            'target': 'Valor',
            'total_size_mb': self._get_total_size()
        }
        
    def _get_total_size(self):
        """计算总数据大小（MB）"""
        total_bytes = 0
        for year in self.years:
            file_path = os.path.join(self.data_path, f'{year}_consum_electricitat_*.csv')
            for f in glob(file_path):
                total_bytes += os.path.getsize(f)
        return round(total_bytes / (1024*1024), 2)
    
    def load_all_data(self):
        """加载所有年份数据并合并"""
        print(f"📊 加载巴塞罗那能耗数据 ({self.metadata['total_size_mb']} MB)...")
        
        all_dfs = []
        for year in self.years:
            # 匹配可能的文件名格式（BCN大写/小写）
            file_patterns = [
                f'{year}_consum_electricitat_bcn.csv',
                f'{year}_consum_electricitat_BCN.csv'
            ]
            
            loaded = False
            for pattern in file_patterns:
                file_path = os.path.join(self.data_path, pattern)
                if os.path.exists(file_path):
                    print(f"  加载 {year}年数据: {pattern}")
                    df = pd.read_csv(file_path)
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
        
        return df
    
    def preprocess(self, df):
        """数据预处理"""
        print("🔄 数据预处理...")
        
        # 重命名列（统一为英文）
        column_map = {
            'Any': 'year',
            'Data': 'date',
            'Codi_Postal': 'postal_code',
            'Sector_Economic': 'sector',
            'Tram_Horari': 'time_slot',
            'Valor': 'consumption'
        }
        
        # 只保留需要的列
        df = df[list(column_map.keys())].copy()
        df.rename(columns=column_map, inplace=True)
        
        # 筛选经济部门
        if self.filter_sectors:
            df = df[df['sector'].isin(self.filter_sectors)]
            print(f"  筛选后部门: {self.filter_sectors}")
        
        # 转换日期时间
        df['datetime'] = pd.to_datetime(df['date'])
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        # 处理时间槽（如果需要更细粒度）
        # df['time_slot'] 包含 'De 00:00:00 a 05:59:59 h' 等信息
        
        # 提取时间段开始小时
        def extract_start_hour(time_slot):
            try:
                return int(time_slot.split('De ')[1].split(':')[0])
            except:
                return 0
        
        df['slot_hour'] = df['time_slot'].apply(extract_start_hour)
        
        # 排序
        df = df.sort_values(['postal_code', 'datetime', 'slot_hour'])
        
        print(f"✅ 预处理完成，剩余 {len(df):,} 行")
        print(f"  邮编区数量: {df['postal_code'].nunique()}")
        print(f"  时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        
        return df
    
    def split_by_postal_code(self, df):
        """按邮编区切分成多个节点"""
        print(f"🔀 切分成 {self.num_nodes} 个节点...")
        
        # 获取所有邮编区，按数据量排序
        postal_codes = df['postal_code'].value_counts().index[:self.num_nodes]
        
        node_data = {}
        for i, pc in enumerate(postal_codes):
            node_df = df[df['postal_code'] == pc].copy()
            
            # 按时间重采样到小时（如果需要）
            # 这里假设已经是小时级数据
            
            node_data[f'node_{i}'] = {
                'postal_code': pc,
                'data': node_df,
                'size': len(node_df)
            }
            print(f"  节点 {i}: 邮编 {pc}, {len(node_df):,} 行")
        
        return node_data
    
    def create_sequences(self, data, value_col='consumption'):
        """为LSTM创建输入序列"""
        X, y = [], []
        values = data[value_col].values
        
        for i in range(len(values) - self.seq_length - self.pred_length):
            X.append(values[i:i+self.seq_length])
            y.append(values[i+self.seq_length:i+self.seq_length+self.pred_length])
        
        return np.array(X), np.array(y)
    
    def normalize_node(self, X_train, y_train, X_val=None, X_test=None):
        """归一化（fit on train, transform all）"""
        # 合并X和y一起归一化
        train_concat = np.concatenate([X_train.flatten(), y_train.flatten()])
        
        scaler = MinMaxScaler()
        scaler.fit(train_concat.reshape(-1, 1))
        
        # 转换
        X_train_norm = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        y_train_norm = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        
        result = {
            'X_train': X_train_norm,
            'y_train': y_train_norm,
            'scaler': scaler
        }
        
        if X_val is not None:
            X_val_norm = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
            result['X_val'] = X_val_norm
            
        if X_test is not None:
            X_test_norm = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            result['X_test'] = X_test_norm
            
        return result
    
    def prepare_federated_data(self):
        """准备联邦学习格式的数据"""
        # 1. 加载所有数据
        df = self.load_all_data()
        
        # 2. 预处理
        df = self.preprocess(df)
        
        # 3. 按邮编区分节点
        node_data = self.split_by_postal_code(df)
        
        # 4. 为每个节点创建序列并划分训练/验证/测试
        federated_data = {}
        
        for node_id, info in node_data.items():
            node_df = info['data']
            
            # 按时间排序
            node_df = node_df.sort_values(['datetime', 'slot_hour'])
            
            # 创建序列
            X, y = self.create_sequences(node_df)
            
            # 划分数据集
            n = len(X)
            train_end = int(n * (1 - self.val_ratio - self.test_ratio))
            val_end = int(n * (1 - self.test_ratio))
            
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]
            
            # 归一化
            normed = self.normalize_node(X_train, y_train, X_val, X_test)
            
            # 保存归一化器
            self.scalers[node_id] = normed['scaler']
            
            federated_data[node_id] = {
                'postal_code': info['postal_code'],
                'X_train': normed['X_train'],
                'y_train': normed['y_train'],
                'X_val': normed.get('X_val'),
                'y_val': y_val,  # 原始值（用于评估）
                'X_test': normed.get('X_test'),
                'y_test': y_test,  # 原始值
                'scaler': normed['scaler']
            }
            
            print(f"✅ 节点 {node_id} 准备完成: "
                  f"训练 {len(X_train)} 序列, "
                  f"验证 {len(X_val)} 序列, "
                  f"测试 {len(X_test)} 序列")
        
        return federated_data
    
    def get_node_data(self, node_id, federated_data=None):
        """获取指定节点的数据"""
        if federated_data is None:
            federated_data = self.prepare_federated_data()
        
        return federated_data.get(node_id)
    
    def inverse_transform(self, node_id, values):
        """将归一化后的值转换回原始尺度"""
        if node_id not in self.scalers:
            raise ValueError(f"节点 {node_id} 的归一化器不存在")
        
        scaler = self.scalers[node_id]
        return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


# ============= 测试代码 =============
if __name__ == "__main__":
    print("🔧 测试巴塞罗那数据加载器...\n")
    
    # 初始化加载器
    loader = BarcelonaEnergyLoader(
        data_path="data/processed",
        years=['2019', '2020', '2021', '2022', '2023', '2024', '2025'],
        num_nodes=5,  # 先取5个邮编区测试
        seq_length=24,  # 用过去24小时预测
        pred_length=1,  # 预测下一小时
        filter_sectors=['Residencial', 'Serveis']  # 只保留居民和服务业
    )
    
    # 准备联邦数据
    fed_data = loader.prepare_federated_data()
    
    print("\n📊 数据统计:")
    for node_id, data in fed_data.items():
        print(f"\n节点 {node_id} (邮编 {data['postal_code']}):")
        print(f"  训练集: X {data['X_train'].shape}, y {data['y_train'].shape}")
        print(f"  验证集: X {data['X_val'].shape}")
        print(f"  测试集: X {data['X_test'].shape}")
        print(f"  能耗范围: {data['y_test'].min():.2f} ~ {data['y_test'].max():.2f}")
    
    print("\n✅ 加载器测试完成！")