"""
下载巴塞罗那天气数据
使用 Open-Meteo 免费 API
"""

import requests
import pandas as pd
import os
from datetime import datetime

LAT = 41.3851
LON = 2.1734
START_DATE = "2019-01-01"
END_DATE = "2025-12-31"

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
    "timezone": "Europe/Madrid"
}

print("=" * 60)
print("下载巴塞罗那天气数据")
print(f"坐标: {LAT}, {LON}")
print(f"时间: {START_DATE} ~ {END_DATE}")
print("=" * 60)

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    
    df.set_index('time', inplace=True)
    df_6h = df.resample('6H').agg({
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'precipitation': 'sum',
        'wind_speed_10m': 'mean'
    }).reset_index()
    
    os.makedirs('data/raw/weather', exist_ok=True)
    output_path = 'data/raw/weather/barcelona_weather_6h.csv'
    df_6h.to_csv(output_path, index=False)
    
    print(f"\n✅ 天气数据已保存: {output_path}")
    print(f"  记录数: {len(df_6h)}")
    print(f"  时间范围: {df_6h['time'].min()} ~ {df_6h['time'].max()}")
    
else:
    print(f"❌ 下载失败: {response.status_code}")
    print(response.text)
