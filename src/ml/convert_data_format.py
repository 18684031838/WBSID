"""
将Excel格式的SQL注入数据集转换为JSON格式
"""
import pandas as pd
import json
from pathlib import Path
import os

def convert_excel_to_json(excel_path: str, json_path: str):
    """
    将Excel数据集转换为JSON格式
    
    Args:
        excel_path: Excel文件路径
        json_path: 输出JSON文件路径
    """
    print(f"Reading Excel file from: {excel_path}")
    
    # 读取Excel文件
    df = pd.read_excel(excel_path)
    print(f"Total samples: {len(df)}")
    
    # 转换标签为整数类型
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)
    
    # 统计标签分布
    label_counts = df['Label'].value_counts()
    print("\nLabel distribution:")
    print(f"Normal queries (0): {label_counts.get(0, 0)}")
    print(f"SQL injections (1): {label_counts.get(1, 0)}")
    
    # 转换为目标格式
    json_data = []
    for _, row in df.iterrows():
        json_data.append({
            'query': str(row['Sentence']),  # 确保查询是字符串类型
            'is_injection': bool(row['Label'])  # 转换为布尔值
        })
    
    # 创建输出目录
    output_dir = Path(json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nData converted and saved to: {json_path}")
    print(f"Total records in JSON: {len(json_data)}")

if __name__ == '__main__':
    # 设置输入输出路径
    excel_path = os.path.join(os.path.dirname(__file__), '../../data/kaggle/SQLiV3.xlsx')
    json_path = os.path.join(os.path.dirname(__file__), 'data/training_data.json')
    
    # 转换数据
    convert_excel_to_json(excel_path, json_path)
