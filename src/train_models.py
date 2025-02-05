"""训练SQL注入检测模型"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_models')

def load_data(data_dir: str = 'data') -> pd.DataFrame:
    """加载训练数据
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        包含查询和标签的DataFrame
    """
    data_path = Path(data_dir) / 'sql_injection_dataset.csv'
    logger.info(f"正在加载数据: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"数据加载完成，共 {len(df)} 条记录")
    return df

def preprocess_data(df: pd.DataFrame) -> tuple:
    """预处理数据
    
    Args:
        df: 原始数据DataFrame
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    logger.info("正在预处理数据...")
    
    # 分割训练集和测试集
    X = df['query'].values
    y = df['is_injection'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train) -> tuple:
    """训练随机森林模型
    
    Args:
        X_train: 训练数据
        y_train: 训练标签
        
    Returns:
        (模型, 特征提取器)
    """
    logger.info("正在训练模型...")
    
    # 创建特征提取器
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=0.001,
        max_df=0.95
    )
    
    # 提取特征
    X_train_features = vectorizer.fit_transform(X_train)
    
    # 创建并训练模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    model.fit(X_train_features, y_train)
    logger.info("模型训练完成")
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """评估模型性能
    
    Args:
        model: 训练好的模型
        vectorizer: 特征提取器
        X_test: 测试数据
        y_test: 测试标签
    """
    logger.info("正在评估模型...")
    
    # 提取测试集特征
    X_test_features = vectorizer.transform(X_test)
    
    # 预测并计算准确率
    accuracy = model.score(X_test_features, y_test)
    logger.info(f"模型准确率: {accuracy:.4f}")

def save_model(model, vectorizer, output_dir: str = '../models'):
    """保存模型和特征提取器
    
    Args:
        model: 训练好的模型
        vectorizer: 特征提取器
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_dir / 'random_forest_model.joblib'
    logger.info(f"正在保存模型: {model_path}")
    joblib.dump(model, model_path)
    
    # 保存特征提取器
    vectorizer_path = output_dir / 'random_forest_extractors.joblib'
    logger.info(f"正在保存特征提取器: {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)
    
    logger.info("模型和特征提取器保存完成")

def main():
    """主函数"""
    try:
        # 加载数据
        df = load_data()
        
        # 预处理数据
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # 训练模型
        model, vectorizer = train_model(X_train, y_train)
        
        # 评估模型
        evaluate_model(model, vectorizer, X_test, y_test)
        
        # 保存模型
        save_model(model, vectorizer)
        
        logger.info("模型训练流程完成")
        
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
