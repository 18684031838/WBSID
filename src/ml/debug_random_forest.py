"""
调试随机森林模型的训练过程
1. 特征分析
2. 超参数调优
3. 学习曲线分析
4. 错误案例分析
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import seaborn as sns

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ml.feature_extractors.statistical import StatisticalExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str, sample_size: int = 1000):
    """加载和预处理数据"""
    logger.info(f"从 {data_path} 加载数据...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 分离正常查询和注入查询
    normal_queries = [item['query'] for item in data if not item['is_injection']]
    injection_queries = [item['query'] for item in data if item['is_injection']]
    
    logger.info(f"原始数据统计:")
    logger.info(f"- 正常查询: {len(normal_queries)}")
    logger.info(f"- 注入查询: {len(injection_queries)}")
    
    # 随机采样
    if len(normal_queries) > sample_size:
        indices = np.random.choice(len(normal_queries), sample_size, replace=False)
        normal_queries = [normal_queries[i] for i in indices]
    if len(injection_queries) > sample_size:
        indices = np.random.choice(len(injection_queries), sample_size, replace=False)
        injection_queries = [injection_queries[i] for i in indices]
    
    # 合并数据
    X = normal_queries + injection_queries
    y = [0] * len(normal_queries) + [1] * len(injection_queries)
    
    return X, y

def extract_features(X, y):
    """提取特征"""
    logger.info("提取特征...")
    extractor = StatisticalExtractor()
    extractor.fit(X)
    features = extractor.transform(X)
    
    logger.info(f"特征矩阵形状: {features.shape}")
    return features, extractor.feature_names

def analyze_features(X, y, feature_names):
    """分析特征重要性和相关性"""
    logger.info("分析特征...")
    
    # 训练一个基础模型来获取特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 计算特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 打印特征重要性
    logger.info("\n特征重要性排序:")
    for f in range(X.shape[1]):
        logger.info(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.title("特征重要性")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # 计算特征相关性
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()
    
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("特征相关性矩阵")
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    
    return importances, indices

def tune_hyperparameters(X, y):
    """超参数调优"""
    logger.info("开始超参数调优...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X, y)
    
    logger.info("\n最佳参数:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"{param}: {value}")
    logger.info(f"最佳F1分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def plot_learning_curve(estimator, X, y):
    """绘制学习曲线"""
    logger.info("绘制学习曲线...")
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='训练集得分')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='验证集得分')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('训练样本数')
    plt.ylabel('F1分数')
    plt.title('学习曲线')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curve.png')

def analyze_errors(model, X, y, X_raw, feature_names):
    """分析错误预测的案例"""
    logger.info("分析错误案例...")
    
    # 获取预测结果
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    # 找出错误预测的索引
    errors = np.where(y_pred != y)[0]
    
    logger.info(f"\n发现 {len(errors)} 个错误预测:")
    for i in errors:
        confidence = y_prob[i].max()
        logger.info(f"\n错误案例 {i+1}:")
        logger.info(f"原始查询: {X_raw[i]}")
        logger.info(f"真实标签: {'注入' if y[i] == 1 else '正常'}")
        logger.info(f"预测标签: {'注入' if y_pred[i] == 1 else '正常'}")
        logger.info(f"预测置信度: {confidence:.2%}")
        
        # 显示该样本的特征值
        logger.info("特征值:")
        for fname, fvalue in zip(feature_names, X[i]):
            logger.info(f"- {fname}: {fvalue:.4f}")

def main():
    """主函数"""
    # 加载数据
    data_path = Path(project_root) / 'data' / 'training_data.json'
    X_raw, y = load_and_preprocess_data(data_path)
    
    # 提取特征
    X, feature_names = extract_features(X_raw, y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(
        X, y, X_raw, test_size=0.2, random_state=42
    )
    
    # 分析特征
    importances, indices = analyze_features(X_train, y_train, feature_names)
    
    # 超参数调优
    best_model, best_params = tune_hyperparameters(X_train, y_train)
    
    # 绘制学习曲线
    plot_learning_curve(best_model, X_train, y_train)
    
    # 在测试集上评估
    logger.info("\n在测试集上评估最佳模型:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 分析错误案例
    analyze_errors(best_model, X_test, y_test, X_raw_test, feature_names)
    
    # 保存最佳模型
    model_path = Path(project_root) / 'models' / 'random_forest_model.joblib'
    import joblib
    joblib.dump(best_model, model_path)
    logger.info(f"\n最佳模型已保存到: {model_path}")

if __name__ == '__main__':
    main()
