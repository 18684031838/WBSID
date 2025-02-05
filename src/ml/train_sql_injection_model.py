"""
训练SQL注入检测模型 - 优化版本
使用TF-IDF和统计特征进行分类
"""
import os
import sys
import json
import logging
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Tuple

# 导入优化后的模型和特征提取器
from models.model_factory import ModelFactory
from feature_extractors.word2vec import Word2VecExtractor
from feature_extractors.sql_semantic import SQLSemanticExtractor
from feature_extractors.statistical import StatisticalExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def load_training_data(data_path: str, batch_size: int = 1000) -> tuple:
    """加载训练数据（优化版本：批量加载）"""
    logger.info(f"Loading training data from {data_path}")
    
    try:
        queries = []
        labels = []
        
        # 使用批量处理读取数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_batches = (len(data) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch = data[start_idx:end_idx]
            
            batch_queries = []
            batch_labels = []
            
            for item in batch:
                query = item['query']
                label = 1 if item['is_injection'] else 0
                batch_queries.append(query)
                batch_labels.append(label)
            
            queries.extend(batch_queries)
            labels.extend(batch_labels)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Loaded {batch_idx + 1}/{total_batches} batches")
        
        labels = np.array(labels)
        logger.info(f"Loaded {len(queries)} samples")
        logger.info(f"Positive samples: {sum(labels)}")
        logger.info(f"Negative samples: {len(labels) - sum(labels)}")
        
        return queries, labels
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}", exc_info=True)
        raise

def extract_features(queries: List[str], labels: List[int], batch_size: int = 1000) -> Tuple[np.ndarray, Tuple[SQLSemanticExtractor, StatisticalExtractor]]:
    """从查询中提取特征

    Args:
        queries: 查询列表
        labels: 标签列表
        batch_size: 批处理大小

    Returns:
        特征矩阵和特征提取器元组
    """
    logger.info("Extracting features")
    
    # 创建特征提取器
    semantic_extractor = SQLSemanticExtractor(
        max_length=128,  # 最大序列长度
        embedding_dim=8   # 嵌入维度
    )
    statistical_extractor = StatisticalExtractor()
    
    # 提取语义特征
    logger.info("Extracting SQL semantic features")
    semantic_extractor.fit(queries, labels)
    semantic_features = semantic_extractor.transform(queries)
    logger.info(f"Semantic features shape: {semantic_features.shape}")
    
    # 提取统计特征
    logger.info("Extracting statistical features")
    statistical_extractor.fit(queries, labels)
    statistical_features = statistical_extractor.transform(queries)
    logger.info(f"Statistical features shape: {statistical_features.shape}")
    
    # 组合特征
    combined_features = np.hstack([semantic_features, statistical_features])
    logger.info(f"Combined features shape: {combined_features.shape}")
    
    # 返回特征和提取器
    return combined_features, (semantic_extractor, statistical_extractor)

def evaluate_model(model, X_test, y_test, model_name: str = ""):
    """评估模型性能"""
    logger.info(f"\n==================== Evaluating {model_name} ====================")
    
    # 预测并生成分类报告
    y_pred = model.predict(X_test)
    
    logger.info("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    logger.info(f"\n{report}")
    
    # 混淆矩阵
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n{cm}")
    
    # 测试推理速度
    logger.info("\nInference Speed Test:")
    num_samples = 1000
    X_speed_test = X_test[:num_samples]
    
    start_time = time.time()
    model.predict(X_speed_test)
    inference_time = time.time() - start_time
    
    if inference_time > 0:
        samples_per_second = num_samples / inference_time
        avg_time_per_sample = (inference_time / num_samples) * 1000  # 转换为毫秒
    else:
        samples_per_second = float('inf')
        avg_time_per_sample = 0
    
    logger.info(f"Time to process {num_samples} samples: {inference_time:.3f} seconds")
    logger.info(f"Samples per second: {samples_per_second:.1f}")
    logger.info(f"Average time per sample: {avg_time_per_sample:.3f} ms")
    
    return model_name, {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': cm,
        'inference_time': inference_time,
        'samples_per_second': samples_per_second,
        'avg_time_per_sample': avg_time_per_sample
    }

def train_model(model_type: str, data_path: str, output_dir: str, batch_size: int = 1000):
    """训练指定类型的模型
    
    Args:
        model_type: 模型类型，可选值：'svm', 'random_forest', 'decision_tree', 'logistic_regression', 'cnn'
        data_path: 训练数据路径
        output_dir: 模型输出目录
        batch_size: 批处理大小
    """
    try:
        # 加载数据
        logger.info(f"Loading training data from {data_path}")
        queries, labels = load_training_data(data_path, batch_size)
        logger.info(f"Loaded {len(queries)} samples")
        logger.info(f"Positive samples: {sum(labels)}")
        logger.info(f"Negative samples: {len(labels) - sum(labels)}")
        
        # 提取特征
        logger.info("Extracting features")
        X, extractor = extract_features(queries, labels, batch_size)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # 创建模型
        model_params = {'input_dim': X.shape[1]} if model_type == 'cnn' else {}
        logger.info(f"Training {model_type} model")
        model = ModelFactory.create_model(model_type, **model_params)
        
        # 训练模型
        model.train(X_train, y_train)
        
        # 评估模型
        result = evaluate_model(model, X_test, y_test, model_type)
        
        # 保存模型和特征提取器
        logger.info("Saving model and feature extractor")
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = str(project_root / "models" / f"{model_type}_model")  # 不加扩展名，由模型自己处理
        semantic_path = str(project_root / "models" / "sql_semantic_extractor.joblib")
        statistical_path = str(project_root / "models" / "statistical_extractor.joblib")
        
        # 创建models目录（如果不存在）
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model.save(model_path)
        logger.info(f"Model saved to {model_path}{model.model_extension}")
        
        # 保存特征提取器
        joblib.dump(extractor[0], semantic_path)
        logger.info(f"SQL semantic extractor saved to {semantic_path}")
        joblib.dump(extractor[1], statistical_path)
        logger.info(f"Statistical extractor saved to {statistical_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

def train_and_evaluate_all_models(data_path: str, output_dir: str, batch_size: int = 1000):
    """训练和评估所有模型"""
    logger.info("Training and comparing all models...")
    
    # 加载数据
    queries, labels = load_training_data(data_path, batch_size)
    
    # 提取特征
    logger.info("Extracting features")
    X, extractor = extract_features(queries, labels, batch_size)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    # 训练和评估每个模型
    models = ModelFactory.get_available_models()
    results = []
    
    for model_type in models:
        logger.info("\n==================================================")
        model_params = {'input_dim': X.shape[1]} if model_type == 'cnn' else {}
        logger.info(f"Training {model_type} model")
        model = ModelFactory.create_model(model_type, **model_params)
        model.train(X_train, y_train)
        
        # 评估模型
        result = evaluate_model(model, X_test, y_test, model_type)
        results.append(result)
        
        # 保存模型
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = str(project_root / "models" / f"{model_type}_model")  # 不加扩展名，由模型自己处理
        model.save(model_path)
        logger.info(f"Model saved to {model_path}{model.model_extension}")
        
    return results

def evaluate_on_test_set(model, semantic_extractor, statistical_extractor):
    """在测试集上评估模型"""
    try:
        # 加载测试数据
        with open('data/test_data.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_queries = [item['query'] for item in test_data]
        test_labels = [1 if item['is_injection'] else 0 for item in test_data]
        
        # 提取特征
        semantic_features = semantic_extractor.transform(test_queries)
        statistical_features = statistical_extractor.transform(test_queries)
        X_test = np.hstack([semantic_features, statistical_features])
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(test_labels, y_pred)
        precision = precision_score(test_labels, y_pred)
        recall = recall_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred)
        
        # 打印评估结果
        print("\n测试集评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(test_labels, y_pred))
        
        # 打印混淆矩阵
        print("\n混淆矩阵:")
        print(confusion_matrix(test_labels, y_pred))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        logger.error(f"Error during test set evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SQL injection detection model')
    parser.add_argument('--model', type=str, choices=['svm', 'random_forest', 'decision_tree', 'logistic_regression', 'cnn'],
                      help='Model type to train')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Batch size for training')
    parser.add_argument('--compare-all', action='store_true',
                      help='Train and compare all models')
    
    args = parser.parse_args()
    
    # 设置项目根目录和输出目录
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "training_data.json"
    test_data_path = project_root / "data" / "test_data.json"
    output_dir = project_root / "models"
    
    if args.compare_all:
        # 训练并评估所有模型
        logger.info("Training and comparing all models...")
        results = train_and_evaluate_all_models(str(data_path), str(output_dir), args.batch_size)
        
        # 加载测试数据
        logger.info("\n=== 开始测试集评估 ===")
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            test_queries = [item['query'] for item in test_data]
            test_labels = [1 if item['is_injection'] else 0 for item in test_data]
            
            # 加载特征提取器
            semantic_path = str(project_root / "models" / "sql_semantic_extractor.joblib")
            semantic_extractor = joblib.load(semantic_path)
            statistical_path = str(project_root / "models" / "statistical_extractor.joblib")
            statistical_extractor = joblib.load(statistical_path)
            
            # 测试集评估
            test_results = {}
            model_types = ['svm', 'random_forest', 'decision_tree', 'logistic_regression']
            
            print("\n=== 测试集评估结果 ===")
            for model_type in model_types:
                print(f"\n测试模型: {model_type}")
                model_path = str(project_root / "models" / f"{model_type}_model.joblib")
                try:
                    model = joblib.load(model_path)
                    results = evaluate_on_test_set(model, semantic_extractor, statistical_extractor)
                    test_results[model_type] = results
                except Exception as e:
                    logger.error(f"Error evaluating {model_type} model: {str(e)}")
            
            # 找出最佳模型
            if test_results:
                best_model = max(test_results.items(), key=lambda x: x[1]['f1'])
                print(f"\n最佳模型: {best_model[0]}")
                print(f"F1分数: {best_model[1]['f1']:.4f}")
            
        except FileNotFoundError:
            logger.error(f"Test data file not found: {test_data_path}")
        except Exception as e:
            logger.error(f"Error during test set evaluation: {str(e)}")
    else:
        # 训练单个模型
        if not args.model:
            parser.error("Please specify a model type with --model")
        
        logger.info(f"Training {args.model} model...")
        train_model(args.model, str(data_path), str(output_dir), args.batch_size)
