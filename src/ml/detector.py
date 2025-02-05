"""
SQL注入检测模块
整合数据处理、特征提取和模型预测
"""
import numpy as np
from typing import Dict, Optional
from .data_processor import DataProcessor
from .feature_extractor import FeatureExtractor
from .models.model_factory import ModelFactory

class SQLInjectionDetector:
    def __init__(self, 
                 model_type: str = 'svm',
                 vector_size: int = 100,
                 window_size: int = 5,
                 confidence_threshold: float = 0.8,
                 **model_kwargs):
        self.data_processor = DataProcessor()
        self.feature_extractor = FeatureExtractor(
            vector_size=vector_size,
            window_size=window_size
        )
        self.model = ModelFactory.create_model(model_type, **model_kwargs)
        self.confidence_threshold = confidence_threshold
        
    def train(self, training_data: list) -> Dict:
        """
        训练检测器
        Args:
            training_data: List of (query, label) tuples
        Returns:
            评估结果
        """
        # 数据预处理
        processed_queries, labels = self.data_processor.preprocess_data(training_data)
        
        # 训练Word2Vec模型
        self.feature_extractor.train_word2vec(processed_queries)
        
        # 提取特征
        X = self.feature_extractor.extract_features(processed_queries)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = self.data_processor.split_data(X, labels)
        
        # 应用SMOTE
        X_train_balanced, y_train_balanced = self.data_processor.apply_smote(X_train, y_train)
        
        # 训练模型
        self.model.train(X_train_balanced, y_train_balanced)
        
        # 预测测试集
        y_pred = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        # 计算评估指标
        from sklearn.metrics import classification_report, confusion_matrix
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'model_params': self.model.get_params()
        }
    
    def detect(self, query: str) -> Dict:
        """
        检测SQL注入
        Args:
            query: SQL查询语句
        Returns:
            检测结果
        """
        # 预处理查询
        processed_query = self.data_processor.clean_query(query)
        processed_query = self.data_processor.normalize_query(processed_query)
        
        # 提取特征
        features = self.feature_extractor.extract_features([processed_query])
        
        # 预测
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        return {
            'is_injection': bool(prediction),
            'confidence': float(confidence),
            'requires_review': confidence < self.confidence_threshold
        }
    
    def save_models(self, word2vec_path: str, model_path: str) -> None:
        """保存模型"""
        self.feature_extractor.save_model(word2vec_path)
        self.model.save(model_path)
    
    def load_models(self, word2vec_path: str, model_path: str) -> None:
        """加载模型"""
        self.feature_extractor.load_model(word2vec_path)
        self.model.load(model_path)
        
    @staticmethod
    def get_available_models() -> list:
        """获取所有可用的模型类型"""
        return ModelFactory.get_available_models()
