"""Unit tests for model training and evaluation"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.sql_injection_middleware.ml_detector import MLDetector
from src.sql_injection_middleware.config import MODEL_CONFIG
from src.ml.feature_extractors.statistical import StatisticalExtractor

class TestModelTraining(unittest.TestCase):
    """Test cases for model training"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test cases"""
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger('test_model_training')
        
        # 加载训练数据
        data_path = Path(project_root) / 'data' / 'training_data.json'
        cls.logger.info(f"从 {data_path} 加载训练数据...")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 将数据分为正常查询和注入查询
            cls.normal_queries = [item['query'] for item in data if not item['is_injection']]
            cls.injection_queries = [item['query'] for item in data if item['is_injection']]
            
            cls.logger.info(f"加载完成：")
            cls.logger.info(f"- 正常查询数量: {len(cls.normal_queries)}")
            cls.logger.info(f"- 注入查询数量: {len(cls.injection_queries)}")
            
            # 为了避免训练时间过长，随机采样一部分数据
            if len(cls.normal_queries) > 1000:
                indices = np.random.choice(len(cls.normal_queries), 1000, replace=False)
                cls.normal_queries = [cls.normal_queries[i] for i in indices]
            if len(cls.injection_queries) > 1000:
                indices = np.random.choice(len(cls.injection_queries), 1000, replace=False)
                cls.injection_queries = [cls.injection_queries[i] for i in indices]
                
            cls.logger.info(f"采样后：")
            cls.logger.info(f"- 正常查询数量: {len(cls.normal_queries)}")
            cls.logger.info(f"- 注入查询数量: {len(cls.injection_queries)}")
            
            # 转换为列表以便后续操作
            cls.normal_queries = list(cls.normal_queries)
            cls.injection_queries = list(cls.injection_queries)
            
        except Exception as e:
            cls.logger.error(f"加载训练数据失败: {str(e)}")
            raise
    
    def test_model_training(self):
        """Test random forest model training and evaluation"""
        self.logger.info("开始模型训练测试...")
        
        try:
            # 1. 准备数据
            self.logger.info("准备训练数据...")
            X = self.normal_queries + self.injection_queries
            y = ([0] * len(self.normal_queries)) + ([1] * len(self.injection_queries))
            
            # 2. 特征提取
            self.logger.info("提取特征...")
            extractor = StatisticalExtractor()
            extractor.fit(X)
            features = extractor.transform(X)
            
            # 3. 划分训练集和测试集
            self.logger.info("划分数据集...")
            X_train, X_test, y_train, y_test = train_test_split(
                features, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            
            # 4. 训练随机森林模型
            self.logger.info("训练随机森林模型...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # 5. 评估模型
            self.logger.info("评估模型性能...")
            y_pred = model.predict(X_test)
            
            # 打印分类报告
            report = classification_report(y_test, y_pred)
            self.logger.info(f"\n分类报告:\n{report}")
            
            # 打印混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            self.logger.info(f"\n混淆矩阵:\n{cm}")
            
            # 计算准确率
            accuracy = (y_pred == y_test).mean()
            self.logger.info(f"准确率: {accuracy:.2%}")
            
            # 验证模型性能达标
            self.assertGreaterEqual(accuracy, 0.8, "模型准确率低于80%")
            
            # 6. 保存模型
            self.logger.info("保存模型...")
            model_dir = Path(project_root) / 'models'
            model_dir.mkdir(exist_ok=True)
            
            model_path = model_dir / 'random_forest_model.joblib'
            joblib.dump(model, model_path)
            self.logger.info(f"模型已保存到: {model_path}")
            
            # 验证模型文件存在
            self.assertTrue(model_path.exists(), "模型文件未成功保存")
            
            # 7. 测试模型加载
            self.logger.info("测试模型加载...")
            loaded_model = joblib.load(model_path)
            y_pred_loaded = loaded_model.predict(X_test)
            
            # 验证加载的模型预测结果一致
            np.testing.assert_array_equal(
                y_pred, y_pred_loaded,
                "加载的模型预测结果与原模型不一致"
            )
            
            # 8. 特征重要性分析
            self.logger.info("\n特征重要性分析:")
            feature_names = extractor.feature_names
            importances = model.feature_importances_
            
            for name, importance in zip(feature_names, importances):
                self.logger.info(f"{name}: {importance:.4f}")
            
            self.logger.info("模型训练测试完成")
            
        except Exception as e:
            self.logger.error(f"模型训练测试失败: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main(verbosity=2)
