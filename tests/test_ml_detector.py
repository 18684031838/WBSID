"""Unit tests for MLDetector"""
import unittest
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.sql_injection_middleware.ml_detector import MLDetector
from src.sql_injection_middleware.config import MODEL_CONFIG

class TestMLDetector(unittest.TestCase):
    """Test cases for MLDetector class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test cases"""
        cls.detector = MLDetector()
        
        # 正常SQL查询样例
        cls.normal_queries = [
            "SELECT * FROM users WHERE id = 1",
            "INSERT INTO logs (user_id, action) VALUES (1, 'login')",
            "UPDATE users SET last_login = NOW() WHERE id = 1",
            "DELETE FROM sessions WHERE expired = true",
        ]
        
        # SQL注入样例
        cls.injection_queries = [
            "SELECT * FROM users WHERE id = 1 OR 1=1",  # 简单的布尔注入
            "SELECT * FROM users WHERE id = 1; DROP TABLE users;--",  # 多语句注入
            "SELECT * FROM users WHERE id = 1 UNION SELECT username, password FROM admin--",  # 联合查询注入
            "SELECT * FROM users WHERE id = '1' OR '1'='1'",  # 引号闭合注入
            "SELECT * FROM users WHERE id = 1 /*comment*/ OR 1=1",  # 注释注入
            "' OR '1'='1",  # 简单的认证绕过
            "admin' --",  # 注释认证绕过
            "1'; EXEC xp_cmdshell('net user');--",  # 命令执行
            "1' UNION SELECT null, LOAD_FILE('/etc/passwd'),null,null--",  # 文件读取
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",  # 时间盲注
        ]
        
        # 边界情况样例
        cls.edge_cases = [
            "",  # 空字符串
            " ",  # 空格
            "null",  # null值
            "undefined",  # undefined
            "SELECT * FROM users WHERE id = 1 " + "A" * 10000,  # 超长查询
            "SELECT/**/*/**/FROM/**/users",  # 异常注释
            "SeLeCtT * FrOm UsErS",  # 大小写混合
            "&#x53;ELECT *",  # HTML编码
            "%53ELECT *",  # URL编码
            "SLEEP(5)-- ",  # 单独的SQL函数
        ]
        
    def test_normal_queries(self):
        """Test detection of normal SQL queries"""
        print("\nTesting normal SQL queries...")
        
        for query in self.normal_queries:
            request_data = {"query": query}
            is_injection, confidence = self.detector.detect(request_data)
            
            print(f"\nQuery: {query}")
            print(f"Detection result: {'Injection' if is_injection else 'Normal'}")
            print(f"Confidence: {confidence:.2%}")
            
            self.assertFalse(is_injection, f"误报: 正常查询被识别为注入: {query}")
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_injection_queries(self):
        """Test detection of SQL injection attacks"""
        print("\nTesting SQL injection attacks...")
        
        for query in self.injection_queries:
            request_data = {"query": query}
            is_injection, confidence = self.detector.detect(request_data)
            
            print(f"\nQuery: {query}")
            print(f"Detection result: {'Injection' if is_injection else 'Normal'}")
            print(f"Confidence: {confidence:.2%}")
            
            self.assertTrue(is_injection, f"漏报: SQL注入未被检测到: {query}")
            self.assertGreaterEqual(confidence, MODEL_CONFIG['confidence_threshold'])
    
    def test_edge_cases(self):
        """Test edge cases"""
        print("\nTesting edge cases...")
        
        for query in self.edge_cases:
            request_data = {"query": query}
            try:
                is_injection, confidence = self.detector.detect(request_data)
                
                print(f"\nQuery: {query}")
                print(f"Detection result: {'Injection' if is_injection else 'Normal'}")
                print(f"Confidence: {confidence:.2%}")
                
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)
            except Exception as e:
                self.fail(f"Edge case导致异常: {query}\nError: {str(e)}")
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        print("\nTesting feature extraction...")
        
        # 测试样本
        test_query = "SELECT * FROM users WHERE id = 1"
        request_data = {"query": test_query}
        
        # 预处理
        query = self.detector.preprocess_request(request_data)
        self.assertIsInstance(query, str)
        print(f"Preprocessed query: {query}")
        
        # 特征提取
        features = self.detector.extract_features(query)
        self.assertIsInstance(features, np.ndarray)
        print(f"Extracted features shape: {features.shape}")
        
        # 检查特征值范围
        self.assertTrue(np.all(np.isfinite(features)))
        print("All feature values are finite")
        
        # 检查是否有缺失值
        self.assertFalse(np.any(np.isnan(features)))
        print("No missing values in features")
    
    def test_model_confidence(self):
        """Test model confidence scores"""
        print("\nTesting model confidence scores...")
        
        # 测试置信度分布
        all_queries = (
            [(q, False) for q in self.normal_queries] +
            [(q, True) for q in self.injection_queries]
        )
        
        confidences = []
        for query, is_injection in all_queries:
            request_data = {"query": query}
            pred_injection, confidence = self.detector.detect(request_data)
            
            print(f"\nQuery: {query}")
            print(f"Expected: {'Injection' if is_injection else 'Normal'}")
            print(f"Predicted: {'Injection' if pred_injection else 'Normal'}")
            print(f"Confidence: {confidence:.2%}")
            
            confidences.append(confidence)
            
            # 检查高置信度预测的正确性
            if confidence > 0.9:
                self.assertEqual(pred_injection, is_injection,
                    f"高置信度预测错误: {query}\n"
                    f"置信度: {confidence:.2%}\n"
                    f"预期: {'Injection' if is_injection else 'Normal'}\n"
                    f"实际: {'Injection' if pred_injection else 'Normal'}")
        
        # 分析置信度分布
        confidences = np.array(confidences)
        print(f"\nConfidence score statistics:")
        print(f"Mean: {np.mean(confidences):.2%}")
        print(f"Std: {np.std(confidences):.2%}")
        print(f"Min: {np.min(confidences):.2%}")
        print(f"Max: {np.max(confidences):.2%}")
        print(f"Median: {np.median(confidences):.2%}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
