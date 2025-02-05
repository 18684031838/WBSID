import unittest
import numpy as np
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ml.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        """测试前的初始化工作"""
        self.extractor = FeatureExtractor(vector_size=100, window_size=5)
        
    def test_tokenize_query_basic(self):
        """测试基本的SQL查询分词"""
        query = "SELECT * FROM users WHERE id = 1"
        tokens = self.extractor.tokenize_query(query)
        expected = ['SELECT', '*', 'FROM', 'users', 'WHERE', 'id', '=', 'NUMBER']
        self.assertEqual(tokens, expected)
        
    def test_tokenize_query_with_strings(self):
        """测试包含字符串的SQL查询分词"""
        query = "INSERT INTO users (name) VALUES ('John Doe')"
        tokens = self.extractor.tokenize_query(query)
        expected = ['INSERT', 'INTO', 'users', '(', 'name', ')', 'VALUES', '(', 'STRING', ')']
        self.assertEqual(tokens, expected)
        
    def test_tokenize_query_with_comments(self):
        """测试包含注释的SQL查询分词"""
        query = "SELECT * FROM users; -- Get all users\n/* Multi-line\ncomment */"
        tokens = self.extractor.tokenize_query(query)
        self.assertIn('COMMENT', tokens)
        self.assertIn('SELECT', tokens)
        self.assertIn('FROM', tokens)
        
    def test_tokenize_query_with_operators(self):
        """测试包含各种运算符的SQL查询分词"""
        query = "SELECT * FROM users WHERE age >= 18 AND salary <= 50000"
        tokens = self.extractor.tokenize_query(query)
        self.assertIn('>=', tokens)
        self.assertIn('<=', tokens)
        self.assertIn('NUMBER', tokens)
        
    def test_tokenize_query_with_injection(self):
        """测试包含SQL注入模式的查询分词"""
        query = "SELECT * FROM users WHERE id = 1 OR 1=1; --"
        tokens = self.extractor.tokenize_query(query)
        expected = ['SELECT', '*', 'FROM', 'users', 'WHERE', 'id', '=', 
                   'NUMBER', 'OR', 'NUMBER', '=', 'NUMBER', 'COMMENT']
        self.assertEqual(tokens, expected)
        
    def test_feature_extraction_workflow(self):
        """测试完整的特征提取工作流程"""
        # 准备测试数据
        queries = [
            "SELECT * FROM users WHERE id = 1",
            "INSERT INTO logs VALUES ('test')",
            "UPDATE users SET name = 'John' WHERE id = 2"
        ]
        
        # 训练模型
        self.extractor.train_word2vec(queries)
        
        # 测试单个查询的特征提取
        query = "SELECT * FROM users WHERE id = 1"
        feature_vector = self.extractor.generate_query_vector(query)
        
        # 验证特征向量的维度
        self.assertEqual(len(feature_vector), self.extractor.vector_size)
        self.assertTrue(isinstance(feature_vector, np.ndarray))
        
        # 测试批量特征提取
        features = self.extractor.extract_features(queries)
        self.assertEqual(features.shape, (len(queries), self.extractor.vector_size))
        
    def test_model_save_load(self):
        """测试模型的保存和加载功能"""
        # 准备测试数据并训练模型
        queries = [
            "SELECT * FROM users WHERE id = 1",
            "INSERT INTO logs VALUES ('test')"
        ]
        self.extractor.train_word2vec(queries)
        
        # 保存模型
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model_path = tmp.name
            self.extractor.save_model(model_path)
            
            # 创建新的特征提取器并加载模型
            new_extractor = FeatureExtractor()
            new_extractor.load_model(model_path)
            
            # 验证两个模型生成的特征向量是否相同
            query = "SELECT * FROM users"
            vector1 = self.extractor.generate_query_vector(query)
            vector2 = new_extractor.generate_query_vector(query)
            np.testing.assert_array_almost_equal(vector1, vector2)
            
        # 清理临时文件
        os.unlink(model_path)

if __name__ == '__main__':
    unittest.main()
