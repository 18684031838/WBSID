import unittest
import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取src目录的路径
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# 将src目录添加到Python路径
sys.path.insert(0, src_dir)

from ml.feature_extractor import FeatureExtractor

class TestSQLTokenizer(unittest.TestCase):
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
        
    def test_tokenize_complex_injection(self):
        """测试复杂的SQL注入模式"""
        query = """SELECT * FROM users WHERE username = 'admin' OR '1'='1' 
                  UNION SELECT null, username, password FROM admin_users--"""
        tokens = self.extractor.tokenize_query(query)
        # 验证关键的注入相关token
        self.assertIn('UNION', tokens)
        self.assertIn('SELECT', tokens)
        self.assertIn('OR', tokens)
        self.assertIn('STRING', tokens)
        self.assertIn('COMMENT', tokens)
        
    def test_tokenize_batch_injection(self):
        """测试批处理SQL注入"""
        query = "SELECT * FROM users; DROP TABLE users; --"
        tokens = self.extractor.tokenize_query(query)
        self.assertIn('DROP', tokens)
        self.assertIn('TABLE', tokens)
        self.assertIn('COMMENT', tokens)

if __name__ == '__main__':
    unittest.main()
