"""
数据预处理模块
负责数据收集、清洗和预处理
"""
import re
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from urllib.parse import parse_qs, unquote
import json

class DataProcessor:
    def __init__(self):
        self.sql_keywords = set([
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 
            'WHERE', 'AND', 'OR', 'LIKE', 'IN', 'BETWEEN'
        ])
        self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'in', 'a', 'an'])
        self.special_chars = set(['\'', '"', ';', '--', '/*', '*/', '#', '=', '+'])
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=self.stop_words,
            ngram_range=(1, 3)
        )
    
    def preprocess_http_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        HTTP请求数据预处理
        Args:
            request_data: HTTP请求数据
        Returns:
            处理后的请求数据
        """
        # 1. 请求解析阶段
        params = {}
        
        # 1.1 提取参数
        if 'GET' in request_data:
            params.update(parse_qs(request_data['GET']))
        if 'POST' in request_data:
            if isinstance(request_data['POST'], str):
                params.update(parse_qs(request_data['POST']))
            elif isinstance(request_data['POST'], dict):
                params.update(request_data['POST'])
        if 'COOKIES' in request_data:
            params.update(request_data['COOKIES'])
        
        # 1.2 参数标准化
        normalized_params = {}
        for key, value in params.items():
            # 统一编码格式
            if isinstance(value, list):
                value = [unquote(v) if isinstance(v, str) else v for v in value]
            else:
                value = unquote(str(value))
            # 转换小写
            key = key.lower()
            normalized_params[key] = value
        
        # 1.3 参数过滤
        filtered_params = {k: v for k, v in normalized_params.items() 
                         if not k.startswith('_') and k not in ['csrf', 'token']}
        
        return filtered_params

    def clean_text(self, text: str) -> str:
        """
        数据清洗
        Args:
            text: 输入文本
        Returns:
            清洗后的文本
        """
        # 2.1 特殊字符处理
        text = text.replace('\\', '')
        text = unquote(text)
        
        # 2.2 空白处理
        text = re.sub(r'\s+', ' ', text)
        
        # 2.3 长度规范
        if len(text) > 1000:  # 设置最大长度限制
            text = text[:1000]
        
        return text.strip()

    def generate_features(self, text: str, feature_type: str = 'all') -> Dict[str, Any]:
        """
        特征数据生成
        Args:
            text: 输入文本
            feature_type: 特征类型 ('statistical', 'textual', 'all')
        Returns:
            特征字典
        """
        features = {}
        
        # 1. 文本分词
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # 2.1 统计特征
        if feature_type in ['statistical', 'all']:
            features.update({
                'length': len(text),
                'token_count': len(tokens),
                'avg_token_length': np.mean([len(t) for t in tokens]) if tokens else 0,
                'special_char_ratio': sum(1 for c in text if c in self.special_chars) / len(text) if text else 0,
                'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
                'sql_keyword_count': sum(1 for t in tokens if t.upper() in self.sql_keywords)
            })
        
        # 2.2 文本特征
        if feature_type in ['textual', 'all']:
            # 使用TF-IDF特征
            if not hasattr(self, '_tfidf_matrix'):
                self._tfidf_matrix = self.vectorizer.fit_transform([text])
            else:
                self._tfidf_matrix = self.vectorizer.transform([text])
            features['tfidf'] = self._tfidf_matrix
        
        return features

    def clean_query(self, query: str) -> str:
        """清理SQL查询语句"""
        if not isinstance(query, str):
            query = str(query)
        query = query.lower()
        query = re.sub(r'\s+', ' ', query)  # 规范化空白字符
        query = query.strip()
        return query
    
    def normalize_query(self, query: str) -> str:
        """标准化SQL查询语句"""
        # 替换数字为占位符
        query = re.sub(r'\d+', 'NUM', query)
        # 替换字符串为占位符
        query = re.sub(r"'.*?'", 'STR', query)
        return query
    
    def preprocess_data(self, raw_data: List[Tuple[str, int]]) -> Tuple[List[str], np.ndarray]:
        """
        预处理数据集
        Args:
            raw_data: List of (query, label) tuples
        Returns:
            处理后的查询列表和标签数组
        """
        processed_queries = []
        labels = []
        
        for query, label in raw_data:
            cleaned_query = self.clean_query(query)
            normalized_query = self.normalize_query(cleaned_query)
            processed_queries.append(normalized_query)
            labels.append(label)
        
        return processed_queries, np.array(labels)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.25) -> Tuple:
        """划分数据集"""
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用SMOTE处理类别不平衡"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
