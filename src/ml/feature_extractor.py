"""
特征提取模块 - 优化版本
使用SQL解析和Word2Vec进行特征提取，支持批处理和向量化操作
"""
import numpy as np
from gensim.models import Word2Vec
from typing import List, Dict, Any, Tuple, Iterator
import re
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 预定义的SQL关键字列表（使用frozenset提高查找效率）
SQL_KEYWORDS = frozenset({
    'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'UNION', 
    'ALL', 'AND', 'OR', 'AS', 'JOIN', 'ON', 'GROUP', 'BY', 'HAVING', 
    'ORDER', 'VALUES', 'INTO', 'CREATE', 'DROP', 'TABLE', 'INDEX',
    'ALTER', 'ADD', 'SET', 'NULL', 'NOT', 'IN', 'LIKE', 'BETWEEN',
    'IS', 'EXISTS', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
})

# 预编译正则表达式以提高性能
TOKEN_PATTERN = re.compile(r"""
    (?:--[^\n]*)|                # 单行注释
    (?:\/\*[\s\S]*?\*\/)|        # 多行注释
    (?:'[^']*')|                 # 单引号字符串
    (?:"[^"]*")|                 # 双引号字符串
    (?:`[^`]*`)|                 # 反引号字符串
    (?:[<>!=]=?|[-+*/%,.()])|    # 运算符和标点
    (?:\b\w+\b)                  # 字母数字字符
""", re.VERBOSE | re.IGNORECASE)

NUMBER_PATTERN = re.compile(r'^-?\d+\.?\d*$|^0x[0-9a-f]+$', re.I)
OPERATOR_PATTERN = re.compile(r'^[<>!=]=?|[-+*/%]$')
PUNCT_PATTERN = re.compile(r'^[,.()]$')
IDENTIFIER_PATTERN = re.compile(r'^\w+$')

class Word2VecExtractor(BaseEstimator, TransformerMixin):
    """使用Word2Vec进行特征提取的类 - 优化版本"""
    
    def __init__(self, vector_size: int = 100, window_size: int = 5, batch_size: int = 1000):
        self.vector_size = vector_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.model = None
        self.n_jobs = multiprocessing.cpu_count()
    
    def tokenize_query(self, query: str) -> List[str]:
        """将SQL查询进行结构化分词（优化版本）"""
        output_tokens = []
        
        # 使用预编译的正则表达式进行分词
        input_tokens = TOKEN_PATTERN.findall(query)
        
        # 批量处理tokens
        for token in input_tokens:
            token = token.strip()
            if not token:
                continue
                
            token_upper = token.upper()
            
            # 使用预编译的正则表达式和frozenset提高性能
            if token_upper in SQL_KEYWORDS:
                output_tokens.append(token_upper)
            elif NUMBER_PATTERN.match(token):
                output_tokens.append('NUMBER')
            elif OPERATOR_PATTERN.match(token):
                output_tokens.append(token)
            elif (token.startswith(("'", '"')) and token.endswith(("'", '"'))):
                output_tokens.append('STRING')
            elif token.startswith('`') and token.endswith('`'):
                output_tokens.append('BACK_TICK')
            elif token == "'":
                output_tokens.append('SINGLE_QUOTE')
            elif token == '"':
                output_tokens.append('DOUBLE_QUOTE')
            elif IDENTIFIER_PATTERN.match(token):
                output_tokens.append(token)
            elif token.startswith(('--', '/*')):
                output_tokens.append('COMMENT')
            elif PUNCT_PATTERN.match(token):
                output_tokens.append(token)
                
        return output_tokens

    def _batch_tokenize(self, queries: List[str]) -> List[List[str]]:
        """批量分词处理"""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            return list(executor.map(self.tokenize_query, queries))

    def fit(self, X, y=None):
        """训练Word2Vec模型（优化版本）"""
        logging.info("开始训练Word2Vec模型...")
        
        # 使用多线程进行分词处理
        tokenized_queries = []
        for i in tqdm(range(0, len(X), self.batch_size), desc="Tokenizing queries"):
            batch = X[i:i + self.batch_size]
            batch_tokens = self._batch_tokenize(batch)
            tokenized_queries.extend(batch_tokens)
        
        # 训练Word2Vec模型
        self.model = Word2Vec(
            sentences=tokenized_queries,  # 使用列表而不是生成器
            vector_size=self.vector_size,
            window=self.window_size,
            min_count=1,
            workers=self.n_jobs,  # 使用所有CPU核心
            compute_loss=True  # 计算训练损失
        )
        
        logging.info("Word2Vec模型训练完成")
        return self
    
    def transform(self, X):
        """生成查询的特征向量（优化版本）"""
        if self.model is None:
            raise ValueError("Word2Vec model has not been trained yet")
        
        n_samples = len(X)
        features = np.zeros((n_samples, self.vector_size))
        
        for i in tqdm(range(0, n_samples, self.batch_size), desc="Extracting features"):
            batch = X[i:i + self.batch_size]
            batch_tokens = self._batch_tokenize(batch)
            
            for j, tokens in enumerate(batch_tokens):
                vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
                if vectors:
                    features[i + j] = np.mean(vectors, axis=0)
        
        return features
    
    def fit_transform(self, X, y=None):
        """训练并转换（优化版本）"""
        return self.fit(X, y).transform(X)
    
    def save_model(self, path: str) -> None:
        """保存Word2Vec模型"""
        if self.model is not None:
            self.model.save(path)
            logging.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str) -> None:
        """加载Word2Vec模型"""
        self.model = Word2Vec.load(path)
        logging.info(f"模型已从 {path} 加载")


class FeatureExtractorPipeline(BaseEstimator, TransformerMixin):
    """组合多个特征提取器的管道类 - 优化版本"""
    
    def __init__(self, extractors: List[Tuple[str, BaseEstimator]], batch_size: int = 1000):
        self.extractors = extractors
        self.batch_size = batch_size
        self.is_fitted = False
        self._y = None
    
    def fit(self, X, y=None):
        """训练所有特征提取器（优化版本）"""
        logging.info("开始训练特征提取器管道...")
        for name, extractor in self.extractors:
            logging.info(f"训练特征提取器: {name}")
            extractor.fit(X, y)
        self.is_fitted = True
        self._y = y
        logging.info("特征提取器管道训练完成")
        return self
    
    def transform(self, X):
        """使用所有特征提取器转换数据（优化版本）"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractorPipeline has not been fitted yet")
        
        n_samples = len(X)
        features_list = []
        
        for name, extractor in self.extractors:
            logging.info(f"使用特征提取器转换数据: {name}")
            features = []
            
            # 批量处理
            for i in tqdm(range(0, n_samples, self.batch_size), desc=f"Processing {name}"):
                batch = X[i:i + self.batch_size]
                batch_features = extractor.transform(batch)
                features.append(batch_features)
            
            # 合并批次结果
            features = np.vstack(features)
            features_list.append(features)
        
        # 水平连接所有特征
        return np.hstack(features_list) if features_list else np.array([])
    
    def fit_transform(self, X, y=None):
        """训练并转换数据（优化版本）"""
        return self.fit(X, y).transform(X)
