"""
SQL语义特征提取器
使用SQL专用分词和语义标签进行特征提取
"""
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import defaultdict
from .base import BaseFeatureExtractor

class SQLSemanticExtractor(BaseFeatureExtractor):
    """SQL语义特征提取器"""
    
    # SQL关键字和符号
    SQL_COMMANDS = {
        'select', 'insert', 'update', 'delete', 'drop', 'create', 'alter', 
        'where', 'from', 'join', 'union', 'group', 'order', 'having',
        'and', 'or', 'not', 'in', 'between', 'like', 'is', 'null',
        'distinct', 'limit', 'offset', 'asc', 'desc'
    }
    
    SQL_SYMBOLS = {
        '=', '<', '>', '<=', '>=', '!=', '<>', '+', '-', '*', '/',
        '(', ')', ',', ';', '--', '/*', '*/', '#', '@', '$',
        '\'', '"', '`', '[', ']', '%'
    }
    
    SQL_FUNCTIONS = {
        'count', 'sum', 'avg', 'min', 'max', 'concat', 'substring',
        'length', 'upper', 'lower', 'trim', 'cast', 'coalesce'
    }
    
    def __init__(self, max_length=128, embedding_dim=8):
        super().__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # 创建词汇表
        self.vocab = list(self.SQL_COMMANDS | self.SQL_SYMBOLS | self.SQL_FUNCTIONS)
        self.vocab_size = len(self.vocab)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        # 初始化语义标签嵌入矩阵
        self.token_embeddings = self._initialize_embeddings()
        self.position_encodings = self._create_position_encodings()
        
    def _initialize_embeddings(self) -> np.ndarray:
        """初始化词嵌入矩阵"""
        # 为每个词创建一个10维的词向量和1维的语义标签
        embeddings = np.random.normal(0, 0.1, (self.vocab_size, self.embedding_dim))
        
        # 根据词的类型设置语义标签
        for word, idx in self.word2idx.items():
            if word in self.SQL_COMMANDS:
                embeddings[idx, -1] = 1  # 命令
            elif word in self.SQL_SYMBOLS:
                embeddings[idx, -1] = 0  # 符号
            else:
                embeddings[idx, -1] = -1  # 函数
        
        return embeddings
        
    def _create_position_encodings(self) -> np.ndarray:
        """创建固定的正弦位置编码"""
        # 计算位置编码
        position = np.arange(self.max_length)[:, np.newaxis]  # shape: (max_length, 1)
        
        # 确保能够平均分配维度
        dim = self.embedding_dim
        if dim % 2 != 0:
            dim = dim - 1  # 如果是奇数维度，则减1以确保能够平均分配
        
        # 创建维度指数
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        
        # 初始化位置编码矩阵
        pe = np.zeros((self.max_length, self.embedding_dim))
        
        # 计算正弦和余弦值
        pe_sin = np.sin(position * div_term)  # shape: (max_length, dim/2)
        pe_cos = np.cos(position * div_term)  # shape: (max_length, dim/2)
        
        # 交错填充正弦和余弦值
        pe[:, 0:dim:2] = pe_sin
        pe[:, 1:dim:2] = pe_cos
        
        # 如果原始维度是奇数，最后一列使用正弦值
        if self.embedding_dim % 2 != 0:
            last_dim = np.sin(position * div_term[:1])  # 使用第一个频率
            pe[:, -1] = last_dim.flatten()
        
        return pe
    
    def _tokenize(self, query: str) -> List[str]:
        """SQL专用分词器"""
        # 转换为小写
        query = query.lower()
        
        # 替换注释
        query = re.sub(r'--.*$', ' -- ', query, flags=re.MULTILINE)  # 单行注释
        query = re.sub(r'/\*.*?\*/', ' /* */ ', query, flags=re.DOTALL)  # 多行注释
        
        # 分割符号
        for symbol in sorted(self.SQL_SYMBOLS, key=len, reverse=True):
            query = query.replace(symbol, f' {symbol} ')
        
        # 分词
        tokens = query.split()
        
        # 只保留SQL关键字、符号和函数
        tokens = [token for token in tokens if token in self.word2idx]
        
        return tokens[:self.max_length]  # 截断到最大长度
        
    def fit(self, X, y=None):
        """拟合特征提取器
        
        Args:
            X: array-like of shape (n_samples,)
            y: array-like of shape (n_samples,), optional
            
        Returns:
            self
        """
        # 不需要训练，直接返回
        self.is_fitted = True
        return self
        
    def transform(self, X) -> np.ndarray:
        """转换数据为特征矩阵
        
        Args:
            X: array-like of shape (n_samples,)
            
        Returns:
            array-like of shape (n_samples, max_length * embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("SQLSemanticExtractor must be fitted before transform")
        
        # 预分配结果数组
        n_samples = len(X)
        features = np.zeros((n_samples, self.max_length * self.embedding_dim))
        
        # 批量处理样本
        batch_size = 1000
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]
            batch_features = []
            
            # 并行处理每个查询
            for query in batch:
                # 分词
                tokens = self._tokenize(query)
                
                # 转换为索引
                token_ids = np.zeros(self.max_length, dtype=np.int32)
                for j, token in enumerate(tokens[:self.max_length]):
                    if token in self.word2idx:
                        token_ids[j] = self.word2idx[token]
                
                # 获取词嵌入
                token_embeddings = self.token_embeddings[token_ids]
                
                # 添加位置编码
                final_embeddings = token_embeddings + self.position_encodings
                
                # 展平为一维向量
                batch_features.append(final_embeddings.flatten())
            
            # 将批次结果存入预分配的数组
            features[i:i+len(batch)] = np.array(batch_features)
            
            # 打印进度
            if (i + batch_size) % 5000 == 0:
                print(f"Processed {i + batch_size}/{n_samples} samples")
        
        return features
        
    @property
    def n_features_out(self) -> int:
        """输出特征数量"""
        return self.max_length * self.embedding_dim
