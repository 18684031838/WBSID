"""
组合特征提取器
整合多个特征提取器的输出
"""
from typing import List, Dict, Any
import numpy as np
from .base import BaseFeatureExtractor
from .statistical import StatisticalExtractor
from .sql_semantic import SQLSemanticExtractor
from .tfidf import TFIDFExtractor
from .word2vec import Word2VecExtractor

class CombinedExtractor(BaseFeatureExtractor):
    """组合多个特征提取器"""
    
    def __init__(self, max_length: int = 128, embedding_dim: int = 128, **kwargs):
        super().__init__()
        semantic_kwargs = {'max_length': max_length, 'embedding_dim': embedding_dim}
        self.extractors = {
            'statistical': StatisticalExtractor(),
            'sql_semantic': SQLSemanticExtractor(**semantic_kwargs),
            'tfidf': TFIDFExtractor(),
            'word2vec': Word2VecExtractor()
        }
        self.enabled_extractors = kwargs.get('enabled_extractors', ['statistical'])
        
    def fit(self, X, y=None):
        """拟合特征提取器
        
        Args:
            X: array-like of shape (n_samples,)
            y: array-like of shape (n_samples,), optional
            
        Returns:
            self
        """
        for name in self.enabled_extractors:
            if name in self.extractors:
                self.extractors[name].fit(X, y)
        self.is_fitted = True
        return self
        
    def transform(self, X) -> np.ndarray:
        """转换数据为特征矩阵
        
        Args:
            X: array-like of shape (n_samples,)
            
        Returns:
            array-like of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("CombinedExtractor must be fitted before transform")
        
        # 提取并组合特征
        features = []
        for name in self.enabled_extractors:
            if name in self.extractors:
                extractor = self.extractors[name]
                feature = extractor.transform(X)
                features.append(feature)
        
        # 水平连接所有特征
        if features:
            return np.hstack(features)
        else:
            raise ValueError("No enabled extractors found")
    
    def save(self, path: str):
        """保存特征提取器
        
        Args:
            path: 保存路径
        """
        import joblib
        state = {
            'extractors': self.extractors,
            'enabled_extractors': self.enabled_extractors,
            'is_fitted': self.is_fitted
        }
        joblib.dump(state, path)
    
    def load(self, path: str):
        """加载特征提取器
        
        Args:
            path: 加载路径
        """
        import joblib
        state = joblib.load(path)
        self.extractors = state['extractors']
        self.enabled_extractors = state['enabled_extractors']
        self.is_fitted = state['is_fitted']
