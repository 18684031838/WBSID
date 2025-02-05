"""
TF-IDF 特征提取器
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import time
import sys

class TFIDFExtractor:
    """使用 TF-IDF 和特征选择的文本特征提取器"""
    
    def __init__(self, max_features=3000, ngram_range=(1, 3), 
                 variance_threshold=0.005, select_k=1000):
        """
        初始化 TF-IDF 特征提取器
        
        Args:
            max_features: TF-IDF 提取的最大特征数
            ngram_range: n-gram 范围，如 (1, 3) 表示 1-gram 到 3-gram
            variance_threshold: 方差阈值，用于移除低方差特征
            select_k: 选择的最佳特征数量
        """
        print("\nInitializing TF-IDF Extractor:")
        print(f"  - Max features: {max_features}")
        print(f"  - N-gram range: {ngram_range}")
        print(f"  - Variance threshold: {variance_threshold}")
        print(f"  - Select k best features: {select_k}")
        sys.stdout.flush()
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.variance_threshold = variance_threshold
        self.select_k = select_k
        
        # 初始化 TF-IDF 向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        # 初始化特征选择器
        self.var_selector = VarianceThreshold(threshold=variance_threshold)
        self.k_selector = SelectKBest(score_func=f_classif, k=select_k)
        
        self.is_fitted = False
        self.use_k_selector = False  # 标记是否使用 k_selector
    
    def fit(self, X, y=None):
        """
        训练 TF-IDF 特征提取器
        
        Args:
            X: 输入文本数据
            y: 标签数据（可选）
            
        Returns:
            self
        """
        print("\nFitting TF-IDF vectorizer...")
        print(f"Input samples: {len(X)}")
        sys.stdout.flush()
        
        # 1. TF-IDF 向量化
        print("Step 1: TF-IDF vectorization...")
        start_time = time.time()
        X_tfidf = self.vectorizer.fit_transform(X)
        print(f"Vectorization completed in {time.time() - start_time:.2f} seconds")
        print(f"Initial feature matrix shape: {X_tfidf.shape}")
        sys.stdout.flush()
        
        # 2. 方差阈值特征选择
        print(f"\nStep 2: Variance threshold feature selection...")
        print(f"Threshold: {self.variance_threshold}")
        start_time = time.time()
        X_tfidf = self.var_selector.fit_transform(X_tfidf.toarray())
        print(f"Variance selection completed in {time.time() - start_time:.2f} seconds")
        print(f"Features after variance selection: {X_tfidf.shape[1]}")
        sys.stdout.flush()
        
        # 3. 选择最佳特征
        if y is not None and X_tfidf.shape[1] > self.select_k:
            print(f"\nStep 3: Selecting k best features...")
            print(f"K: {self.select_k}")
            start_time = time.time()
            X_tfidf = self.k_selector.fit_transform(X_tfidf, y)
            print(f"K-best selection completed in {time.time() - start_time:.2f} seconds")
            print(f"Final feature matrix shape: {X_tfidf.shape}")
            sys.stdout.flush()
            self.use_k_selector = True
        else:
            self.use_k_selector = False
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        转换输入文本数据
        
        Args:
            X: 输入文本数据
            
        Returns:
            转换后的特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("TFIDFExtractor must be fitted before transform")
        
        # 1. TF-IDF 向量化
        X_tfidf = self.vectorizer.transform(X)
        
        # 2. 方差阈值特征选择
        X_tfidf = self.var_selector.transform(X_tfidf.toarray())
        
        # 3. 选择最佳特征（如果在训练时使用了）
        if self.use_k_selector:
            X_tfidf = self.k_selector.transform(X_tfidf)
        
        return X_tfidf
