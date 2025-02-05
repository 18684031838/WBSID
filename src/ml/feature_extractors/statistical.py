"""
Statistical feature extractor module
"""
import re
import numpy as np
from .base import BaseFeatureExtractor
import time
from tqdm import tqdm

class StatisticalExtractor(BaseFeatureExtractor):
    """Statistical feature extractor for SQL injection detection"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_name_list = [
            'length',
            'word_count',
            'avg_word_length',
            'special_char_ratio',
            'digit_ratio',
            'uppercase_ratio',
            'space_ratio',
            'keyword_count',
            'function_count',
            'comment_count'
        ]
        
        self.sql_keywords = {
            'select', 'insert', 'update', 'delete', 'drop', 'union',
            'where', 'from', 'join', 'having', 'group'
        }
        
        self.sql_functions = {
            'count', 'max', 'min', 'avg', 'sum', 'concat', 'substring',
            'length', 'upper', 'lower', 'cast', 'convert'
        }
        
        print("\nInitializing Statistical Feature Extractor:")
        print(f"  - Number of features: {len(self.feature_name_list)}")
        print("  - Features:", ", ".join(self.feature_name_list))
        print(f"  - SQL keywords: {len(self.sql_keywords)}")
        print(f"  - SQL functions: {len(self.sql_functions)}")
        
    def fit(self, X, y=None):
        """Fit the statistical extractor
        
        Args:
            X: array-like of shape (n_samples,)
            y: array-like of shape (n_samples,), optional
            
        Returns:
            self
        """
        print("\nFitting Statistical Extractor...")
        print(f"Number of samples: {len(X)}")
        self.is_fitted = True
        return self
        
    def transform(self, X):
        """Transform the data using statistical features
        
        Args:
            X: array-like of shape (n_samples,)
            
        Returns:
            array-like of shape (n_samples, n_features)
        """
        print("\nExtracting statistical features...")
        start_time = time.time()
        features = []
        
        for i, x in enumerate(tqdm(X, desc="Processing samples", unit="sample")):
            x = str(x).lower()
            feature_vector = []
            
            # Length features
            feature_vector.append(len(x))
            
            # Word count
            words = x.split()
            feature_vector.append(len(words))
            
            # Average word length
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            feature_vector.append(avg_word_len)
            
            # Character ratios
            total_len = len(x) if len(x) > 0 else 1
            special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', x))
            digits = len(re.findall(r'\d', x))
            uppercase = len(re.findall(r'[A-Z]', x))
            spaces = len(re.findall(r'\s', x))
            
            feature_vector.extend([
                special_chars / total_len,
                digits / total_len,
                uppercase / total_len,
                spaces / total_len
            ])
            
            # SQL specific features
            keyword_count = sum(1 for word in words if word in self.sql_keywords)
            function_count = sum(1 for word in words if word in self.sql_functions)
            comment_count = len(re.findall(r'(--|\*/|/\*|#)', x))
            
            feature_vector.extend([
                keyword_count,
                function_count,
                comment_count
            ])
            
            features.append(feature_vector)
            
        features = np.array(features)
        processing_time = time.time() - start_time
        
        print(f"\nFeature extraction completed:")
        print(f"  Time taken: {processing_time:.2f}s")
        print(f"  Samples processed: {len(X)}")
        print(f"  Features per sample: {len(self.feature_name_list)}")
        print(f"  Feature matrix shape: {features.shape}")
        
        return features

    @property
    def feature_names(self):
        """Get feature names
        
        Returns:
            list of str
        """
        return self.feature_name_list
