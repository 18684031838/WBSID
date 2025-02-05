"""
Base feature extractor module
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the feature extractor
        
        Args:
            X: array-like of shape (n_samples,)
            y: array-like of shape (n_samples,), optional
            
        Returns:
            self
        """
        pass
        
    @abstractmethod
    def transform(self, X):
        """Transform the data
        
        Args:
            X: array-like of shape (n_samples,)
            
        Returns:
            array-like of shape (n_samples, n_features)
        """
        pass
        
    def fit_transform(self, X, y=None):
        """Fit and transform the data
        
        Args:
            X: array-like of shape (n_samples,)
            y: array-like of shape (n_samples,), optional
            
        Returns:
            array-like of shape (n_samples, n_features)
        """
        return self.fit(X, y).transform(X)
        
    @property
    def feature_names(self):
        """Get feature names
        
        Returns:
            list of str
        """
        return []
