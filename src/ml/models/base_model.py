"""
模型基类，定义了所有模型需要实现的接口
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 模型保存路径，子类应该根据自己的模型类型选择合适的文件扩展名
                 例如：PyTorch模型使用.pt，scikit-learn模型使用.joblib
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path: 模型加载路径，子类应该能处理自己对应的模型文件格式
        """
        pass
    
    @property
    @abstractmethod
    def model_extension(self) -> str:
        """返回模型文件的扩展名（包含点号）
        
        Returns:
            str: 例如'.pt'表示PyTorch模型，'.joblib'表示scikit-learn模型
        """
        pass
