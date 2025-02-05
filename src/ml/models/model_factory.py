"""
模型工厂类，用于创建不同类型的模型
"""
from typing import Dict, Type
from .base_model import BaseModel
from .svm_model import SVMModel
from .decision_tree_model import DecisionTreeModel
from .logistic_regression_model import LogisticRegressionModel
from .cnn_model import CNNModel
from .random_forest import RandomForestModel

class ModelFactory:
    _models: Dict[str, Type[BaseModel]] = {
        'svm': SVMModel,
        'decision_tree': DecisionTreeModel,
        'logistic_regression': LogisticRegressionModel,
        'cnn': CNNModel,
        'random_forest': RandomForestModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        创建指定类型的模型
        Args:
            model_type: 模型类型，可选值：'svm', 'decision_tree', 'logistic_regression', 'cnn', 'random_forest'
            **kwargs: 模型初始化参数
        Returns:
            模型实例
        """
        if model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """获取所有可用的模型类型"""
        return list(cls._models.keys())
