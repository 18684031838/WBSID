"""
scikit-learn模型的基类，实现了通用的保存和加载功能
"""
from typing import Dict, Any
import numpy as np
import joblib
from .base_model import BaseModel

class SklearnBaseModel(BaseModel):
    """scikit-learn模型的基类"""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """适配scikit-learn的fit接口，映射到train方法"""
        return self.train(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.model.get_params()
    
    def save(self, path: str) -> None:
        """保存模型，使用joblib格式"""
        print(f"Original save path: {path}")
        # 如果路径已经有扩展名，先移除
        if '.' in path.split('/')[-1]:
            path = path.rsplit('.', 1)[0]
            print(f"Path after extension removal: {path}")
        
        # 添加正确的扩展名
        path = path + self.model_extension
        print(f"Final save path with extension: {path}")
        
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """加载模型，使用joblib格式"""
        # 如果路径已经有扩展名，先移除
        if '.' in path.split('/')[-1]:
            path = path.rsplit('.', 1)[0]
            
        # 添加正确的扩展名
        path = path + self.model_extension
            
        self.model = joblib.load(path)
    
    @property
    def model_extension(self) -> str:
        """返回模型文件扩展名"""
        return '.joblib'
