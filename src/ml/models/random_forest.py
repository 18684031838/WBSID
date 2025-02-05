"""
随机森林模型实现
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from typing import Dict, Any
from .sklearn_base_model import SklearnBaseModel

class RandomForestModel(SklearnBaseModel):
    """Random Forest model for SQL injection detection"""
    
    def __init__(self):
        """Initialize the Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=100,  # 树的数量
            criterion='gini',  # 使用基尼系数
            max_depth=None,  # 树的最大深度，None表示不限制
            min_samples_split=2,  # 分裂内部节点所需的最小样本数
            min_samples_leaf=1,  # 叶节点所需的最小样本数
            max_features='sqrt',  # 寻找最佳分割时考虑的特征数量
            bootstrap=True,  # 使用bootstrap样本
            n_jobs=-1,  # 使用所有CPU核心
            random_state=42,  # 随机种子
            class_weight='balanced'  # 处理类别不平衡
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        return self.model.predict_proba(X)
        
    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters"""
        return self.model.get_params()
