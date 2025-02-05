"""
逻辑回归模型实现
"""
from typing import Dict, Any
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from .sklearn_base_model import SklearnBaseModel

class LogisticRegressionModel(SklearnBaseModel):
    def __init__(self):
        """初始化逻辑回归分类器"""
        self.model = LogisticRegression(
            penalty='l2',  # L2正则化
            C=1.0,  # 正则化强度的倒数
            solver='lbfgs',  # 优化算法
            max_iter=1000,  # 最大迭代次数
            n_jobs=-1,  # 使用所有CPU核心
            random_state=42,  # 随机种子
            class_weight='balanced'  # 处理类别不平衡
        )
        self.param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga', 'lbfgs']
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练模型"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()
