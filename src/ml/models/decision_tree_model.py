"""
决策树模型实现
"""
from typing import Dict, Any
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from .sklearn_base_model import SklearnBaseModel

class DecisionTreeModel(SklearnBaseModel):
    def __init__(self):
        """初始化决策树分类器"""
        self.model = DecisionTreeClassifier(
            criterion='gini',  # 使用基尼系数
            max_depth=None,  # 树的最大深度，None表示不限制
            min_samples_split=2,  # 分裂内部节点所需的最小样本数
            min_samples_leaf=1,  # 叶节点所需的最小样本数
            max_features=None,  # 寻找最佳分割时考虑的特征数量
            random_state=42,  # 随机种子
            class_weight='balanced'  # 处理类别不平衡
        )
        self.param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
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
