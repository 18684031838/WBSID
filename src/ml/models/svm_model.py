"""
SVM模型实现 - 优化版本
使用线性SVM，启用早停策略和并行计算
"""
from typing import Dict, Any
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from .sklearn_base_model import SklearnBaseModel

class SVMModel(SklearnBaseModel):
    def __init__(self):
        """初始化线性SVM模型"""
        # 使用线性SVM，配置优化参数
        self.model = LinearSVC(
            loss='squared_hinge',  # 使用平方铰链损失，收敛更快
            dual=False,  # 样本数量>特征数时，dual=False更快
            tol=1e-4,  # 优化的容差
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # 处理类别不平衡
        )
        # 使用CalibratedClassifierCV来获取概率输出
        self.calibrated_model = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练模型"""
        # 使用CalibratedClassifierCV包装LinearSVC以获得概率输出
        self.calibrated_model = CalibratedClassifierCV(
            self.model, 
            cv=5,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.calibrated_model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签"""
        return self.calibrated_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.calibrated_model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.model.get_params()
    
    def save(self, path: str) -> None:
        """保存校准后的模型"""
        # 如果路径已经有扩展名，先移除
        if '.' in path.split('/')[-1]:
            path = path.rsplit('.', 1)[0]
        # 添加正确的扩展名
        path = path + self.model_extension
        joblib.dump(self.calibrated_model, path)
    
    def load(self, path: str) -> None:
        """加载校准后的模型"""
        # 如果路径已经有扩展名，先移除
        if '.' in path.split('/')[-1]:
            path = path.rsplit('.', 1)[0]
        # 添加正确的扩展名
        path = path + self.model_extension
        self.calibrated_model = joblib.load(path)
