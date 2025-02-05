"""
模型训练模块
使用SVM进行训练，包含参数优化和模型评估
"""
import numpy as np
from typing import Dict, Tuple
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
from tqdm import tqdm
import time
import warnings

class ModelTrainer:
    def __init__(self):
        print("\nInitializing SQL Injection Detection Model:")
        print("  - Model Type: Linear Support Vector Machine")
        print("  - Implementation: scikit-learn LinearSVC")
        print("  - Features Used:")
        print("    * TF-IDF features (character-level n-grams)")
        print("    * Statistical features (special characters, keywords, etc.)")
        print("    * SQL Semantic features (position-aware embeddings)")
        
        # 使用线性SVM，配置早停和并行计算
        self.model = LinearSVC(
            C=1.0,                # 正则化参数
            loss='squared_hinge', # 平方铰链损失，收敛更快
            penalty='l2',         # L2正则化
            dual=False,           # 对偶优化，样本数>特征数时用False
            tol=1e-4,            # 优化收敛容差
            max_iter=1000,        # 最大迭代次数
            random_state=42,
            verbose=1             # 显示训练进度
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练SVM模型
        
        Args:
            X_train: array-like of shape (n_samples, n_features)
            y_train: array-like of shape (n_samples,)
        """
        print("\nTraining SVM model...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of samples: {X_train.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        
        # 训练模型
        print("\nStarting model training...")
        start_time = time.time()
        
        # 使用概率校准，这样可以获得预测概率
        self.model = CalibratedClassifierCV(
            self.model,
            cv=3,           # 3折交叉验证
            n_jobs=-1       # 使用所有CPU核心
        )
        
        self.model.fit(X_train, y_train)
        total_time = time.time() - start_time
        
        print(f"\nModel training completed in {total_time:.2f} seconds")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型性能"""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model has not been trained yet")
        
        print("\nEvaluating model on test set...")
        print(f"Test data shape: {X_test.shape}")
        
        # 预测
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # 生成评估报告
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # 打印详细的评估结果
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        print(f"\nPrediction time for {len(X_test)} samples: {prediction_time:.2f} seconds")
        print(f"Average prediction time per sample: {(prediction_time/len(X_test))*1000:.2f} ms")
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'prediction_time': prediction_time
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用模型进行预测"""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """获取预测概率"""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        if hasattr(self.model, 'predict'):
            print(f"\nSaving model to: {path}")
            joblib.dump(self.model, path)
            print("Model saved successfully!")
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        print(f"\nLoading model from: {path}")
        self.model = joblib.load(path)
        print("Model loaded successfully!")
