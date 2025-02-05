"""SQL注入检测模型工厂，支持不同场景的模型配置和实例化"""
from enum import Enum
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import psutil
from .performance_monitor import PerformanceMonitor, PerformanceMetrics, MonitorConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import time

class ModelType(Enum):
    """支持的模型类型"""
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    CNN = "cnn"

class CNNModel(nn.Module):
    """CNN模型用于SQL注入检测"""
    def __init__(self, input_dim, embedding_dim=128, num_filters=128, filter_sizes=(3, 4, 5), num_classes=2, dropout=0.5):
        super(CNNModel, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        # Dropout和全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        # x: (batch_size, sequence_length)
        
        # 嵌入: (batch_size, sequence_length, embedding_dim)
        x = self.embedding(x)
        
        # 添加通道维度: (batch_size, 1, sequence_length, embedding_dim)
        x = x.unsqueeze(1)
        
        # 应用卷积
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        # 拼接
        x = torch.cat(x, 1)
        
        # Dropout
        x = self.dropout(x)
        
        # 全连接层
        logits = self.fc(x)
        return logits

class FeatureType(Enum):
    """支持的特征类型"""
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    ENCODING = "encoding"
    ALL = "all"

@dataclass
class SceneConfig:
    """场景配置"""
    name: str
    description: str
    model_type: ModelType
    feature_types: List[FeatureType]
    model_params: Dict[str, Any]
    feature_params: Dict[str, Any]
    
class PredefinedScenes:
    """预定义的场景配置"""
    
    @staticmethod
    def get_web_api_scene() -> SceneConfig:
        """Web API场景
        
        特点：
        1. 参数较短
        2. 结构化程度高
        3. 响应时间要求高
        """
        return SceneConfig(
            name="web_api",
            description="Web API场景，适用于REST API等接口的SQL注入检测",
            model_type=ModelType.RANDOM_FOREST,
            feature_types=[
                FeatureType.PATTERN,
                FeatureType.STATISTICAL,
                FeatureType.ENCODING
            ],
            model_params={
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 5,
                "class_weight": "balanced"
            },
            feature_params={
                "pattern": {
                    "use_regex": True,
                    "max_patterns": 50
                },
                "statistical": {
                    "use_char_dist": True,
                    "use_length_features": True
                },
                "encoding": {
                    "check_url_encoding": True,
                    "check_hex_encoding": True
                }
            }
        )
    
    @staticmethod
    def get_form_submission_scene() -> SceneConfig:
        """表单提交场景
        
        特点：
        1. 参数较多
        2. 包含多种数据类型
        3. 可能包含富文本
        """
        return SceneConfig(
            name="form_submission",
            description="表单提交场景，适用于处理用户提交的表单数据",
            model_type=ModelType.GRADIENT_BOOSTING,
            feature_types=[
                FeatureType.TFIDF,
                FeatureType.WORD2VEC,
                FeatureType.PATTERN,
                FeatureType.STATISTICAL
            ],
            model_params={
                "n_estimators": 200,
                "max_depth": 20,
                "learning_rate": 0.1,
                "subsample": 0.8
            },
            feature_params={
                "tfidf": {
                    "max_features": 5000,
                    "ngram_range": (1, 3)
                },
                "word2vec": {
                    "vector_size": 100,
                    "window": 5,
                    "min_count": 1
                },
                "pattern": {
                    "use_regex": True,
                    "max_patterns": 100
                },
                "statistical": {
                    "use_char_dist": True,
                    "use_length_features": True,
                    "use_word_features": True
                }
            }
        )
    
    @staticmethod
    def get_search_query_scene() -> SceneConfig:
        """搜索查询场景
        
        特点：
        1. 参数自由度高
        2. 可能包含复杂查询语法
        3. 需要较高准确度
        """
        return SceneConfig(
            name="search_query",
            description="搜索查询场景，适用于处理用户搜索输入",
            model_type=ModelType.SVM,
            feature_types=[
                FeatureType.TFIDF,
                FeatureType.WORD2VEC,
                FeatureType.PATTERN,
                FeatureType.ENCODING
            ],
            model_params={
                "kernel": "rbf",
                "C": 10.0,
                "gamma": "scale",
                "class_weight": "balanced"
            },
            feature_params={
                "tfidf": {
                    "max_features": 10000,
                    "ngram_range": (1, 4)
                },
                "word2vec": {
                    "vector_size": 200,
                    "window": 7,
                    "min_count": 1
                },
                "pattern": {
                    "use_regex": True,
                    "max_patterns": 150,
                    "use_advanced_patterns": True
                },
                "encoding": {
                    "check_url_encoding": True,
                    "check_hex_encoding": True,
                    "check_unicode_encoding": True,
                    "check_base64_encoding": True
                }
            }
        )
    
    @staticmethod
    def get_high_security_scene() -> SceneConfig:
        """高安全性场景
        
        特点：
        1. 安全要求极高
        2. 可以接受较高的误报率
        3. 性能要求相对较低
        """
        return SceneConfig(
            name="high_security",
            description="高安全性场景，适用于对安全性要求极高的系统",
            model_type=ModelType.GRADIENT_BOOSTING,
            feature_types=[
                FeatureType.ALL
            ],
            model_params={
                "n_estimators": 500,
                "max_depth": 30,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "n_iter_no_change": 20
            },
            feature_params={
                "tfidf": {
                    "max_features": 20000,
                    "ngram_range": (1, 5)
                },
                "word2vec": {
                    "vector_size": 300,
                    "window": 10,
                    "min_count": 1,
                    "negative": 10
                },
                "pattern": {
                    "use_regex": True,
                    "max_patterns": 200,
                    "use_advanced_patterns": True,
                    "custom_patterns": True
                },
                "statistical": {
                    "use_char_dist": True,
                    "use_length_features": True,
                    "use_word_features": True,
                    "use_entropy": True
                },
                "encoding": {
                    "check_all": True,
                    "recursive_decoding": True
                }
            }
        )

@dataclass
class FactoryConfig:
    """工厂配置"""
    monitor_config: Optional[MonitorConfig] = None  # 性能监控配置
    use_gpu: bool = True  # 是否使用GPU
    random_seed: Optional[int] = 42  # 随机种子
    verbose: bool = True  # 是否输出详细日志

class ModelFactory:
    """模型工厂类，负责根据场景配置创建合适的模型实例"""
    
    def __init__(self, scene_config: SceneConfig, factory_config: Optional[FactoryConfig] = None):
        """初始化模型工厂
        
        Args:
            scene_config: 场景配置
            factory_config: 工厂配置，如果为None则使用默认配置
        """
        self.logger = logging.getLogger(__name__)
        self.config = scene_config
        self.factory_config = factory_config or FactoryConfig()
        
        if self.factory_config.verbose:
            self.logger.info(f"初始化模型工厂，场景: {scene_config.name}")
        
        # 设置随机种子
        if self.factory_config.random_seed is not None:
            np.random.seed(self.factory_config.random_seed)
            torch.manual_seed(self.factory_config.random_seed)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.factory_config.use_gpu else "cpu")
        
        # 初始化性能监控器
        self.performance_monitor = PerformanceMonitor(self.factory_config.monitor_config)
    
    def _measure_inference_time(self, model: Any, X: np.ndarray) -> tuple:
        """测量模型推理时间
        
        Args:
            model: 模型实例
            X: 输入数据
            
        Returns:
            tuple: (预测结果, 推理时间)
        """
        start_time = time.time()
        
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                y_pred = model(torch.from_numpy(X).to(self.device))
                y_pred = y_pred.cpu().numpy()
        else:
            y_pred = model.predict(X)
        
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        return y_pred, inference_time
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用情况
        
        Returns:
            float: 内存使用量（MB）
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # 转换为MB
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """获取特征重要性
        
        Args:
            model: 模型实例
            
        Returns:
            Optional[Dict[str, float]]: 特征重要性字典
        """
        if hasattr(model, 'feature_importances_'):
            # 随机森林和决策树等模型
            return dict(zip(
                [f"feature_{i}" for i in range(len(model.feature_importances_))],
                model.feature_importances_
            ))
        elif hasattr(model, 'coef_'):
            # 逻辑回归和SVM等线性模型
            return dict(zip(
                [f"feature_{i}" for i in range(len(model.coef_[0]))],
                np.abs(model.coef_[0])
            ))
        return None
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray):
        """评估模型性能
        
        Args:
            model: 模型实例
            X: 测试数据
            y: 真实标签
        """
        # 如果性能监控未启用，直接返回None
        if not self.factory_config.monitor_config or not self.factory_config.monitor_config.enabled:
            return None
            
        # 测量推理时间和内存使用
        y_pred, inference_time = self._measure_inference_time(model, X)
        memory_usage = self._get_memory_usage()
        
        # 计算性能指标
        metrics = PerformanceMetrics(
            model_type=self.config.model_type.value,
            feature_types=[ft.value for ft in self.config.feature_types],
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred),
            recall=recall_score(y, y_pred),
            f1=f1_score(y, y_pred),
            inference_time=inference_time,
            memory_usage=memory_usage,
            confusion_matrix=confusion_matrix(y, y_pred).tolist(),
            feature_importance=self._get_feature_importance(model),
            model_params=self.config.model_params
        )
        
        # 记录性能指标
        self.performance_monitor.record_metrics(metrics)
        
        return metrics
    
    def analyze_performance(self, **kwargs):
        """分析模型性能
        
        Args:
            **kwargs: 传递给performance_monitor.analyze_performance的参数
        """
        if not self.factory_config.monitor_config or not self.factory_config.monitor_config.enabled:
            self.logger.warning("性能监控未启用")
            return None
            
        return self.performance_monitor.analyze_performance(**kwargs)
    
    def plot_performance_trends(self, **kwargs):
        """绘制性能趋势图
        
        Args:
            **kwargs: 传递给performance_monitor.plot_performance_trends的参数
        """
        if not self.factory_config.monitor_config or not self.factory_config.monitor_config.enabled:
            self.logger.warning("性能监控未启用")
            return
            
        self.performance_monitor.plot_performance_trends(**kwargs)
    
    def export_performance_report(self, output_path: str):
        """导出性能报告
        
        Args:
            output_path: 报告保存路径
        """
        if not self.factory_config.monitor_config or not self.factory_config.monitor_config.enabled:
            self.logger.warning("性能监控未启用")
            return
            
        self.performance_monitor.export_report(output_path)

    def create_model(self) -> Any:
        """创建模型实例
        
        Returns:
            Any: 模型实例（可能是Pipeline或nn.Module）
        """
        # 1. 创建特征转换器
        feature_transformers = []
        
        # TF-IDF特征
        if FeatureType.TFIDF in self.config.feature_types or FeatureType.ALL in self.config.feature_types:
            tfidf_params = self.config.feature_params.get("tfidf", {})
            tfidf = TfidfVectorizer(**tfidf_params)
            feature_transformers.append(("tfidf", tfidf))
        
        # Word2Vec特征
        if FeatureType.WORD2VEC in self.config.feature_types or FeatureType.ALL in self.config.feature_types:
            word2vec_params = self.config.feature_params.get("word2vec", {})
            feature_transformers.append(("word2vec", None))
        
        # 统计特征
        if FeatureType.STATISTICAL in self.config.feature_types or FeatureType.ALL in self.config.feature_types:
            statistical_params = self.config.feature_params.get("statistical", {})
            feature_transformers.append(("statistical", StandardScaler()))
        
        # 模式特征
        if FeatureType.PATTERN in self.config.feature_types or FeatureType.ALL in self.config.feature_types:
            pattern_params = self.config.feature_params.get("pattern", {})
            feature_transformers.append(("pattern", None))
        
        # 编码特征
        if FeatureType.ENCODING in self.config.feature_types or FeatureType.ALL in self.config.feature_types:
            encoding_params = self.config.feature_params.get("encoding", {})
            feature_transformers.append(("encoding", None))
        
        # 2. 创建模型实例
        if self.config.model_type == ModelType.CNN:
            # CNN模型需要特殊处理
            model_params = self.config.model_params
            model = CNNModel(
                input_dim=model_params.get("input_dim", 10000),
                embedding_dim=model_params.get("embedding_dim", 128),
                num_filters=model_params.get("num_filters", 128),
                filter_sizes=model_params.get("filter_sizes", (3, 4, 5)),
                num_classes=model_params.get("num_classes", 2),
                dropout=model_params.get("dropout", 0.5)
            ).to(self.device)
            return model
        else:
            # 创建传统机器学习模型
            if self.config.model_type == ModelType.SVM:
                model = SVC(**self.config.model_params)
            elif self.config.model_type == ModelType.LOGISTIC_REGRESSION:
                model = LogisticRegression(**self.config.model_params)
            elif self.config.model_type == ModelType.DECISION_TREE:
                model = DecisionTreeClassifier(**self.config.model_params)
            elif self.config.model_type == ModelType.RANDOM_FOREST:
                model = RandomForestClassifier(**self.config.model_params)
            else:
                raise ValueError(f"不支持的模型类型: {self.config.model_type}")
            
            # 创建管道
            pipeline = Pipeline([
                ("features", ColumnTransformer(feature_transformers)),
                ("model", model)
            ])
            
            self.logger.info(f"创建模型实例完成，特征类型: {[ft.value for ft in self.config.feature_types]}")
            return pipeline
    
    @staticmethod
    def get_default_model_params(model_type: ModelType) -> Dict[str, Any]:
        """获取模型的默认参数配置
        
        Args:
            model_type: 模型类型
            
        Returns:
            Dict[str, Any]: 默认参数配置
        """
        params_map = {
            ModelType.SVM: {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "class_weight": "balanced"
            },
            ModelType.LOGISTIC_REGRESSION: {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1
            },
            ModelType.DECISION_TREE: {
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "random_state": 42
            },
            ModelType.RANDOM_FOREST: {
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1
            },
            ModelType.CNN: {
                "input_dim": 10000,
                "embedding_dim": 128,
                "num_filters": 128,
                "filter_sizes": (3, 4, 5),
                "num_classes": 2,
                "dropout": 0.5,
                "learning_rate": 0.001,
                "batch_size": 64,
                "num_epochs": 10
            }
        }
        return params_map.get(model_type, {})
