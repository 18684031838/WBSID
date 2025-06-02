"""机器学习模型实现SQL注入检测，基于HTTP请求参数"""
import re
import json
import numpy as np
import pandas as pd
import torch
import sys
import pickle
from urllib.parse import unquote
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Union
import time

# 添加项目根目录到Python路径
import os
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from .config import MODEL_CONFIG
from ml.feature_extractors.combined import CombinedExtractor
from ml.feature_extractors.statistical import StatisticalExtractor
from ml.feature_extractors.sql_semantic import SQLSemanticExtractor
from ml.feature_extractors.tfidf import TFIDFExtractor
from ml.feature_extractors.word2vec import Word2VecExtractor

# 特征提取器映射
EXTRACTOR_CLASSES = {
    'statistical': StatisticalExtractor,
    'sql_semantic': SQLSemanticExtractor,
    'tfidf': TFIDFExtractor,
    'word2vec': Word2VecExtractor
}

# 设置控制台输出编码为UTF-8
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    if sys.version_info[0] == 3:
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class MLDetector:
    """机器学习模型实现SQL注入检测"""
    
    def __init__(self):
        """初始化ML检测器"""
        self.logger = logging.getLogger('sql_injection_middleware.ml_detector')
        
        # 加载配置参数
        self.confidence_threshold = MODEL_CONFIG['confidence_threshold']
        self.model_type = MODEL_CONFIG['model_type']
        
        try:
            # 获取模型路径
            model_path = MODEL_CONFIG['model_paths'][self.model_type]
            
            # 检查模型文件是否存在
            if not Path(model_path).exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载模型
            self.logger.info(f"正在加载{self.model_type}模型: {model_path}")
            self.model = joblib.load(model_path)
            self.logger.info("模型加载成功")
            
            # 初始化组合特征提取器
            enabled_extractors = MODEL_CONFIG.get('enabled_extractors', ['statistical', 'sql_semantic'])
            if not enabled_extractors:
                self.logger.warning("未配置启用的特征提取器，使用默认特征提取器: statistical, sql_semantic")
                enabled_extractors = ['statistical', 'sql_semantic']
                
            self.feature_extractor = CombinedExtractor(
                enabled_extractors=enabled_extractors,
                max_length=MODEL_CONFIG.get('max_sequence_length', 128),
                embedding_dim=MODEL_CONFIG.get('embedding_dim', 8)
            )
            self.feature_extractor.fit([])  # 空列表足够进行初始化
            self.logger.info(f"组合特征提取器初始化成功，启用的特征: {enabled_extractors}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def preprocess_request(self, request_data: Dict[str, Any]) -> str:
        """预处理HTTP请求数据
        
        Args:
            request_data: HTTP请求数据字典
            
        Returns:
            str: 预处理后的查询字符串
        """
        try:
            # 提取查询参数
            query = ''
            if isinstance(request_data, dict):
                # 将所有参数值拼接成字符串
                query = ' '.join(str(v) for v in request_data.values())
            else:
                query = str(request_data)
            
            # URL解码
            query = unquote(query)
            
            # 移除多余的空白字符
            query = re.sub(r'\s+', ' ', query).strip()
            
            # 截断到最大序列长度
            if len(query) > MODEL_CONFIG['max_sequence_length']:
                query = query[:MODEL_CONFIG['max_sequence_length']]
                self.logger.warning(f"查询被截断到最大长度 {MODEL_CONFIG['max_sequence_length']}")
            
            self.logger.debug(f"预处理后的查询: {query}")
            return query
            
        except Exception as e:
            self.logger.error(f"请求预处理失败: {str(e)}")
            raise
    
    def extract_features(self, query: str) -> np.ndarray:
        """从查询字符串中提取特征
        
        Args:
            query: 预处理后的查询字符串
            
        Returns:
            np.ndarray: 特征向量
        """
        try:
            # 使用组合特征提取器提取特征
            features = self.feature_extractor.transform([query])
            self.logger.debug(f"提取的特征维度: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {str(e)}")
            raise
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """使用模型进行预测
        
        Args:
            features: 特征向量
            
        Returns:
            (预测结果, 置信度)
        """
        try:
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            return bool(prediction), float(confidence)
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {str(e)}")
            raise
    
    def detect(self, request_data: Dict[str, Any]) -> Tuple[bool, float]:
        """检测HTTP请求是否包含SQL注入
        
        Args:
            request_data: HTTP请求数据
            
        Returns:
            (是否是SQL注入, 置信度)
        """
        start_time = time.time()
        try:
            # 预处理请求数据
            query = self.preprocess_request(request_data)
            preprocess_time = time.time() - start_time
            
            # 提取特征
            feature_start = time.time()
            features = self.extract_features(query)
            feature_time = time.time() - feature_start
            
            # 使用模型预测
            predict_start = time.time()
            is_injection, confidence = self.predict(features)
            predict_time = time.time() - predict_start
            
            total_time = time.time() - start_time
            
            # 记录检测结果
            if is_injection:
                self.logger.warning(
                    "检测到SQL注入攻击\n"
                    f"检测参数:\n"
                    f"  - 原始请求: {json.dumps(request_data, ensure_ascii=False)}\n"
                    f"  - 预处理后: {query}\n"
                    f"检测结果:\n"
                    f"  - 是否注入: {is_injection}\n"
                    f"  - 置信度: {confidence:.2%}\n"
                    f"时间消耗:\n"
                    f"  - 预处理: {preprocess_time:.3f}s\n"
                    f"  - 特征提取: {feature_time:.3f}s\n"
                    f"  - 模型预测: {predict_time:.3f}s\n"
                    f"  - 总耗时: {total_time:.3f}s"
                )
            else:
                self.logger.debug(
                    f"SQL注入检测完成 [正常请求]\n"
                    f"参数: {query}\n"
                    f"置信度: {confidence:.2%}\n"
                    f"总耗时: {total_time:.3f}s"
                )
            
            return is_injection, confidence
            
        except Exception as e:
            self.logger.error(f"SQL注入检测失败: {str(e)}")
            raise