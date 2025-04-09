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
from src.ml.feature_extractors.statistical import StatisticalExtractor
from src.ml.feature_extractors.sql_semantic import SQLSemanticExtractor
from src.ml.feature_extractors.tfidf import TFIDFExtractor
from src.ml.feature_extractors.word2vec import Word2VecExtractor

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
        self.feature_extractors = {}
        
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
            
            # 加载启用的特征提取器
            enabled_extractors = MODEL_CONFIG.get('enabled_extractors', ['statistical', 'sql_semantic'])
            if not enabled_extractors:
                self.logger.warning("未配置启用的特征提取器，使用默认特征提取器: statistical, sql_semantic")
                enabled_extractors = ['statistical', 'sql_semantic']
                
            for extractor_name in enabled_extractors:
                if extractor_name not in EXTRACTOR_CLASSES:
                    raise ValueError(f"未知的特征提取器类型: {extractor_name}")
                
                # 初始化特征提取器类
                extractor_class = EXTRACTOR_CLASSES[extractor_name]
                extractor = extractor_class()
                # 初始化特征提取器
                extractor.fit([])  # 空列表足够进行初始化
                self.feature_extractors[extractor_name] = extractor
                self.logger.info(f"特征提取器 {extractor_name} 初始化成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def preprocess_request(self, request_data: Union[Dict[str, Any], str]) -> str:
        """预处理HTTP请求数据
        
        Args:
            request_data: 可以是字典(JSON)或字符串(URL参数)
        
        Returns:
            str: 预处理后的查询字符串
        """
        try:
            # 记录原始请求格式
            self.logger.debug(f"原始请求数据: {request_data}")
            
            # 处理不同类型请求
            if isinstance(request_data, dict):
                query = ' '.join(str(v) for v in request_data.values())
            elif isinstance(request_data, str):
                query = request_data
            else:
                raise ValueError(f"不支持的请求数据类型: {type(request_data)}")
            
            # 统一解码URL编码字符
            query = unquote(query)
            
            # 截断到最大序列长度
            if len(query) > MODEL_CONFIG['max_sequence_length']:
                query = query[:MODEL_CONFIG['max_sequence_length']]
                self.logger.warning(f"查询被截断到最大长度 {MODEL_CONFIG['max_sequence_length']}")
            
            self.logger.debug(f"预处理结果: {query}")
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
            # 提取所有特征并合并
            all_features = []
            for extractor_name, extractor in self.feature_extractors.items():
                self.logger.debug(f"使用 {extractor_name} 提取特征")
                features = extractor.transform([query])
                all_features.append(features)
            
            # 水平连接所有特征
            combined_features = np.hstack(all_features)
            self.logger.debug(f"合并后的特征维度: {combined_features.shape}")
            return combined_features
            
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