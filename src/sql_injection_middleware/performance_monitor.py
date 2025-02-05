"""性能监控模块，支持配置控制"""
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from pathlib import Path
from . import config

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    model_type: str
    feature_types: List[str]
    timestamp: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    inference_time: float
    memory_usage: float
    training_time: Optional[float] = None
    false_positive_rate: Optional[float] = None
    true_positive_rate: Optional[float] = None
    auc_score: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_params: Optional[Dict[str, Any]] = None

@dataclass
class MonitorConfig:
    """性能监控配置"""
    enabled: bool = True  # 是否启用监控
    log_to_file: bool = True  # 是否记录到文件
    log_dir: str = "logs/performance"  # 日志目录
    metrics_to_monitor: List[str] = None  # 要监控的指标，None表示全部
    sampling_rate: float = 1.0  # 采样率，1.0表示记录所有数据
    max_history_size: int = 1000  # 最大历史记录数
    export_format: str = "json"  # 导出格式：json或csv
    
    @classmethod
    def from_config(cls) -> 'MonitorConfig':
        """从全局配置创建监控配置实例"""
        return cls(
            enabled=config.MONITOR_CONFIG['enabled'],
            log_to_file=config.MONITOR_CONFIG['log_to_file'],
            log_dir=config.MONITOR_CONFIG['log_dir'],
            metrics_to_monitor=config.MONITOR_CONFIG['metrics_to_monitor'],
            sampling_rate=config.MONITOR_CONFIG['sampling_rate'],
            max_history_size=config.MONITOR_CONFIG['max_history_size'],
            export_format=config.MONITOR_CONFIG['export_format']
        )
    
    def __post_init__(self):
        """初始化后处理"""
        if self.metrics_to_monitor is None:
            self.metrics_to_monitor = [
                "accuracy", "precision", "recall", "f1",
                "inference_time", "memory_usage"
            ]

class PerformanceMonitor:
    """性能监控类，负责记录和分析模型性能"""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """初始化性能监控器
        
        Args:
            config: 监控配置，如果为None则使用全局配置
        """
        self.config = config or MonitorConfig.from_config()
        self.logger = logging.getLogger(__name__)
        
        if self.config.enabled:
            if self.config.log_to_file:
                self._setup_logging()
            
            # 性能指标历史记录
            self.metrics_history: List[PerformanceMetrics] = []
    
    def _should_record(self) -> bool:
        """判断是否应该记录当前指标
        
        Returns:
            bool: 是否记录
        """
        if not self.config.enabled:
            return False
            
        # 根据采样率决定是否记录
        return np.random.random() < self.config.sampling_rate
    
    def _maintain_history_size(self):
        """维护历史记录大小"""
        if len(self.metrics_history) > self.config.max_history_size:
            # 保留最新的记录
            self.metrics_history = self.metrics_history[-self.config.max_history_size:]
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """记录性能指标
        
        Args:
            metrics: 性能指标数据
        """
        if not self._should_record():
            return
            
        # 过滤指标，但保留必需字段
        metrics_dict = asdict(metrics)
        required_fields = [
            'model_type', 'feature_types', 'timestamp',
            'accuracy', 'precision', 'recall', 'f1',
            'inference_time', 'memory_usage', 'training_time'
        ]
        
        # 添加到历史记录
        self.metrics_history.append(metrics)
        self._maintain_history_size()
        
        # 记录日志
        if self.config.log_to_file:
            self.logger.info(f"记录性能指标 - 模型: {metrics.model_type}")
            for metric_name in required_fields:
                if hasattr(metrics, metric_name):
                    value = getattr(metrics, metric_name)
                    if isinstance(value, float):
                        self.logger.info(f"{metric_name}: {value:.4f}")
                    else:
                        self.logger.info(f"{metric_name}: {value}")
        
        # 保存到文件
        if self.config.log_to_file:
            if self.config.export_format == "json":
                self._save_to_json(metrics)
            else:
                self._save_to_csv(metrics)
    
    def _setup_logging(self):
        """设置日志配置"""
        import os
        
        # 创建日志目录
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # 创建性能日志文件处理器
        performance_log = logging.FileHandler(
            f"{self.config.log_dir}/performance_{datetime.now().strftime('%Y%m%d')}.log"
        )
        performance_log.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        performance_log.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(performance_log)
    
    def _save_to_json(self, metrics: PerformanceMetrics):
        """保存性能指标到JSON文件
        
        Args:
            metrics: 性能指标数据
        """
        import os
        
        # 构建文件路径
        file_path = os.path.join(
            self.config.log_dir,
            f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # 转换为字典并保存
        with open(file_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=4)
    
    def _save_to_csv(self, metrics: PerformanceMetrics):
        """保存性能指标到CSV文件
        
        Args:
            metrics: 性能指标数据
        """
        import os
        import pandas as pd
        
        # 构建文件路径
        file_path = os.path.join(
            self.config.log_dir,
            f"metrics_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        # 转换为DataFrame
        df = pd.DataFrame([asdict(metrics)])
        
        # 追加到CSV文件
        mode = 'a' if os.path.exists(file_path) else 'w'
        header = not os.path.exists(file_path)
        df.to_csv(file_path, mode=mode, header=header, index=False)
    
    def analyze_performance(self, 
                          model_type: Optional[str] = None,
                          feature_types: Optional[List[str]] = None,
                          time_range: Optional[tuple] = None) -> pd.DataFrame:
        """分析性能指标
        
        Args:
            model_type: 模型类型过滤
            feature_types: 特征类型过滤
            time_range: 时间范围过滤 (start_time, end_time)
            
        Returns:
            pd.DataFrame: 性能分析结果
        """
        # 转换为DataFrame
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        
        # 应用过滤
        if model_type:
            df = df[df['model_type'] == model_type]
        
        if feature_types:
            df = df[df['feature_types'].apply(lambda x: all(ft in x for ft in feature_types))]
        
        if time_range:
            start_time, end_time = time_range
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        
        # 计算统计信息
        stats = {
            '平均准确率': df['accuracy'].mean(),
            '平均精确率': df['precision'].mean(),
            '平均召回率': df['recall'].mean(),
            '平均F1分数': df['f1'].mean(),
            '平均推理时间(ms)': df['inference_time'].mean(),
            '平均内存使用(MB)': df['memory_usage'].mean()
        }
        
        self.logger.info("性能分析结果:")
        for metric, value in stats.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return df
    
    def plot_performance_trends(self, 
                              metrics: List[str],
                              model_types: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
        """绘制性能趋势图
        
        Args:
            metrics: 要绘制的指标列表
            model_types: 要包含的模型类型
            save_path: 图表保存路径
        """
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        
        if model_types:
            df = df[df['model_type'].isin(model_types)]
        
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            plt.plot(df['timestamp'], df[metric], label=metric)
        
        plt.title('模型性能趋势')
        plt.xlabel('时间')
        plt.ylabel('指标值')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def compare_models(self, metric: str = 'f1') -> pd.DataFrame:
        """比较不同模型的性能
        
        Args:
            metric: 用于比较的指标
            
        Returns:
            pd.DataFrame: 模型比较结果
        """
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        
        comparison = df.groupby('model_type')[metric].agg([
            '平均值', '标准差', '最小值', '最大值'
        ]).round(4)
        
        self.logger.info(f"\n模型性能比较 (基于 {metric}):")
        self.logger.info(f"\n{comparison}")
        
        return comparison
    
    def get_feature_importance(self, model_type: str) -> Optional[Dict[str, float]]:
        """获取特征重要性
        
        Args:
            model_type: 模型类型
            
        Returns:
            Optional[Dict[str, float]]: 特征重要性字典
        """
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        model_data = df[df['model_type'] == model_type].iloc[-1]
        
        if model_data['feature_importance'] is not None:
            importance = model_data['feature_importance']
            
            # 按重要性排序
            importance = dict(sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            self.logger.info(f"\n{model_type} 特征重要性:")
            for feature, score in importance.items():
                self.logger.info(f"{feature}: {score:.4f}")
            
            return importance
        
        return None
    
    def export_report(self, output_path: str):
        """导出性能报告
        
        Args:
            output_path: 报告保存路径
        """
        # 将性能指标转换为DataFrame
        metrics_list = []
        for m in self.metrics_history:
            metrics_dict = {
                'model_type': m.model_type,
                'accuracy': m.accuracy,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'inference_time': m.inference_time,
                'memory_usage': m.memory_usage,
                'training_time': m.training_time if m.training_time else 0
            }
            metrics_list.append(metrics_dict)
            
        df = pd.DataFrame(metrics_list)
        
        # 创建报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# SQL注入检测模型性能报告\n\n")
            
            # 1. 总体性能统计
            f.write("## 1. 总体性能统计\n\n")
            overall_stats = df[['accuracy', 'precision', 'recall', 'f1']].mean()
            f.write(f"- 平均准确率: {overall_stats['accuracy']:.4f}\n")
            f.write(f"- 平均精确率: {overall_stats['precision']:.4f}\n")
            f.write(f"- 平均召回率: {overall_stats['recall']:.4f}\n")
            f.write(f"- 平均F1分数: {overall_stats['f1']:.4f}\n\n")
            
            # 2. 各模型性能对比
            f.write("## 2. 各模型性能对比\n\n")
            f.write("| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | 推理时间(秒) | 内存使用(MB) | 训练时间(秒) |\n")
            f.write("|------|--------|--------|--------|---------|--------------|--------------|-------------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['model_type']} | {row['accuracy']:.4f} | {row['precision']:.4f} | ")
                f.write(f"{row['recall']:.4f} | {row['f1']:.4f} | {row['inference_time']:.4f} | ")
                f.write(f"{row['memory_usage']:.2f} | {row['training_time']:.2f} |\n")
            
            f.write("\n## 3. 性能分析\n\n")
            
            # 找出最佳模型
            best_model = df.loc[df['f1'].idxmax()]
            f.write(f"### 最佳模型: {best_model['model_type']}\n")
            f.write(f"- F1分数: {best_model['f1']:.4f}\n")
            f.write(f"- 准确率: {best_model['accuracy']:.4f}\n")
            f.write(f"- 推理时间: {best_model['inference_time']:.4f}秒\n")
            f.write(f"- 内存使用: {best_model['memory_usage']:.2f}MB\n\n")
            
            # 性能权衡分析
            fastest_model = df.loc[df['inference_time'].idxmin()]
            f.write("### 性能权衡分析\n\n")
            f.write(f"- 最快模型: {fastest_model['model_type']}\n")
            f.write(f"  - 推理时间: {fastest_model['inference_time']:.4f}秒\n")
            f.write(f"  - F1分数: {fastest_model['f1']:.4f}\n\n")
            
            lowest_memory = df.loc[df['memory_usage'].idxmin()]
            f.write(f"- 最低内存使用: {lowest_memory['model_type']}\n")
            f.write(f"  - 内存使用: {lowest_memory['memory_usage']:.2f}MB\n")
            f.write(f"  - F1分数: {lowest_memory['f1']:.4f}\n")
