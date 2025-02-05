"""
SQL注入防御中间件配置
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging.config
import logging

# 设置控制台输出编码为UTF-8
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    if sys.version_info[0] == 3:
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 设置环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 配置基本日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # 添加编码设置
)
logger = logging.getLogger('sql_injection_middleware.config')

# 加载环境变量配置文件
env_path = Path(__file__).parent / 'config.env'
if env_path.exists():
    logger.info(f"正在从 {env_path} 加载配置...")
    load_dotenv(env_path)
    logger.info("配置文件加载完成")
else:
    logger.warning(f"配置文件 {env_path} 不存在，将使用默认配置")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 模型配置
MODEL_CONFIG = {
    'model_type': os.getenv('MODEL_TYPE', 'random_forest'),  # 默认使用随机森林模型
    'model_paths': {
        'random_forest': str(PROJECT_ROOT / 'models' / 'random_forest_model.joblib'),
        'svm': str(PROJECT_ROOT / 'models' / 'svm_model.joblib'),
        'decision_tree': str(PROJECT_ROOT / 'models' / 'decision_tree_model.joblib'),
        'logistic_regression': str(PROJECT_ROOT / 'models' / 'logistic_regression_model.joblib'),
    },
    'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),  # 检测置信度阈值
    'max_sequence_length': int(os.getenv('MAX_SEQUENCE_LENGTH', '1000')),  # 最大序列长度
    'feature_extractors': {
        'statistical': str(PROJECT_ROOT / os.getenv('STATISTICAL_EXTRACTOR', 'models/statistical_extractor.joblib')),
        'sql_semantic': str(PROJECT_ROOT / os.getenv('SQL_SEMANTIC_EXTRACTOR', 'models/sql_semantic_extractor.joblib')),
        'tfidf': str(PROJECT_ROOT / os.getenv('TFIDF_EXTRACTOR', 'models/tfidf_extractor.joblib')),
        'word2vec': str(PROJECT_ROOT / os.getenv('WORD2VEC_EXTRACTOR', 'models/word2vec_extractor.joblib'))
    },
    # 默认启用的特征提取器
    'enabled_extractors': os.getenv('ENABLED_EXTRACTORS', 'statistical,sql_semantic').split(',')
}

# Redis配置
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'password': os.getenv('REDIS_PASSWORD', ''),
    'db': int(os.getenv('REDIS_DB', 0)),
    'cache_db': int(os.getenv('REDIS_CACHE_DB', 1)),
    'cache_ttl': int(os.getenv('REDIS_CACHE_TTL', 3600))
}

# 后端服务配置
BACKEND_CONFIG = {
    'url': os.getenv('BACKEND_URL', 'http://localhost:8000'),
    'timeout': int(os.getenv('BACKEND_TIMEOUT', 30))
}

# 性能监控配置
PERFORMANCE_CONFIG = {
    'enabled': True,
    'log_to_file': True,
    'log_dir': 'logs/performance',
    'sampling_rate': 1.0,
    'max_history_size': 1000,
    'export_format': 'json',
    'metrics_to_monitor': [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'inference_time',
        'memory_usage'
    ]
}

# 日志配置
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': str(Path(__file__).parent / 'logs' / 'sql_injection.log'),
            'mode': 'a',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'sql_injection_middleware': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# 输出配置信息
logger.info(f"模型配置: {MODEL_CONFIG}")
logger.info(f"Redis配置: {REDIS_CONFIG}")
logger.info(f"后端服务配置: {BACKEND_CONFIG}")
logger.info(f"性能监控配置: {PERFORMANCE_CONFIG}")
logger.info(f"日志配置: {LOGGING_CONFIG}")
