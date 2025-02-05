"""
启动SQL注入防护中间件
"""
import sys
import os
from werkzeug.serving import run_simple
from .sql_injection_middleware import SQLInjectionMiddleware
from .config import BACKEND_CONFIG
from pathlib import Path
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

def create_app():
    """创建应用"""
    logger = logging.getLogger('sql_injection_middleware')
    
    try:
        logger.info("正在初始化SQL注入防护中间件...")
        middleware = SQLInjectionMiddleware(app=None, config={
            'model_path': str(Path(__file__).parent.parent / 'ml' / 'models' / 'sql_injection_model.joblib'),
            'confidence_threshold': 0.8
        })
        logger.info("SQL注入防护中间件初始化完成")
        return middleware
    except Exception as e:
        logger.error(f"中间件初始化失败: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    logger = logging.getLogger('sql_injection_middleware')
    try:
        app = create_app()
        logger.info("启动服务器...")
        run_simple('localhost', 5000, app, use_reloader=True)
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}", exc_info=True)
        sys.exit(1)
