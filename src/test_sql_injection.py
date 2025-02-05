"""
SQL注入中间件测试脚本

环境变量配置：
REDIS_HOST: Redis服务器地址（默认：localhost）
REDIS_PORT: Redis服务器端口（默认：6379）
REDIS_PASSWORD: Redis服务器密码
REDIS_DB: Redis数据库编号（默认：0）
REDIS_CACHE_DB: Redis缓存数据库编号（默认：1）
REDIS_CACHE_TTL: Redis缓存过期时间（默认：3600秒）
BACKEND_SERVICE_URL: 后端服务地址（默认：http://localhost:8000）
BACKEND_SERVICE_TIMEOUT: 后端服务超时时间（默认：30秒）
"""
import sys
from pathlib import Path

# Add the parent directory to sys.path
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from flask import Flask
from sql_injection_middleware import SQLInjectionMiddleware

app = Flask(__name__)

# 配置SQL注入中间件
config = {
    'model_path': 'models/sql_injection_model.pkl',
    'confidence_threshold': 0.8
}

# 应用SQL注入中间件
app.wsgi_app = SQLInjectionMiddleware(app.wsgi_app, config)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
