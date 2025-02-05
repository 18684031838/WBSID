"""
SQL注入防御中间件
实现了基于布隆过滤器的快速检测和机器学习模型的精确检测
"""
import json
import logging
import logging.config
import time
from redis import Redis, RedisError
import requests
import sys
from urllib.parse import urljoin
from werkzeug.wrappers import Request, Response
from .bloom_filter import SQLInjectionBloomFilter
from .ml_detector import MLDetector
from .config import LOGGING_CONFIG, REDIS_CONFIG, BACKEND_CONFIG
import redis

# 设置控制台输出编码为UTF-8
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    if sys.version_info[0] == 3:
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class SQLInjectionMiddleware:
    """SQL注入防御中间件
    
    实现了基于布隆过滤器的快速检测和机器学习模型的精确检测
    """
    
    def __init__(self, app=None, config=None):
        """初始化中间件
        
        Args:
            app: WSGI应用
            config: 配置字典，包含model_path和confidence_threshold
        """
        self.app = app
        self.config = config or {}
        
        # 配置日志
        logging.config.dictConfig(LOGGING_CONFIG)
        self.logger = logging.getLogger('sql_injection_middleware')
        
        try:
            # 打印Redis配置信息
            redis_config = {
                'host': REDIS_CONFIG.get('host', 'localhost'),
                'port': REDIS_CONFIG.get('port', 6379),
                'db': REDIS_CONFIG.get('db', 0),
                'password': REDIS_CONFIG.get('password', None),
                'socket_timeout': 10,  # 增加超时时间
                'socket_connect_timeout': 10,  # 增加连接超时时间
                'retry_on_timeout': True,
                'decode_responses': True,
                'health_check_interval': 30  # 添加健康检查间隔
            }
            self.logger.info(f"Redis配置信息: {redis_config}")
            
            # 初始化Redis连接
            self.logger.info("正在连接Redis服务器...")
            self.redis_client = Redis(**redis_config)
            
            # 测试连接
            self.logger.info("正在测试Redis连接...")
            pong = self.redis_client.ping()
            self.logger.info(f"Redis连接测试结果: {pong}")
            
            # 初始化布隆过滤器
            self.logger.info("正在初始化布隆过滤器...")
            self.bloom_filter = SQLInjectionBloomFilter()
            self.logger.info("布隆过滤器初始化成功")
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.logger.warning(f"Redis连接失败，将禁用布隆过滤器: {str(e)}")
            self.redis_client = None
            self.bloom_filter = None
        except Exception as e:
            self.logger.error(f"Redis或布隆过滤器初始化时发生未知错误: {str(e)}", exc_info=True)
            self.redis_client = None
            self.bloom_filter = None
            
        # 初始化ML检测器
        try:
            self.logger.info("正在初始化ML检测器...")
            self.ml_detector = MLDetector()
            self.logger.info("ML检测器初始化成功")
        except Exception as e:
            self.logger.error(f"ML检测器初始化失败: {str(e)}", exc_info=True)
            raise
            
        # 初始化Redis连接
        self.cache = redis.Redis(
            host=REDIS_CONFIG['host'],
            port=REDIS_CONFIG['port'],
            password=REDIS_CONFIG['password'],
            db=REDIS_CONFIG['cache_db']
        )
        self.cache_ttl = REDIS_CONFIG['cache_ttl']
        
        self.logger.info("SQL注入防御中间件已初始化")
    
    def __call__(self, environ, start_response):
        request = Request(environ)
        
        # 记录请求基本信息
        self.logger.info(f"收到新请求 - 方法: {request.method}, URL: {request.url}")
        self.logger.debug(f"请求头: {dict(request.headers)}")
        
        # 记录请求参数
        if request.args:
            self.logger.debug(f"URL参数: {dict(request.args)}")
        
        if request.form:
            self.logger.debug(f"表单数据: {dict(request.form)}")
            
        if request.is_json:
            self.logger.debug(f"JSON数据: {request.get_json()}")
            
        if request.cookies:
            self.logger.debug(f"Cookies: {dict(request.cookies)}")
        
        # 检查是否存在SQL注入
        start_time = time.time()
        is_injection = self._check_sql_injection(request)
        check_time = time.time() - start_time
        
        if is_injection:
            self.logger.warning(f"检测到SQL注入攻击: {request.url}, 检测耗时: {check_time:.3f}秒")
            response = Response(
                json.dumps({
                    'error': 'Forbidden',
                    'message': 'Potential SQL injection detected'
                }),
                status=403,
                content_type='application/json'
            )
            return response(environ, start_response)
        
        self.logger.info(f"SQL注入检测通过，检测耗时: {check_time:.3f}秒")
        
        # 转发请求到后端服务
        try:
            start_time = time.time()
            backend_response = self._forward_request(request)
            forward_time = time.time() - start_time
            
            self.logger.info(f"请求转发完成 - 状态码: {backend_response.status_code}, 耗时: {forward_time:.3f}秒")
            if backend_response.status_code >= 400:
                self.logger.warning(f"后端服务返回错误 - 状态码: {backend_response.status_code}, URL: {request.url}")
            
            response = Response(
                response=backend_response.content,
                status=backend_response.status_code,
                content_type=backend_response.headers.get('content-type')
            )
            return response(environ, start_response)
        except Exception as e:
            self.logger.error(f"转发请求失败: {str(e)}", exc_info=True)
            response = Response(
                json.dumps({
                    'error': 'Internal Server Error',
                    'message': 'Failed to forward request to backend service'
                }),
                status=500,
                content_type='application/json'
            )
            return response(environ, start_response)
    
    def _preprocess_parameters(self, params):
        """预处理参数列表，处理各种编码和混淆
        
        Args:
            params: 参数列表
            
        Returns:
            list: 处理后的参数列表
        """
        processed_params = []
        self.logger.debug(f"开始预处理参数，原始参数数量: {len(params)}")
        
        for param in params:
            if not param:  # 跳过空参数
                continue
            
            self.logger.debug(f"处理参数前: {param}")
                
            # 1. 处理混淆的空白字符
            param = param.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
            
            # 2. 处理注释和内联注释
            param = param.replace('/*', ' ').replace('*/', ' ').replace('--', ' ')
            
            # 3. 处理特殊字符
            param = param.replace('%20', ' ')  # URL编码的空格
            param = param.replace('\x00', '')  # 空字节
            
            # 4. 处理SQL关键字的常见变体
            param = param.replace('UNION/**/SELECT', 'UNION SELECT')
            param = param.replace('UN/**/ION', 'UNION')
            param = param.replace('SEL/**/ECT', 'SELECT')
            
            # 5. 处理常见的绕过技术
            param = param.replace('/*!', '')  # MySQL版本注释
            param = param.replace('`', '')    # 反引号
            param = param.replace('"', "'")   # 统一引号
            
            self.logger.debug(f"处理参数后: {param}")
            processed_params.append(param)
            
        self.logger.debug(f"参数预处理完成，处理后参数数量: {len(processed_params)}")
        return processed_params
    
    def _check_sql_injection(self, request):
        """检查请求中是否存在SQL注入
        
        检测流程：
        1. 首先收集所有可能包含SQL注入的参数
        2. 使用布隆过滤器进行快速检测
        3. 如果布隆过滤器检测到可疑，则使用机器学习模型进行进一步检测
        4. 如果布隆过滤器未检测到，则直接返回安全
        """
        self.logger.info(f"开始检查SQL注入 - 请求方法: {request.method}, URL: {request.url}")
        
        # 获取所有可能包含SQL注入的参数
        params = []
        
        # 检查URL参数
        url_args = list(request.args.values())
        url_arg_keys = list(request.args.keys())
        self.logger.debug(f"URL参数: {url_args}")
        self.logger.debug(f"URL参数名: {url_arg_keys}")
        params.extend(url_args)
        params.extend(url_arg_keys)
        
        # 检查URL路径
        self.logger.debug(f"URL路径: {request.path}")
        params.append(request.path)
        
        # 检查请求头
        headers = [(name, value) for name, value in request.headers]
        self.logger.debug(f"请求头: {headers}")
        for header_name, header_value in headers:
            params.extend([header_name, header_value])
        
        # 检查Cookie
        cookies = list(request.cookies.items())
        self.logger.debug(f"Cookies: {cookies}")
        params.extend(request.cookies.values())
        params.extend(request.cookies.keys())
        
        # 检查POST数据
        if request.method == 'POST':
            self.logger.debug(f"Content-Type: {request.content_type}")
            if request.is_json:
                self.logger.debug("检测到JSON数据")
                # 递归检查JSON数据
                def extract_values(obj):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            params.append(str(key))
                            if isinstance(value, (dict, list)):
                                extract_values(value)
                            else:
                                params.append(str(value))
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, (dict, list)):
                                extract_values(item)
                            else:
                                params.append(str(item))
                
                json_data = request.get_json()
                self.logger.debug(f"JSON数据: {json_data}")
                extract_values(json_data)
            else:
                form_data = list(request.form.items())
                self.logger.debug(f"表单数据: {form_data}")
                params.extend(request.form.values())
                params.extend(request.form.keys())
        
        # 预处理所有参数
        self.logger.info("开始预处理参数")
        params = self._preprocess_parameters(params)
        
        # 检查每个参数
        for param in params:
            if not param:  # 跳过空参数
                continue
                
            self.logger.debug(f"检查参数: {param}")
            
            # 首先使用布隆过滤器快速检测
            if self.bloom_filter:
                bloom_result = self.bloom_filter.check_sql_injection(param)
                self.logger.debug(f"布隆过滤器检测结果: {bloom_result}, 参数: {param}")
                
                if bloom_result:
                    # 布隆过滤器检测到可疑，使用ML模型进行精确检测
                    self.logger.info(f"布隆过滤器检测到可疑参数，使用ML模型进行精确检测: {param}")
                    is_injection, confidence = self.ml_detector.detect(param)
                    self.logger.info(f"ML模型检测结果 - 是否SQL注入: {is_injection}, 置信度: {confidence}, 参数: {param}")
                    
                    if is_injection:
                        self.logger.warning(f"检测到SQL注入攻击, 参数: {param}, 置信度: {confidence}")
                        return True
                    else:
                        self.logger.info(f"布隆过滤器误报, 参数: {param}, ML模型置信度: {confidence}")
                else:
                    self.logger.debug(f"布隆过滤器未检测到可疑，判定为安全: {param}")
            else:
                self.logger.warning(f"布隆过滤器未启用，直接使用ML模型进行检测: {param}")
                is_injection, confidence = self.ml_detector.detect(param)
                self.logger.info(f"ML模型检测结果 - 是否SQL注入: {is_injection}, 置信度: {confidence}, 参数: {param}")
                
                if is_injection:
                    self.logger.warning(f"检测到SQL注入攻击, 参数: {param}, 置信度: {confidence}")
                    return True
        
        self.logger.info("SQL注入检测完成，未发现攻击")
        return False
    
    def _forward_request(self, request):
        """转发请求到后端服务"""
        # 构建目标URL
        target_url = urljoin(BACKEND_CONFIG['url'], request.path)
        
        # 准备请求参数
        kwargs = {
            'params': request.args,
            'headers': {k: v for k, v in request.headers if k.lower() != 'host'},
            'timeout': BACKEND_CONFIG['timeout']
        }
        
        # 添加请求体
        if request.method == 'POST':
            if request.is_json:
                kwargs['json'] = request.get_json()
            else:
                kwargs['data'] = request.form
        
        # 发送请求
        try:
            response = requests.request(
                method=request.method,
                url=target_url,
                **kwargs
            )
            response.raise_for_status()  # 检查状态码
            return response
        except requests.RequestException as e:
            self.logger.error(f"转发请求失败: {str(e)}", exc_info=True)
            raise
