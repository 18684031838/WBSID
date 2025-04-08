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
import re
import html
import urllib.parse

# 设置控制台输出编码为UTF-8
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    if sys.version_info[0] == 3:
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class RedisConnectionError(Exception):
    """Redis连接失败时抛出的异常"""
    pass

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
        self.logger = logging.getLogger(__name__)
        self.logger.info("正在连接Redis服务器...")
        
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
            #self.logger.info(f"Redis配置信息: {redis_config}")
            
            # 初始化Redis连接
            self.logger.info("正在测试Redis连接...")
            self.redis_client = Redis(**redis_config)
            
            # 测试连接
            pong = self.redis_client.ping()
            self.logger.info(f"Redis连接测试结果: {pong}")
            
            # 初始化布隆过滤器
            self.logger.info("正在初始化布隆过滤器...")
            self.bloom_filter = SQLInjectionBloomFilter()
            self.logger.info("布隆过滤器初始化成功")
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.logger.error(f"Redis连接失败: {str(e)}")
            raise RedisConnectionError(f"Redis连接失败: {str(e)}")
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
        
        self.app = app
        self.config = config or {}
        self.fallback_mode = False
        
        # 配置日志
        logging.config.dictConfig(LOGGING_CONFIG)
        
        # 初始化HTTP连接池
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
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
    
    def _preprocess_parameters(self, parameters):
        """预处理请求参数，清理和标准化输入

        Args:
            parameters: 请求参数列表

        Returns:
            处理后的参数列表
        """
        if not parameters:
            return []

        # SQL关键字列表
        sql_keywords = {
            r'SEL\s*E?\s*C?\s*T(?:\s*\*)?': 'SELECT',  # 匹配SELECT和SELECT *，但只保留SELECT
            r'UNI\s*O?\s*N': 'UNION',
            r'ORD\s*E?\s*R': 'ORDER',
            r'WH\s*E?\s*RE': 'WHERE',
            r'FR\s*O?\s*M': 'FROM',
            r'DR\s*O?\s*P': 'DROP',
            r'DE\s*LE?\s*TE': 'DELETE',
            r'UP\s*D?\s*A?\s*TE': 'UPDATE',
            r'IN\s*S?\s*E?\s*RT': 'INSERT',
            r'EXE\s*C?\s*U?\s*TE': 'EXECUTE'
        }

        processed = []
        for param in parameters:
            if not isinstance(param, str):
                continue

            # 1. 处理多重URL编码
            while '%' in param:
                prev_param = param
                param = urllib.parse.unquote(param)
                if prev_param == param:
                    break

            # 2. HTML实体解码
            param = html.unescape(param)

            # 3. 处理十六进制编码
            def hex_decode(match):
                try:
                    return bytes.fromhex(match.group(1)).decode('ascii')
                except (ValueError, UnicodeDecodeError):
                    return match.group(0)

            param = re.sub(r'0x([0-9a-fA-F]+)', hex_decode, param)

            # 4. 删除注释
            param = re.sub(r'/\*[\s\S]*?\*/', ' ', param)  # 多行注释
            param = re.sub(r'--[^\n]*', ' ', param)  # 单行注释
            param = re.sub(r'#[^\n]*', ' ', param)   # MySQL风格注释

            # 5. 处理特殊字符
            special_chars = ['\x00', '\x08', '\x09', '\x0a', '\x0d', '\x1a', '\xa0']
            for char in special_chars:
                param = param.replace(char, ' ')

            # 6. 标准化空白字符
            param = ' '.join(param.split())

            # 7. 标准化SQL关键字
            param_lower = param.lower()
            
            # 先处理SELECT *的情况
            select_star_pattern = r'\bSEL\s*E?\s*C?\s*T\s*\*'
            if re.search(select_star_pattern, param_lower, re.IGNORECASE):
                param = re.sub(select_star_pattern, 'SELECT', param, flags=re.IGNORECASE)
            
            # 处理其他SQL关键字
            for pattern, replacement in sql_keywords.items():
                # 使用正则的单词边界，避免误匹配
                full_pattern = fr'\b{pattern}\b'
                if re.search(full_pattern, param_lower, re.IGNORECASE):
                    param = re.sub(full_pattern, replacement, param, flags=re.IGNORECASE)

            processed.append(param)

        return processed
    
    def _check_sql_injection(self, request):
        """检查请求中是否存在SQL注入
        
        检测流程：
        1. 收集所有可能包含SQL注入的参数值，包括：
           - URL查询参数值
           - URL路径中的参数部分
           - POST表单数据值
           - JSON数据中的值
        2. 使用布隆过滤器进行快速检测
        3. 如果布隆过滤器检测到可疑，则使用机器学习模型进行进一步检测
        4. 如果布隆过滤器未检测到，则直接返回安全
        """
        start_time = time.time()
        try:
            # 获取所有请求参数值
            params = []
            
            # 检查URL路径中的参数部分
            path = request.path
            path_segments = path.split('/')
            # 检查路径中每个部分，主要关注可能包含参数的部分
            for i, segment in enumerate(path_segments):
                if not segment:
                    continue                                  
                params.append(segment)               
              
            # 检查URL查询参数值（只检查值不检查键）
            url_args = list(request.args.values())
            self.logger.debug(f"URL参数值: {url_args}")
            params.extend(url_args)
            
            # 检查POST数据
            if request.method == 'POST':
                self.logger.debug(f"Content-Type: {request.content_type}")
                if request.is_json:
                    self.logger.debug("检测到JSON数据")
                    # 递归提取JSON数据中的值（不包括键名）
                    def extract_values(obj):
                        values = []
                        if isinstance(obj, dict):
                            for value in obj.values():
                                if isinstance(value, (dict, list)):
                                    values.extend(extract_values(value))
                                else:
                                    values.append(str(value))
                        elif isinstance(obj, list):
                            for item in obj:
                                if isinstance(item, (dict, list)):
                                    values.extend(extract_values(item))
                                else:
                                    values.append(str(item))
                        return values
                    
                    json_data = request.get_json()
                    self.logger.debug(f"JSON数据: {json_data}")
                    params.extend(extract_values(json_data))
                else:
                    # 只获取表单值（不包括键名）
                    form_values = list(request.form.values())
                    self.logger.debug(f"表单数据值: {form_values}")
                    params.extend(form_values)
            
            # 预处理所有参数
            self.logger.info("开始预处理参数")
            processed_params = self._preprocess_parameters(params)
            
            # 检查每个参数
            for param in processed_params:
                if not param:  # 跳过空参数
                    continue
                    
                self.logger.debug(f"检查参数: {param}")
                
                # 首先使用布隆过滤器快速检测
                bloom_start = time.time()
                if self.bloom_filter:
                    bloom_result = self.bloom_filter.check_sql_injection(param)
                    bloom_time = time.time() - bloom_start
                    
                    if bloom_result:
                        self.logger.info(f"布隆过滤器检测到可疑参数 [耗时: {bloom_time:.3f}s]")
                        self.logger.debug(f"可疑参数: {param}")
                        
                        # 使用机器学习模型进行进一步检测
                        if self.ml_detector:
                            ml_start = time.time()
                            is_injection, confidence = self.ml_detector.detect({'param': param})
                            ml_time = time.time() - ml_start
                            
                            if is_injection and confidence >= self.ml_detector.confidence_threshold:
                                self.logger.warning(f"机器学习模型确认SQL注入 [置信度: {confidence:.3f}, 耗时: {ml_time:.3f}s]")
                                return True
                            else:
                                self.logger.info(f"机器学习模型判定为误报 [置信度: {confidence:.3f}, 耗时: {ml_time:.3f}s]")
                        else:
                            # 如果ML检测器不可用，则相信布隆过滤器的结果
                            return True
                else:
                    # 如果布隆过滤器不可用，则使用机器学习模型
                    if self.ml_detector:
                        ml_start = time.time()
                        is_injection, confidence = self.ml_detector.detect({'param': param})
                        ml_time = time.time() - ml_start
                        
                        if is_injection and confidence >= self.ml_detector.confidence_threshold:
                            self.logger.warning(f"机器学习模型检测到SQL注入 [置信度: {confidence:.3f}, 耗时: {ml_time:.3f}s]")
                            return True
            
            check_time = time.time() - start_time
            self.logger.info(f"SQL注入检测完成 [总耗时: {check_time:.3f}s]")
            return False
            
        except Exception as e:
            self.logger.error(f"SQL注入检测过程中发生错误: {str(e)}", exc_info=True)
            # 发生错误时返回False，因为我们不能确定是否存在注入
            return False
    
    def _forward_request(self, request):
        """高性能的请求转发方法"""
        target_url = urljoin(BACKEND_CONFIG['url'], request.path)
        
        # 预构建常用请求头
        headers = {
            k: v for k, v in request.headers 
            if k.lower() not in ['host', 'connection']
        }
        
        # 准备请求参数
        kwargs = {
            'params': request.args,
            'headers': headers,
            'timeout': BACKEND_CONFIG['timeout']
        }
        
        # 添加请求体
        if request.method == 'POST':
            if request.is_json:
                kwargs['json'] = request.get_json()
            else:
                kwargs['data'] = request.form
        
        try:
            # 使用连接池发送请求
            response = self.session.request(
                method=request.method,
                url=target_url,
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"转发请求失败: {str(e)}", exc_info=True)
            raise
