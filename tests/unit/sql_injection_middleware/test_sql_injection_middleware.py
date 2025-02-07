import pytest
from src.sql_injection_middleware.sql_injection_middleware import SQLInjectionMiddleware, RedisConnectionError
from werkzeug.wrappers import Request
from werkzeug.test import EnvironBuilder
import urllib.parse
import html
import logging
import redis
import io

class TestHandler(logging.Handler):
    """用于测试的日志处理器"""
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def reset(self):
        self.records = []

@pytest.fixture
def test_logger():
    """创建测试用的日志处理器"""
    handler = TestHandler()
    logger = logging.getLogger('src.sql_injection_middleware.sql_injection_middleware')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    handler.reset()

class TestSQLInjectionMiddlewarePreprocessing:
    @pytest.fixture
    def middleware(self):
        middleware = SQLInjectionMiddleware()
        middleware.redis_conn = None
        return middleware

    @pytest.fixture
    def mock_middleware(self, mocker):
        """创建中间件实例的fixture"""
        try:
            # Mock Redis和布隆过滤器
            mocker.patch('redis.Redis.ping', return_value=True)
            mocker.patch('src.sql_injection_middleware.sql_injection_middleware.SQLInjectionBloomFilter')
            middleware = SQLInjectionMiddleware()
            return middleware
        except RedisConnectionError:
            # 如果Redis连接失败，返回None
            return None

    @pytest.fixture
    def dummy_request(self):
        """创建测试用的请求对象"""
        def _request(method='GET', query=None, data=None, json=None):
            if json is not None:
                environ = EnvironBuilder(method=method, query_string=query, json=json).get_environ()
            else:
                environ = EnvironBuilder(method=method, query_string=query, data=data).get_environ()
            return Request(environ)
        return _request

    @pytest.mark.parametrize("input_params, expected", [
        (["%20AND%201=1", "O'Neil"], ["AND 1=1", "O'Neil"]),
        (["%2520SELECT%2520*", "%2540var"], ["SELECT", "@var"]),
        (["&#34; DROP TABLE", "&#39; OR 1=1"], ['" DROP TABLE', "' OR 1=1"]),
        (["0x4F52444552", "0x53454C454354"], ["ORDER", "SELECT"]),
        (["SEL/*comment*/ECT", "UNI\x00ON"], ["SELECT", "UNION"]),
        (["; --\n", "|| ls -la"], [";", "|| ls -la"]),
        (["%252f%252a*/UNION%20%23blah%0ASELECT", "admin' OR &#40;1=1"],
         ["UNION SELECT", "admin' OR (1=1"])
    ])
    def test_parameter_preprocessing(self, mock_middleware, dummy_request, input_params, expected):
        # Test with query parameters
        test_request = dummy_request(data={'q': input_params})
        raw_params = input_params
        processed = mock_middleware._preprocess_parameters(raw_params)
        assert processed == expected

        # Test with form data
        test_request = dummy_request(data={'input': input_params})
        raw_params = input_params
        processed = mock_middleware._preprocess_parameters(raw_params)
        assert processed == expected

        # Test with JSON data
        test_request = dummy_request(json={'data': input_params})
        raw_params = input_params
        processed = mock_middleware._preprocess_parameters(raw_params)
        assert processed == expected

    def test_redis_connection_failure(self, mocker, test_logger):
        """测试Redis连接失败的情况"""
        # Mock Redis以模拟连接失败
        mocker.patch('redis.Redis.ping', side_effect=redis.ConnectionError("Connection refused"))

        # 验证创建中间件实例时会抛出RedisConnectionError异常
        with pytest.raises(RedisConnectionError) as excinfo:
            SQLInjectionMiddleware()

        # 验证异常消息包含原始错误信息
        assert "Connection refused" in str(excinfo.value)

        # 验证错误日志被记录
        error_logs = [record for record in test_logger.records
                     if record.levelno == logging.ERROR and "Redis连接失败" in record.msg]
        assert len(error_logs) > 0, "没有找到预期的错误日志消息"
        assert "Connection refused" in error_logs[0].msg
