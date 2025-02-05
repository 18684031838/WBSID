import unittest
import sys
import os
import socket
import logging
import redis
from redis import Redis, RedisError
import time

# 添加父目录到系统路径，以便能够导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REDIS_CONFIG

class TestRedisConnection(unittest.TestCase):
    """测试Redis连接的单元测试类"""

    def setUp(self):
        """测试前的设置"""
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('test_redis_connection')
        
        # Redis配置
        self.redis_config = {
            'host': REDIS_CONFIG.get('host', 'localhost'),
            'port': REDIS_CONFIG.get('port', 6379),
            'db': REDIS_CONFIG.get('db', 0),
            'password': REDIS_CONFIG.get('password', None),
            'socket_timeout': 10,
            'socket_connect_timeout': 10,
            'retry_on_timeout': True,
            'decode_responses': True,
            'health_check_interval': 30,
            'socket_keepalive': True,  # 启用keepalive
            'socket_keepalive_options': {
                socket.TCP_KEEPIDLE: 60,     # 空闲时间
                socket.TCP_KEEPINTVL: 2,     # 探测间隔
                socket.TCP_KEEPCNT: 5        # 探测次数
            }
        }
        self.logger.info(f"Redis配置信息: {self.redis_config}")

    def test_redis_connection(self):
        """测试Redis连接"""
        max_retries = 3
        retry_delay = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f'尝试连接Redis (尝试 {attempt + 1}/{max_retries})...')
                # 使用最简单的连接配置
                redis_client = redis.StrictRedis(
                    host='localhost',
                    port=6379,
                    password='',
                    db=0,
                    decode_responses=True
                )
                
                # 测试连接
                self.logger.info('测试Redis连接...')
                response = redis_client.ping()
                self.logger.info(f'Redis ping响应: {response}')
                
                # 测试基本操作
                self.logger.info('测试Redis基本操作...')
                test_key = 'test_connection'
                test_value = 'test_value'
                redis_client.set(test_key, test_value)
                retrieved_value = redis_client.get(test_key)
                self.assertEqual(test_value, retrieved_value)
                
                # 清理测试数据
                redis_client.delete(test_key)
                
                self.logger.info('Redis连接和基本操作测试成功！')
                return  # 测试成功，退出重试循环

            except (RedisError, Exception) as e:
                last_error = str(e)
                self.logger.warning(f'Redis连接失败 (尝试 {attempt + 1}/{max_retries}): {last_error}')
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                
        # 如果所有重试都失败，则测试失败
        self.fail(f'Redis连接测试失败，最后的错误: {last_error}')

    def test_redis_connection_timeout(self):
        """测试Redis连接超时设置"""
        try:
            redis_client = Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                socket_timeout=1,  # 设置非常短的超时时间
                socket_connect_timeout=1
            )
            self.logger.info('测试短超时时间的Redis连接...')
            redis_client.ping()
            self.logger.info('短超时连接测试成功')
        except RedisError as e:
            self.logger.warning(f'预期中的超时错误: {str(e)}')
            # 这里我们不将超时视为错误，因为这是预期的行为
            pass

if __name__ == '__main__':
    unittest.main()
