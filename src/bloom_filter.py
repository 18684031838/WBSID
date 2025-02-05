import redis
from pybloom_live import ScalableBloomFilter

class SQLInjectionBloomFilter:
    """基于Redis的SQL注入关键词布隆过滤器
    
    使用布隆过滤器存储SQL注入的关键词，提供快速的检测能力
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db
        )
        self.bloom_filter = ScalableBloomFilter(
            initial_capacity=1000,
            error_rate=0.001
        )
        self._initialize_keywords()
        
    def _initialize_keywords(self):
        """初始化SQL注入关键词
        
        将常见的SQL注入关键词添加到布隆过滤器中
        """
        sql_keywords = {
            # 基本SQL关键词
            'SELECT', 'UNION', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'EXEC',
            'EXECUTE', 'TRUNCATE', 'INTO', 'DISTINCT', 'GROUP BY', 'ORDER BY',
            'HAVING', 'WHERE',
            
            # 特殊字符和操作符
            '--', '/*', '*/', ';', '@@', '@', '||', '|', '&',
            
            # 函数名
            'CHAR', 'CONCAT', 'CAST', 'CONVERT', 'DELAY', 'WAITFOR',
            'BENCHMARK', 'SLEEP', 'IF', 'SUBSTR', 'MID', 'VERSION',
            'DATABASE', 'USER', 'SYSTEM_USER', 'SESSION_USER',
            
            # 特定攻击模式
            'OR 1=1', 'OR TRUE', 'OR \'1\'=\'1', 'AND 1=1', 'AND TRUE',
            '1 OR 1=1', 'TRUE OR TRUE', '1 AND 1=1', 'TRUE AND TRUE',
            
            # 数据库特定关键词
            'INFORMATION_SCHEMA', 'SYSOBJECTS', 'SYSUSERS', 'SYSTEM_USER',
            'TABLE_NAME', 'COLUMN_NAME', 'TABLE_SCHEMA', 'SCHEMA_NAME'
        }
        
        # 添加关键词到布隆过滤器
        for keyword in sql_keywords:
            self.bloom_filter.add(keyword.lower())
            
    def may_contain_injection(self, params):
        """检查参数中是否可能包含SQL注入
        
        Args:
            params: dict, 包含请求参数的字典
            
        Returns:
            bool: True表示可能包含SQL注入，False表示安全
        """
        for value in params.values():
            if not isinstance(value, str):
                continue
                
            value = value.lower()
            # 分割参数值为单词
            words = value.split()
            
            # 检查每个单词是否在布隆过滤器中
            for word in words:
                if self.bloom_filter.add(word):
                    return True
                    
            # 检查完整的参数值
            if self.bloom_filter.add(value):
                return True
                
        return False
