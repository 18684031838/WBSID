"""
基于布隆过滤器的SQL注入检测
使用Redis作为存储后端
"""
import redis
import logging
import sys
import urllib.parse
import html
import re
import mmh3
from .config import REDIS_CONFIG

# 设置控制台输出编码为UTF-8
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    if sys.version_info[0] == 3:
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class SQLInjectionBloomFilter:
    """SQL注入布隆过滤器"""
    
    def __init__(self, redis_host=None, redis_port=None, redis_password=None, redis_db=None):
        """初始化布隆过滤器
        
        Args:
            redis_host: Redis服务器地址，如果为None则使用配置文件中的值
            redis_port: Redis服务器端口，如果为None则使用配置文件中的值
            redis_password: Redis服务器密码，如果为None则使用配置文件中的值
            redis_db: Redis数据库编号，如果为None则使用配置文件中的值
        """
        self.logger = logging.getLogger('sql_injection_middleware.bloom_filter')
        
        try:
            # 获取Redis配置
            redis_config = {
                'host': redis_host or REDIS_CONFIG.get('host', 'localhost'),
                'port': redis_port or REDIS_CONFIG.get('port', 6379),
                'db': redis_db or REDIS_CONFIG.get('db', 0),
                'password': redis_password if redis_password is not None else REDIS_CONFIG.get('password', None),
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'retry_on_timeout': True,
                'decode_responses': True  # 添加此配置以正确处理中文
            }
            
            # 打印Redis配置信息
            safe_config = redis_config.copy()
            if safe_config['password']:
                safe_config['password'] = '******'
            #self.logger.info(f"布隆过滤器Redis配置: {safe_config}")
            
            # 初始化Redis连接
            self.logger.info("正在连接Redis服务器...")
            self.redis_client = redis.Redis(**redis_config)
            
            # 测试连接
            self.logger.info("正在测试Redis连接...")
            pong = self.redis_client.ping()
            self.logger.info(f"Redis连接测试结果: {pong}")
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.logger.error(f"Redis连接失败: {str(e)}")
            self.redis_client = None
            raise
        except Exception as e:
            self.logger.error(f"Redis初始化时发生未知错误: {str(e)}", exc_info=True)
            self.redis_client = None
            raise
            
        self.key = 'sql_injection_bloom'
        self.num_hash_functions = 6
        self.bit_size = 2**20  # 1MB的位数组
        
        # SQL注入特征模式
        self.sql_patterns = [
            # 1. 基本SQL注入
            r"'\s*OR\s*'?[0-9]+\s*'?\s*=\s*'?[0-9]+\s*'?",  # ' OR '1'='1
            r"'\s*OR\s*'.*?'=\s*'.*?'",  # ' OR 'x'='x
            r"'\s*OR\s*'.*?'\s*LIKE\s*'.*?'",  # ' OR 'x' LIKE 'x
            r"'\s*OR\s*[a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_]+",  # ' OR username=username
            r"'\s*OR\s*'1'\s*=\s*'1'",  # ' OR '1' = '1
            r"'\s*OR\s*1\s*=\s*1",      # ' OR 1 = 1
            r"'\s*OR\s*'1'\s*=\s*'1",    # ' OR '1' = '1
            r"'\s*OR\s*'1\s*'=\s*'1'",   # ' OR '1 '='1'
            r"'\s*OR\s*1\s*=\s*1\s*--",  # ' OR 1 = 1 --
            
            # 2. UNION注入
            r"UNION\s+ALL\s+SELECT",
            r"UNION\s+SELECT",
            r"SELECT\s+FROM",
            r"INSERT\s+INTO",
            r"UPDATE\s+.*?\s+SET",
            r"DELETE\s+FROM",
            
            # 3. 注释符号和终止符
            r"--\s+",
            r"#\s*$",
            r"/\*.*?\*/",
            r";.*?$",           # 语句终止符后跟其他语句
            
            # 4. 系统表和函数
            r"INFORMATION_SCHEMA\.",
            r"SYSOBJECTS",
            r"SYSCOLUMNS",
            r"@@VERSION",
            r"USER\s*\(\s*\)",
            r"DATABASE\s*\(\s*\)",
            r"SYSTEM_USER",
            r"CURRENT_USER",
            
            # 5. 条件注入
            r"AND\s+1\s*=\s*1",
            r"AND\s+'1'\s*=\s*'1'",
            r"OR\s+1\s*=\s*1",
            r"OR\s+'1'\s*=\s*'1'",
            
            # 6. 时间延迟注入
            r"WAITFOR\s+DELAY",
            r"SLEEP\s*\(\s*\d+\s*\)",
            r"BENCHMARK\s*\(\s*\d+\s*,",
            
            # 7. 堆叠注入
            r";\s*SELECT",
            r";\s*INSERT",
            r";\s*UPDATE",
            r";\s*DELETE",
            
            # 8. 特殊字符和编码
            r"%27",      # URL编码的单引号
            r"%22",      # URL编码的双引号
            r"%2527",    # 双重URL编码的单引号
            r"\x27",     # 十六进制编码的单引号
            
            # 9. 常见的绕过技术
            r"/\*!.*?\*/",     # MySQL版本注释
            r"/\*!50000.*?\*/", # 特定MySQL版本注释
            r"\/\*[^*]*\*+([^/*][^*]*\*+)*\/", # 内联注释
            r";\s*--",         # 注释行终止
            r";\s*#",          # Hash注释终止
            r"\|\|",           # Oracle连接符
            r"&&",             # 逻辑与
            
            # 10. 空白字符变体
            r"\s+OR\s+'1'\s*=\s*'1'",    # 带有额外空白
            r"\t+OR\t+'1'\t*=\t*'1'",    # 制表符
            r"\n+OR\n+'1'\n*=\n*'1'",    # 换行符
            
            # 11. 大小写混合
            r"(?i)OR\s+'1'\s*=\s*'1'",   # 不区分大小写
            r"(?i)UNION\s+SELECT",
            r"(?i)SELECT\s+FROM",
        ]
        
        # 预加载常见SQL注入特征
        self._init_patterns()
        
        self.logger.info("布隆过滤器初始化完成")
    
    def _init_patterns(self):
        """初始化常见SQL注入特征"""
        patterns = [
            "' OR '1'='1",
            "' OR 1=1--",
            "'; DROP TABLE",
            "UNION SELECT",
            "INFORMATION_SCHEMA",
            "@@version",
            "' UNION ALL SELECT",
            "AND 1=1--",
            "' OR 'x'='x",
            "; SELECT *",
            "admin'--",
            "1' OR '1'='1",
            "1 OR 1=1",
            "1' OR '1'='1'#",
            "1' OR '1'='1'/*",
            "' OR 1=1#",
            "' OR 1=1/*",
            "') OR '1'='1'--",
            "') OR ('1'='1'--",
            "1' ORDER BY 1--",
            "1' ORDER BY 2--",
            "1' ORDER BY 3--",
            "1' GROUP BY 1--",
            "1' GROUP BY 2--",
            "1' GROUP BY 3--",
            "' HAVING 1=1--",
            "' HAVING '1'='1'--",
            "' GROUP BY columnnames having 1=1--",
            "' OR '1'='1' UNION ALL SELECT 1--",
            "' OR '1'='1' UNION ALL SELECT 1,2--",
            "' OR '1'='1' UNION ALL SELECT 1,2,3--"
        ]
        for pattern in patterns:
            self.add(pattern.lower())
    
    def _normalize_input(self, input_str):
        """对输入进行标准化处理
        
        处理各种编码和转义:
        1. URL解码
        2. HTML实体解码
        3. Unicode标准化
        4. 移除注释
        5. 统一空白字符
        6. 处理各种编码变体
        """
        if not isinstance(input_str, str):
            input_str = str(input_str)
            
        # URL解码（可能多层编码）
        for _ in range(3):  # 最多解码3层
            decoded = urllib.parse.unquote(input_str)
            if decoded == input_str:
                break
            input_str = decoded
            
        # HTML实体解码
        input_str = html.unescape(input_str)
        
        # 移除注释
        input_str = re.sub(r'/\*.*?\*/', ' ', input_str)
        input_str = re.sub(r'--.*?$', ' ', input_str, flags=re.MULTILINE)
        
        # 统一空白字符
        input_str = re.sub(r'\s+', ' ', input_str)
        
        # 处理十六进制编码
        input_str = re.sub(r'0x[0-9a-fA-F]+', lambda m: str(int(m.group(0)[2:], 16)), input_str)
        
        # 处理char编码
        def replace_char(match):
            try:
                nums = [int(x.strip()) for x in match.group(1).split(',')]
                return ''.join(chr(x) for x in nums)
            except:
                return match.group(0)
        input_str = re.sub(r'CHAR\(([^)]+)\)', replace_char, input_str, flags=re.IGNORECASE)
        
        return input_str.lower().strip()
    
    def _get_hash_values(self, item):
        """获取一个项的多个哈希值"""
        hash_values = []
        for seed in range(self.num_hash_functions):
            hash_val = mmh3.hash(str(item), seed) % self.bit_size
            hash_values.append(hash_val)
        return hash_values
    
    def add(self, item):
        """添加一个项到布隆过滤器"""
        if not self.redis_client:
            return
            
        try:
            item = self._normalize_input(item)
            for bit_pos in self._get_hash_values(item):
                self.redis_client.setbit(self.key, bit_pos, 1)
        except redis.RedisError:
            pass
    
    def contains(self, item):
        """检查一个项是否可能在集合中"""
        if not self.redis_client:
            return False
            
        try:
            item = self._normalize_input(item)
            for bit_pos in self._get_hash_values(item):
                if not self.redis_client.getbit(self.key, bit_pos):
                    return False
            return True
        except redis.RedisError:
            return False
    
    def check_sql_injection(self, query):
        """检查SQL查询是否包含注入特征
        
        Args:
            query: SQL查询字符串
            
        Returns:
            bool: 如果可能包含SQL注入则返回True
        """
        if not self.redis_client:
            return False
            
        try:
            # 标准化输入
            query = self._normalize_input(query)
            
            # 1. 检查完整查询
            if self.contains(query):
                return True
                
            # 2. 检查查询中的片段
            words = query.split()
            for i in range(len(words)):
                for j in range(i + 1, min(i + 6, len(words) + 1)):  # 增加检查窗口大小
                    fragment = ' '.join(words[i:j])
                    if self.contains(fragment):
                        return True
            
            # 3. 使用正则模式匹配
            for pattern in self.sql_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return True
            
            return False
        except redis.RedisError:
            return False
