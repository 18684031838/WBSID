import csv
import random
import os
from faker import Faker

# 多样化的SQL注入攻击向量
SQL_INJECTIONS = [
    # 基础注入
    "' OR '1'='1",
    "' OR 1=1--",
    "'; DROP TABLE users--",
    "' UNION SELECT * FROM users--",
    
    # 编码变换
    "' OR 0x50=0x50--",  # HEX编码
    "' OR CHAR(49)=CHAR(49)--",  # CHAR函数
    "' OR BINARY 'a'='a'--",  # BINARY操作符
    "' OR SOUNDEX('test')=SOUNDEX('test')--",  # SOUNDEX函数
    
    # 时间延迟注入
    "' OR (SELECT * FROM (SELECT(SLEEP(5)))bAKL)--",
    "' OR BENCHMARK(10000000,MD5(NOW()))--",
    
    # 混淆技术
    "'/**/OR/**/'1'='1",  # 注释混淆
    "'||'1'='1",  # 管道符
    "'%20OR%201=1--",  # URL编码
    "'%27%20OR%201=1--",  # 单引号URL编码
    
    # 高级注入技术
    "'; EXEC xp_cmdshell('dir')--",  # MSSQL命令执行
    "' UNION ALL SELECT LOAD_FILE('/etc/passwd')--",  # 文件读取
    "' AND (SELECT 1 FROM(SELECT COUNT(*),CONCAT(0x3a,(SELECT CURRENT_USER()),0x3a,FLOOR(RAND(0)*2))x FROM INFORMATION_SCHEMA.PLUGINS GROUP BY x)a)--",  # 基于错误的注入
    
    # 二阶注入
    "'+(SELECT '0' FROM users WHERE username='admin' AND SUBSTRING(password,1,1)='a')+'"
]

def encode_injection(injection):
    """随机应用编码变换"""
    if random.random() > 0.7:  # 30%概率应用编码
        encoding_type = random.choice(['hex', 'url', 'char'])
        if encoding_type == 'hex':
            return ''.join(f'0x{ord(c):02x}' for c in injection)
        elif encoding_type == 'url':
            return ''.join(f'%{ord(c):02x}' if c not in ('-', ' ') else c for c in injection)
        elif encoding_type == 'char':
            return ''.join(f'CHAR({ord(c)})' for c in injection)
    return injection

def generate_normal_username(fake):
    return fake.user_name()

def generate_malicious_username(fake):
    # 选择注入类型
    injection = random.choice(SQL_INJECTIONS)
    
    # 随机应用编码变换
    injection = encode_injection(injection)
    
    # 50%概率混合正常用户名，50%概率纯注入
    if random.random() > 0.5:
        return fake.user_name() + injection
    else:
        return injection

def generate_test_data():
    """生成1万条测试数据(15% SQL注入)"""
    fake = Faker()
    total = 10000
    malicious_count = int(total * 0.15)  # 15% SQL注入
    normal_count = total - malicious_count  # 85%正常请求
    
    data = []
    
    # 生成正常请求
    for _ in range(normal_count):
        data.append({
            'name': generate_normal_username(fake),
            'is_malicious': '0'
        })
    
    # 生成恶意请求
    for _ in range(malicious_count):
        data.append({
            'name': generate_malicious_username(fake),
            'is_malicious': '1'
        })
    
    random.shuffle(data)
    
    # 使用绝对路径确保写入正确位置
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data.csv')
    
    with open(test_data_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'is_malicious'])
        writer.writeheader()
        
        count = 0
        for row in data:
            writer.writerow(row)
            count += 1
            
        print(f"成功生成 {count} 条测试数据到 {test_data_path}")
        assert count == 10000, "生成数据量不符合预期"

def main():
    generate_test_data()

if __name__ == "__main__":
    main()
