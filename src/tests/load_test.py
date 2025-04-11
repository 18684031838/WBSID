import csv
import random
import asyncio
import aiohttp
import time
import math
import os
import subprocess
from tqdm import tqdm
from collections import defaultdict
import urllib.parse

class TestMetrics:
    def __init__(self):
        self.total_requests = 0
        self.success_count = 0
        self.failure_count = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.response_times = []
        self.start_time = time.time()
        self.results = []
    
    def add_result(self, response, expected):
        self.total_requests += 1
        if response.get('success', False):
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # For detection metrics
        detected = response.get('is_malicious', False)
        if expected == 1 and detected:
            self.true_positives += 1
        elif expected == 1 and not detected:
            self.false_negatives += 1
        elif expected == 0 and not detected:
            self.true_negatives += 1
        elif expected == 0 and detected:
            self.false_positives += 1
        
        self.results.append({
            'expected': expected,
            'actual': 1 if detected else 0,
            'response_time': response.get('response_time', 0)
        })
    
    def add_response_time(self, response_time):
        self.response_times.append(response_time)
    
    def calculate_metrics(self):
        duration = time.time() - self.start_time
        qps = self.total_requests / duration if duration > 0 else 0
        
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        fpr = self.false_positives / (self.false_positives + self.true_negatives) if (self.false_positives + self.true_negatives) > 0 else 0
        
        avg_response_time = sum(self.response_times)/len(self.response_times) if self.response_times else 0
        
        return {
            'qps': qps,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'avg_response_time': avg_response_time,
            'success_rate': self.success_count / self.total_requests if self.total_requests > 0 else 0,
            'total_requests': self.total_requests,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives
        }

    def generate_report(self, scenario_name):
        """生成并保存CSV格式的测试报告"""
        report_time = time.strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_report_{scenario_name}_{report_time}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            writer.writerow([
                'Scenario', 'Total Requests', 'Success Rate', 
                'Avg Response Time (ms)', 'TP', 'FP', 'TN', 'FN',
                'Detection Accuracy', 'Test Duration (min)'
            ])
            
            # 计算统计数据
            success_rate = (self.success_count / self.total_requests) * 100 if self.total_requests > 0 else 0
            avg_response = sum(self.response_times)/len(self.response_times) if self.response_times else 0
            duration_min = (time.time() - self.start_time) / 60
            
            # 写入数据行
            writer.writerow([
                scenario_name, self.total_requests, f"{success_rate:.2f}%",
                f"{avg_response:.2f}", self.true_positives, self.false_positives,
                self.true_negatives, self.false_negatives,
                f"{(self.true_positives + self.true_negatives)/self.total_requests * 100:.2f}%",
                f"{duration_min:.2f}"
            ])
        
        return filename

async def make_request(session, url, data, use_bloom):
    """
    发送测试请求(GET方法)
    
    参数:
        session: aiohttp客户端会话
        url: 基础URL
        data: 查询参数
        use_bloom: 是否使用Bloom过滤器
    """
    try:
        # 构造完整请求URL
        target_url = f"{url}/products/unsafe-search/{urllib.parse.quote(data['name'])}"
        
        # 添加Bloom过滤器头
        headers = {'IsBloomFilter': '1' if use_bloom else '0'}
        
        async with session.get(target_url, headers=headers) as response:
            result = await response.json()
            return {
                'success': result.get('success', False),
                'is_malicious': result.get('is_malicious', False),
                'status': response.status
            }
    except Exception as e:
        return {'success': False, 'error': str(e), 'status': 500}

async def worker(session, url, test_data, use_bloom, metrics, interval_dist, pbar):
    for row in test_data:
        expected = 0  
        try:
            if row.get('is_malicious') not in (None, ''):
                expected = int(row['is_malicious'])
        except ValueError:
            expected = 0
            
        response = await make_request(session, url, row, use_bloom)
        metrics.add_result(response, expected)
        
        # 每次请求后固定延迟3秒
        await asyncio.sleep(3)
        
        pbar.update(1)

async def run_test(url, test_data, use_bloom, users=100, max_requests=None, interval_dist=None):
    """
    max_requests: 最大请求次数，None表示使用全部测试数据
    """
    print(f"Running {'with' if use_bloom else 'without'} Bloom filter - {users} users")
    metrics = TestMetrics()
    timeout = aiohttp.ClientTimeout(total=10)
    conn = aiohttp.TCPConnector(limit=users)
    
    if max_requests is None:
        max_requests = len(test_data)
    
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        workers = []
        with tqdm(total=max_requests, desc="Requests") as pbar:
            for i in range(users):
                worker_data = test_data[i::users][:max_requests//users]
                workers.append(worker(session, url, worker_data, use_bloom, metrics, interval_dist or {}, pbar))
            
            await asyncio.gather(*workers)
    
    return metrics

def load_test_data():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    
    if not os.path.exists(test_data_path):
        subprocess.run(['python', 'generate_test_data.py'], check=True)
    
    with open(test_data_path, 'r') as f:
        data = []
        for row in csv.DictReader(f):
            if 'name' not in row:
                continue
                
            malicious = row.get('is_malicious')
            if malicious is None or malicious == '':
                row['is_malicious'] = '0'
            else:
                try:
                    row['is_malicious'] = str(int(malicious))
                except ValueError:
                    row['is_malicious'] = '0'
            
            data.append(row)
            
        if not data:
            raise ValueError("没有有效的测试数据")
        return data

def generate_report(results, scenario_name):
    true_pos = sum(1 for r in results if r['expected'] == 1 and r['actual'] == 1)
    false_pos = sum(1 for r in results if r['expected'] == 0 and r['actual'] == 1)
    true_neg = sum(1 for r in results if r['expected'] == 0 and r['actual'] == 0)
    false_neg = sum(1 for r in results if r['expected'] == 1 and r['actual'] == 0)
    
    total = len(results)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
    
    report = f"""## {scenario_name} Test Report
    
### Detection Metrics
| Metric | Value |
|--------|-------|
| True Positives | {true_pos} |
| False Positives | {false_pos} |
| True Negatives | {true_neg} |
| False Negatives | {false_neg} |
| Precision | {precision:.2%} |
| Recall | {recall:.2%} |
| F1 Score | {f1_score:.2%} |
    
### Performance Metrics
| Metric | Value |
|--------|-------|
| Total Requests | {total} |
| Success Rate | {(true_pos + true_neg)/total:.2%} |
| Avg Response Time | {sum(r['response_time'] for r in results)/total:.4f}s |
"""
    
    with open('test_report.md', 'w') as f:
        f.write(report)
    
    return report

if __name__ == "__main__":
    test_data = load_test_data()
    url = "http://localhost:5000"  
    
    print("Running initial test (1 user, 10 requests, 2s interval)")
    initial_test = asyncio.run(run_test(
        url="http://localhost:5000",  # 中间件服务
        test_data=test_data,
        use_bloom=True,
        users=1,
        max_requests=10,
        interval_dist={'type': 'fixed', 'value': 2}  # 2秒间隔
    ))
    generate_report(initial_test.results, "Initial Test (1 user, 10 requests)")

    print("\nRunning Normal Load Test (100 users, 5000 requests)")
    normal_test = asyncio.run(run_test(
        url="http://localhost:5000",
        test_data=test_data,
        use_bloom=True,
        users=100,
        max_requests=5000,
        interval_dist={'type': 'random', 'max': 1.0}
    ))
    generate_report(normal_test.results, "Normal Load Test")

    print("\nRunning Normal Load Test (100 users, 5min, random intervals <1s)")
    bloom_normal = asyncio.run(run_test(
        url, test_data, True, 
        users=100, 
        max_requests=5000,
        interval_dist={'type': 'random', 'max': 1.0}
    ))
    generate_report(bloom_normal.results, "Normal Load Test (100 users, 5min)")
    
    nobloom_normal = asyncio.run(run_test(
        url, test_data, False, 
        users=100, 
        max_requests=5000,
        interval_dist={'type': 'random', 'max': 1.0}
    ))
    generate_report(nobloom_normal.results, "Normal Load Test (100 users, 5min) Without Bloom")
    
    print("\nRunning Peak Load Test (1000 users, 10min, fixed 0.1s intervals)")
    bloom_peak = asyncio.run(run_test(
        url, test_data, True, 
        users=1000, 
        max_requests=10000,
        interval_dist={'type': 'fixed', 'value': 0.1}
    ))
    generate_report(bloom_peak.results, "Peak Load Test (1000 users, 10min)")
    
    nobloom_peak = asyncio.run(run_test(
        url, test_data, False, 
        users=1000, 
        max_requests=10000,
        interval_dist={'type': 'fixed', 'value': 0.1}
    ))
    generate_report(nobloom_peak.results, "Peak Load Test (1000 users, 10min) Without Bloom")
    
    print("\nTest completed! Markdown report saved to test_report.md")
