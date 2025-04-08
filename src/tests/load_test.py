import csv
import random
import asyncio
import aiohttp
import time
import math
from tqdm import tqdm
from collections import defaultdict

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

async def make_request(session, url, row, use_bloom):
    headers = {'IsBloomFilter': '1' if use_bloom else '0'}
    endpoint = f"{url}/products/unsafe-search/{row['username']}"
    start_time = time.time()
    try:
        async with session.get(endpoint, headers=headers) as resp:
            response = await resp.json()
            response_time = time.time() - start_time
            return response, response_time
    except Exception as e:
        return {'success': False, 'error': str(e)}, time.time() - start_time

async def worker(session, url, test_data, use_bloom, metrics, interval_dist):
    while True:
        row = random.choice(test_data)
        expected = int(row['is_malicious'])
        response, response_time = await make_request(session, url, row, use_bloom)
        metrics.add_result(response, expected)
        metrics.add_response_time(response_time)
        
        if interval_dist['type'] == 'fixed':
            await asyncio.sleep(interval_dist['value'])
        elif interval_dist['type'] == 'random':
            await asyncio.sleep(random.uniform(0, interval_dist['max']))
        elif interval_dist['type'] == 'exponential':
            await asyncio.sleep(random.expovariate(1.0/interval_dist['mean']))

async def run_test(url, test_data, use_bloom, users, duration_minutes, interval_dist):
    metrics = TestMetrics()
    timeout = aiohttp.ClientTimeout(total=10)
    conn = aiohttp.TCPConnector(limit=users)
    
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        workers = [worker(session, url, test_data, use_bloom, metrics, interval_dist) 
                  for _ in range(users)]
        
        duration_seconds = duration_minutes * 60
        await asyncio.wait_for(asyncio.gather(*workers), timeout=duration_seconds)
    
    return metrics.calculate_metrics()

def load_test_data():
    with open('test_data.csv', 'r') as f:
        return [{'username': row['username'], 'is_malicious': row['is_malicious']} for row in csv.DictReader(f)]

def generate_report(results, scenario_name):
    report = f"""
## {scenario_name}

### Performance Metrics
| Metric | Value |
|--------|-------|
| QPS | {results['qps']:.2f} |
| Avg Response Time | {results['avg_response_time']:.4f}s |
| Success Rate | {results['success_rate']:.2%} |

### Detection Metrics
| Metric | Value |
|--------|-------|
| Precision | {results['precision']:.2%} |
| Recall | {results['recall']:.2%} |
| False Positive Rate | {results['fpr']:.2%} |

### Detection Counts
| Type | Count |
|------|-------|
| True Positives | {results['true_positives']} |
| False Positives | {results['false_positives']} |
| True Negatives | {results['true_negatives']} |
| False Negatives | {results['false_negatives']} |
"""
    return report

if __name__ == "__main__":
    test_data = load_test_data()
    url = "http://localhost:5000"  # Base URL without endpoint
    
    # Scenario 1: Concurrency Test
    print("Running Concurrency Test (100 users, 30min, random intervals)")
    bloom_concurrency = asyncio.run(run_test(
        url, test_data, True, 
        users=100, 
        duration_minutes=30,
        interval_dist={'type': 'random', 'max': 1.0}
    ))
    
    nobloom_concurrency = asyncio.run(run_test(
        url, test_data, False, 
        users=100, 
        duration_minutes=30,
        interval_dist={'type': 'random', 'max': 1.0}
    ))
    
    # Scenario 2: Peak Load Test
    print("\nRunning Peak Load Test (1000 users, 10min, fixed 0.1s intervals)")
    bloom_peak = asyncio.run(run_test(
        url, test_data, True, 
        users=1000, 
        duration_minutes=10,
        interval_dist={'type': 'fixed', 'value': 0.1}
    ))
    
    nobloom_peak = asyncio.run(run_test(
        url, test_data, False, 
        users=1000, 
        duration_minutes=10,
        interval_dist={'type': 'fixed', 'value': 0.1}
    ))
    
    # Scenario 3: Stability Test
    print("\nRunning Stability Test (500 users, 60min, exponential intervals)")
    bloom_stability = asyncio.run(run_test(
        url, test_data, True, 
        users=500, 
        duration_minutes=60,
        interval_dist={'type': 'exponential', 'mean': 0.5}
    ))
    
    nobloom_stability = asyncio.run(run_test(
        url, test_data, False, 
        users=500, 
        duration_minutes=60,
        interval_dist={'type': 'exponential', 'mean': 0.5}
    ))
    
    # Generate reports
    with open('test_report.md', 'w') as f:
        f.write("# SQL Injection Detection Test Report\n\n")
        f.write("## Bloom Filter + ML Model\n")
        f.write(generate_report(bloom_concurrency, "Concurrency Test (100 users, 30min)"))
        f.write(generate_report(bloom_peak, "Peak Load Test (1000 users, 10min)"))
        f.write(generate_report(bloom_stability, "Stability Test (500 users, 60min)"))
        
        f.write("\n## ML Model Only\n")
        f.write(generate_report(nobloom_concurrency, "Concurrency Test (100 users, 30min)"))
        f.write(generate_report(nobloom_peak, "Peak Load Test (1000 users, 10min)"))
        f.write(generate_report(nobloom_stability, "Stability Test (500 users, 60min)"))
    
    print("\nTest completed! Markdown report saved to test_report.md")
