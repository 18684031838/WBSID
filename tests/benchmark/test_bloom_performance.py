import pytest

def test_bloom_filter_insert_performance(benchmark):
    from src.sql_injection_middleware.bloom_filter import BloomFilter
    bf = BloomFilter(capacity=10_000, error_rate=0.001)
    
    def setup():
        return ["sql_keyword_{}".format(i) for i in range(1000)], {}
    
    samples, _ = benchmark.pedantic(bf.add, setup=setup, rounds=100)
    assert len(samples) == 1000
