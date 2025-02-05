"""
Word2Vec feature extractor module
实现了基于Word2Vec的SQL注入特征提取
"""
from gensim.models import Word2Vec
import numpy as np
from collections import defaultdict
import math
from .base import BaseFeatureExtractor

class Word2VecExtractor(BaseFeatureExtractor):
    """Word2Vec feature extractor for SQL injection detection"""
    
    def __init__(self, 
                 vector_size=100,
                 window=5,
                 min_count=1,
                 negative=5,
                 ns_exponent=0.75,
                 hs=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.hs = hs
        self.model = None
        self.vocab_freqs = defaultdict(int)
        
    def _preprocess(self, X):
        """预处理输入数据
        
        Args:
            X: array-like of shape (n_samples,)
            
        Returns:
            list of list of str
        """
        sentences = []
        for x in X:
            # 分词并转换为小写
            tokens = str(x).lower().split()
            # 更新词频统计
            for token in tokens:
                self.vocab_freqs[token] += 1
            sentences.append(tokens)
        return sentences
        
    def _build_huffman_tree(self):
        """构建Huffman树用于层次化Softmax
        
        Returns:
            dict: 词到Huffman编码的映射
        """
        # 构建叶子节点
        nodes = [(freq, word) for word, freq in self.vocab_freqs.items()]
        nodes.sort(key=lambda x: x[0], reverse=True)  # 按频率排序
        
        # 构建Huffman树
        huffman_tree = {}
        while len(nodes) > 1:
            freq1, word1 = nodes.pop()
            freq2, word2 = nodes.pop()
            
            # 创建新的内部节点
            internal = (freq1 + freq2, [word1, word2])
            nodes.append(internal)
            nodes.sort(key=lambda x: x[0], reverse=True)  # 按频率排序
            
            # 更新Huffman编码
            if isinstance(word1, str):
                if word1 in huffman_tree:
                    huffman_tree[word1] = '0' + huffman_tree[word1]
                else:
                    huffman_tree[word1] = '0'
            
            if isinstance(word2, str):
                if word2 in huffman_tree:
                    huffman_tree[word2] = '1' + huffman_tree[word2]
                else:
                    huffman_tree[word2] = '1'
                
        return huffman_tree
        
    def _create_negative_table(self, table_size=100000):
        """创建负采样表
        
        Args:
            table_size: int, 采样表大小
            
        Returns:
            np.ndarray: 负采样表
        """
        # 计算总频率
        total_freq = float(sum(self.vocab_freqs.values()))
        
        # 计算每个词的采样概率
        probs = {word: (freq/total_freq)**self.ns_exponent 
                for word, freq in self.vocab_freqs.items()}
        
        # 归一化概率
        prob_sum = float(sum(probs.values()))
        probs = {word: prob/prob_sum for word, prob in probs.items()}
        
        # 创建采样表（使用整数索引而不是字符串）
        word_to_idx = {word: idx for idx, word in enumerate(probs.keys())}
        negative_table = []
        for word, prob in probs.items():
            negative_table.extend([word_to_idx[word]] * int(prob * table_size))
            
        # 确保表大小正确
        if len(negative_table) > table_size:
            negative_table = negative_table[:table_size]
        elif len(negative_table) < table_size:
            # 如果表太小，复制最后一个元素
            negative_table.extend([negative_table[-1]] * (table_size - len(negative_table)))
            
        return np.array(negative_table, dtype=np.int32)
        
    def fit(self, X, y=None):
        """拟合Word2Vec模型
        
        Args:
            X: array-like of shape (n_samples,)
            y: array-like of shape (n_samples,), optional
            
        Returns:
            self
        """
        # 预处理数据
        sentences = self._preprocess(X)
        
        # 构建Huffman树（如果使用层次化Softmax）
        if self.hs:
            huffman_tree = self._build_huffman_tree()
        
        # 创建负采样表（如果使用负采样）
        if self.negative > 0:
            negative_table = self._create_negative_table()
        
        # 训练Word2Vec模型
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=1,                # 使用Skip-gram
            hs=self.hs,         # 是否使用层次化Softmax
            negative=self.negative,  # 负采样数量
            ns_exponent=self.ns_exponent,  # 负采样指数
            compute_loss=True    # 计算训练损失
        )
        
        self.is_fitted = True
        return self
        
    def transform(self, X):
        """使用Word2Vec转换数据
        
        Args:
            X: array-like of shape (n_samples,)
            
        Returns:
            array-like of shape (n_samples, vector_size)
        """
        if not self.is_fitted:
            raise ValueError("Word2VecExtractor must be fitted before transform")
            
        sentences = self._preprocess(X)
        features = []
        
        for sentence in sentences:
            # 计算句子的加权平均向量
            vectors = []
            weights = []
            
            for word in sentence:
                if word in self.model.wv:
                    vectors.append(self.model.wv[word])
                    # 使用TF-IDF类似的权重
                    tf = self.vocab_freqs[word] / len(sentence)
                    idf = math.log(len(self.vocab_freqs) / (1 + self.vocab_freqs[word]))
                    weights.append(tf * idf)
            
            if vectors:
                # 加权平均
                weights = np.array(weights)
                weights = weights / weights.sum()  # 归一化权重
                sentence_vector = np.average(vectors, axis=0, weights=weights)
            else:
                sentence_vector = np.zeros(self.vector_size)
                
            features.append(sentence_vector)
            
        return np.array(features)
        
    def get_similar_words(self, word, topn=10):
        """获取与给定词最相似的词
        
        Args:
            word: str, 输入词
            topn: int, 返回的相似词数量
            
        Returns:
            list of tuple: (word, similarity)
        """
        if not self.is_fitted or word not in self.model.wv:
            return []
            
        return self.model.wv.most_similar(word, topn=topn)
        
    @property
    def feature_names(self):
        """获取特征名称
        
        Returns:
            list of str
        """
        return [f'word2vec_dim_{i}' for i in range(self.vector_size)]
        
    def get_vocabulary(self):
        """获取词汇表及词频
        
        Returns:
            dict: 词到频率的映射
        """
        return dict(self.vocab_freqs)
