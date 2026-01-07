#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线嵌入模块
提供基于TF-IDF的文本嵌入功能，作为HuggingFace模型的备选方案
"""

import os
import pickle
import logging
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class TFIDFEmbeddings:
    """
    基于TF-IDF的嵌入模型
    完全离线运行，不需要网络连接
    """
    
    def __init__(self, max_features=5000, cache_file="./vectorstore/tfidf_model.pkl"):
        """
        初始化TF-IDF嵌入模型
        
        Args:
            max_features: 最大特征数量
            cache_file: 缓存文件路径
        """
        self.max_features = max_features
        self.cache_file = cache_file
        self.vectorizer = None
        self.fitted = False
        
        # 尝试加载已训练的模型
        self._load_model()
        
        logger.info(f"TF-IDF嵌入模型初始化完成，max_features={max_features}")
    
    def _load_model(self):
        """
        从缓存文件加载已训练的模型
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                    self.fitted = True
                logger.info(f"成功加载缓存的TF-IDF模型: {self.cache_file}")
            except Exception as e:
                logger.warning(f"加载TF-IDF模型缓存失败: {str(e)}")
                self._init_vectorizer()
        else:
            self._init_vectorizer()
    
    def _init_vectorizer(self):
        """
        初始化TF-IDF向量化器
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),  # 使用1-2gram
            min_df=1,
            max_df=0.95
        )
        self.fitted = False
        logger.info("初始化新的TF-IDF向量化器")
    
    def _save_model(self):
        """
        保存训练好的模型到缓存文件
        """
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"TF-IDF模型已保存到: {self.cache_file}")
        except Exception as e:
            logger.error(f"保存TF-IDF模型失败: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行嵌入
        
        Args:
            texts: 文档文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not self.fitted:
            # 如果模型未训练，先训练
            logger.info("训练TF-IDF模型...")
            self.vectorizer.fit(texts)
            self.fitted = True
            self._save_model()
        
        # 转换文档为向量
        vectors = self.vectorizer.transform(texts)
        
        # 转换为密集矩阵并返回列表格式
        dense_vectors = vectors.toarray()
        return [vector.tolist() for vector in dense_vectors]
    
    def embed_query(self, text: str) -> List[float]:
        """
        对查询文本进行嵌入
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        if not self.fitted:
            logger.warning("TF-IDF模型未训练，无法处理查询")
            # 返回零向量
            return [0.0] * self.max_features
        
        # 转换查询为向量
        vector = self.vectorizer.transform([text])
        dense_vector = vector.toarray()[0]
        return dense_vector.tolist()
    
    def similarity_search(self, query: str, documents: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """
        基于相似度搜索最相关的文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            k: 返回的文档数量
            
        Returns:
            List[Dict]: 相似文档列表
        """
        if not documents:
            return []
        
        # 如果模型未训练，先训练
        if not self.fitted:
            self.embed_documents(documents)
        
        # 获取查询和文档的向量
        query_vector = self.vectorizer.transform([query])
        doc_vectors = self.vectorizer.transform(documents)
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        # 获取最相似的k个文档
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回有相似度的文档
                results.append({
                    'content': documents[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return results
    
    def get_dimension(self) -> int:
        """
        获取嵌入向量的维度
        
        Returns:
            int: 向量维度
        """
        if self.fitted and self.vectorizer:
            return len(self.vectorizer.get_feature_names_out())
        return self.max_features