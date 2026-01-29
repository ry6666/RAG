#!/usr/bin/env python3
"""BM25 Keyword Retrieval with pre-built index"""

import os
import sys
import re
import numpy as np
from typing import List, Set, Optional
from dataclasses import dataclass

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


@dataclass
class BM25RetrievalResult:
    doc_id: str
    question: str
    text: str
    score: float
    rank: int
    source: str = "bm25"


class BM25Retriever:
    """BM25 关键词检索（使用预建索引）"""
    
    _STOPWORDS: Set[str] = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'and', 'but', 'or', 'not', 'what', 'which', 'who',
        'this', 'that', 'it', 'its'
    }
    
    def __init__(self, kb_loader, top_k: int = 20):
        self.kb = kb_loader
        self._default_top_k = top_k
        self._token_cache = {}
        self._result_cache = {}
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        return f"{query[:50]}|||{top_k}"
    
    def _tokenize(self, text: str) -> List[str]:
        """分词（带缓存）"""
        if text in self._token_cache:
            return self._token_cache[text]
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        tokens = [w for w in words if w and len(w) > 1 and w not in self._STOPWORDS]
        
        self._token_cache[text] = tokens
        return tokens
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[BM25RetrievalResult]:
        """
        执行BM25检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            BM25RetrievalResult列表
        """
        if not query or not query.strip():
            print(f"[BM25Retriever] Warning: Empty query received")
            return []
        
        if self.kb.bm25 is None:
            print(f"[BM25Retriever] Warning: BM25 model not initialized")
            return []
        
        if top_k is None:
            top_k = self._default_top_k
        
        cache_key = self._get_cache_key(query, top_k)
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]
        
        try:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                words = query.lower().split()
                query_tokens = [w for w in words if w and len(w) > 1]
            
            if not query_tokens:
                print(f"[BM25Retriever] Warning: No valid tokens after tokenization")
                return []
            
            scores = self.kb.bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
        except Exception as e:
            print(f"[BM25Retriever] Error during retrieval: {e}")
            return []
        
        results = []
        seen = set()
        for rank, idx in enumerate(top_indices):
            if idx < 0 or idx in seen:
                continue
            if idx >= len(self.kb.docs):
                continue
            seen.add(idx)
            doc = self.kb.docs[idx]
            results.append(BM25RetrievalResult(
                doc_id=doc.get('id', f'doc_{idx}'),
                question=doc.get('question', ''),
                text=doc.get('text', ''),
                score=float(scores[idx]),
                rank=len(results) + 1,
                source="bm25"
            ))
        
        self._result_cache[cache_key] = results
        return results
    
    def clear_cache(self):
        """清除缓存"""
        self._token_cache.clear()
        self._result_cache.clear()
    
    def batch_retrieve(self, queries: List[str], top_k: Optional[int] = None) -> List[List[BM25RetrievalResult]]:
        """
        批量BM25检索（优化性能）
        
        Args:
            queries: 查询文本列表
            top_k: 返回结果数量
            
        Returns:
            每个查询的BM25RetrievalResult列表
        """
        if top_k is None:
            top_k = self._default_top_k
        
        batch_results = []
        for query in queries:
            results = self.retrieve(query, top_k)
            batch_results.append(results)
        return batch_results
    
    def clear_cache(self):
        """清除token缓存"""
        self._token_cache.clear()
