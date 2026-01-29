#!/usr/bin/env python3
"""Vector Retrieval with BGE embeddings + FAISS"""

import os
import sys
import faiss
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.models.embedding_en import EmbeddingClient


@dataclass
class VectorRetrievalResult:
    doc_id: str
    question: str
    text: str
    score: float
    rank: int
    source: str = "vector"


class VectorRetriever:
    """向量检索（BGE embeddings + FAISS）"""
    
    def __init__(self, kb_loader, top_k: int = 20):
        self.kb = kb_loader
        self.embedding = EmbeddingClient()
        self._default_top_k = top_k
        self._result_cache = {}
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        return f"{query[:50]}|||{top_k}"
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[VectorRetrievalResult]:
        """
        执行向量检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            VectorRetrievalResult列表
        """
        if top_k is None:
            top_k = self._default_top_k
        
        cache_key = self._get_cache_key(query, top_k)
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]
        
        if not query or not query.strip():
            print(f"[VectorRetriever] Warning: Empty query received")
            return []
        
        try:
            query_emb = self.embedding.encode([query], normalize=True)
            if query_emb is None or len(query_emb) == 0:
                print(f"[VectorRetriever] Warning: Failed to encode query")
                return []
            search_k = min(top_k, self.kb.vector_index.ntotal)
            scores, indices = self.kb.vector_index.search(query_emb, search_k)
        except Exception as e:
            print(f"[VectorRetriever] Error during retrieval: {e}")
            return []
        
        results = []
        seen = set()
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx in seen:
                continue
            if idx >= len(self.kb.docs):
                continue
            seen.add(idx)
            doc = self.kb.docs[idx]
            results.append(VectorRetrievalResult(
                doc_id=doc.get('id', f'doc_{idx}'),
                question=doc.get('question', ''),
                text=doc.get('text', ''),
                score=float(score),
                rank=len(results) + 1,
                source="vector"
            ))
        
        self._result_cache[cache_key] = results
        return results
    
    def clear_cache(self):
        """清除缓存"""
        self._result_cache.clear()
    
    def batch_retrieve(self, queries: List[str], top_k: Optional[int] = None) -> List[List[VectorRetrievalResult]]:
        """
        批量向量检索（优化性能）
        
        Args:
            queries: 查询文本列表
            top_k: 返回结果数量
            
        Returns:
            每个查询的VectorRetrievalResult列表
        """
        if top_k is None:
            top_k = self._default_top_k
        
        try:
            query_embs = self.embedding.encode(queries, normalize=True)
            all_scores, all_indices = self.kb.vector_index.search(query_embs, min(top_k, self.kb.vector_index.ntotal))
        except Exception as e:
            print(f"[VectorRetriever] Error during batch retrieval: {e}")
            return [[] for _ in queries]
        
        batch_results = []
        for query_idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
            query = queries[query_idx]
            results = []
            seen = set()
            for rank, (idx, score) in enumerate(zip(indices, scores)):
                if idx < 0 or idx in seen:
                    continue
                seen.add(idx)
                doc = self.kb.docs[idx]
                results.append(VectorRetrievalResult(
                    doc_id=doc['id'],
                    question=doc.get('question', ''),
                    text=doc.get('text', ''),
                    score=float(score),
                    rank=len(results) + 1,
                    source="vector"
                ))
            batch_results.append(results)
        return batch_results
