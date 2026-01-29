#!/usr/bin/env python3
"""Knowledge Base Manager - Singleton Pattern for KB Loading

功能：
- 单例模式封装知识库加载
- 避免重复加载，提高评估效率
- 提供全局访问点

Usage:
    from src.kb_manager import KBManager
    
    # 获取知识库实例（首次加载，后续复用）
    kb_loader = KBManager.get_instance()
    
    # 手动重新加载
    KBManager.reload()
    
    # 获取统计信息
    stats = KBManager.get_stats()
"""

import os
import sys
import json
import pickle
import time
from typing import Optional, Dict, Any

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


class KBManager:
    """知识库管理器 - 单例模式"""
    
    _instance = None
    _kb_loader = None
    _load_time = None
    _load_count = 0
    
    def __new__(cls, kb_dir: str = "kb"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, kb_dir: str = "kb"):
        if KBManager._kb_loader is None:
            self._kb_dir = kb_dir
            self._load_kb()
    
    def _load_kb(self):
        """加载知识库"""
        start_time = time.time()
        
        print(f"[KBManager] Loading KB from {self._kb_dir}...")
        
        KBManager._kb_loader = _KBLoaderImpl(self._kb_dir)
        
        KBManager._load_time = time.time() - start_time
        KBManager._load_count += 1
        
        print(f"[KBManager] KB loaded in {KBManager._load_time:.2f}s (load count: {KBManager._load_count})")
    
    @classmethod
    def get_instance(cls, kb_dir: str = "kb") -> '_KBLoaderImpl':
        """获取知识库实例（单例模式）"""
        if cls._instance is None:
            cls._instance = cls(kb_dir)
        return cls._instance._kb_loader
    
    @classmethod
    def reload(cls, kb_dir: str = "kb"):
        """重新加载知识库"""
        cls._kb_loader = None
        cls._instance = None
        return cls.get_instance(kb_dir)
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """获取加载统计信息"""
        if cls._instance is None or cls._instance._kb_loader is None:
            return {
                'loaded': False,
                'load_count': 0,
                'load_time': 0,
                'doc_count': 0,
                'kg_nodes': 0,
                'kg_edges': 0
            }
        
        kb_loader = cls._instance._kb_loader
        return {
            'loaded': True,
            'load_count': cls._load_count,
            'load_time': cls._load_time,
            'doc_count': len(kb_loader.docs),
            'kg_nodes': kb_loader.kg.number_of_nodes() if kb_loader.kg else 0,
            'kg_edges': kb_loader.kg.number_of_edges() if kb_loader.kg else 0,
            'kb_dir': kb_loader.kb_dir if hasattr(kb_loader, 'kb_dir') else 'unknown'
        }
    
    @classmethod
    def get_loader(cls) -> Optional['_KBLoaderImpl']:
        """获取原始加载器对象"""
        return cls._instance._kb_loader if cls._instance else None
    
    @classmethod
    def clear(cls):
        """清除单例（用于测试）"""
        cls._instance = None
        cls._load_time = None


class _KBLoaderImpl:
    """知识库加载实现（内部类）"""
    
    def __init__(self, kb_dir: str = "kb"):
        self.kb_dir = kb_dir
        self.docs = []
        self.doc_map = {}
        self.kg = None
        self.entity_map = {}
        self.bm25 = None
        self.bm25_doc_ids = []
        self.vector_index = None
        self.metadata = {}
        self._load_all()
    
    def _load_all(self):
        """加载所有知识库组件"""
        print(f"  [KB] Loading docs...")
        with open(f"{self.kb_dir}/docs.json", 'r') as f:
            self.docs = json.load(f)
        
        for doc in self.docs:
            self.doc_map[doc['id']] = doc
        print(f"  [KB] Loaded {len(self.docs)} docs")
        
        print(f"  [KB] Loading BM25...")
        with open(f"{self.kb_dir}/bm25.pkl", 'rb') as f:
            bm25_data = pickle.load(f)
            if isinstance(bm25_data, dict):
                self.bm25 = bm25_data.get('model')
                self.bm25_doc_ids = bm25_data.get('doc_ids', [])
            else:
                self.bm25 = bm25_data
                self.bm25_doc_ids = list(range(len(self.docs)))
        
        print(f"  [KB] Loading vector index...")
        import faiss
        self.vector_index = faiss.read_index(f"{self.kb_dir}/vector.index")
        
        print(f"  [KB] Loading metadata...")
        with open(f"{self.kb_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        print(f"  [KB] Loading Knowledge Graph...")
        graph_pkl = f"{self.kb_dir}/graph.pkl"
        graph_graphml = f"{self.kb_dir}/graph.graphml"
        
        import networkx as nx
        
        if os.path.exists(graph_pkl):
            with open(graph_pkl, 'rb') as f:
                data = pickle.load(f)
            self.kg = data['graph']
        elif os.path.exists(graph_graphml):
            self.kg = nx.read_graphml(graph_graphml)
        else:
            raise FileNotFoundError("No graph file found (graph.pkl or graph.graphml)")
        
        for node in self.kg.nodes():
            self.entity_map[node.lower()] = node
        
        print(f"  [KB] Done: {len(self.docs)} docs, dim={self.vector_index.d}")
        print(f"  [KB] KG: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
    
    def get_doc_by_id(self, doc_id: str) -> Optional[Dict]:
        """根据ID获取文档"""
        if not doc_id or not doc_id.strip():
            return None
        return self.doc_map.get(doc_id)
    
    def get_entity_neighbors(self, entity: str, top_k: int = 10):
        """获取实体的邻居节点"""
        normalized = entity.lower()
        node = self.entity_map.get(normalized, entity)
        
        if node not in self.kg:
            return []
        
        neighbors = []
        for nbr in self.kg.neighbors(node):
            weight = self.kg[node][nbr].get('weight', 1)
            neighbors.append((nbr, weight))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]
    
    @property
    def docs_count(self) -> int:
        return len(self.docs)
    
    @property
    def kg_stats(self) -> Dict[str, int]:
        if self.kg:
            return {
                'nodes': self.kg.number_of_nodes(),
                'edges': self.kg.number_of_edges()
            }
        return {'nodes': 0, 'edges': 0}


def get_kb(kb_dir: str = "kb") -> _KBLoaderImpl:
    """便捷函数：获取知识库实例"""
    return KBManager.get_instance(kb_dir)


def reload_kb(kb_dir: str = "kb") -> _KBLoaderImpl:
    """便捷函数：重新加载知识库"""
    return KBManager.reload(kb_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("KB Manager Test")
    print("=" * 60)
    
    print("\n[1] First load:")
    kb1 = get_kb()
    print(f"    Docs: {kb1.docs_count}")
    
    print("\n[2] Second get (should be instant):")
    start = time.time()
    kb2 = get_kb()
    elapsed = time.time() - start
    print(f"    Got in {elapsed*1000:.1f}ms (same instance: {kb1 is kb2})")
    
    print("\n[3] Stats:")
    stats = KBManager.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    print("\n[4] Reload:")
    kb3 = reload_kb()
    print(f"    New instance: {kb1 is not kb3}")
