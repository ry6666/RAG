#!/usr/bin/env python3
"""Graph-based Retrieval using Knowledge Graph"""

import os
import sys
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import src.retrieval.entity_extractor as ee


@dataclass
class GraphRetrievalResult:
    doc_id: str
    question: str
    text: str
    score: float
    rank: int
    source: str = "graph"
    bridge_entities: List[Dict] = field(default_factory=list)


class GraphRetriever:
    """基于知识图谱的检索"""
    
    def __init__(self, kb_loader, entity_extractor=None):
        self.kb = kb_loader
        self.entity_extractor = entity_extractor or ee.EntityExtractor()
    
    def get_entity_neighbors(self, entity: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """获取实体的邻居节点"""
        if not entity or not entity.strip():
            return []
        
        if self.kb.kg is None:
            print(f"[GraphRetriever] Warning: Knowledge graph not initialized")
            return []
        
        normalized = entity.lower()
        node = self.kb.entity_map.get(normalized, entity)
        
        if node not in self.kb.kg:
            return []
        
        neighbors = []
        for nbr in self.kb.kg.neighbors(node):
            weight = self.kb.kg[node][nbr].get('weight', 1)
            neighbors.append((nbr, weight))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]
    
    def find_bridge_entities(self, question_entities: List[str]) -> List[Dict]:
        """查找桥接实体"""
        candidates = []
        
        for entity in question_entities:
            neighbors = self.get_entity_neighbors(entity, top_k=10)
            for neighbor, weight in neighbors:
                if neighbor.lower() not in [e.lower() for e in question_entities]:
                    candidates.append({
                        'entity': neighbor,
                        'source': entity,
                        'weight': weight,
                        'type': self.kb.kg.nodes[neighbor].get('type', 'UNKNOWN') if neighbor in self.kb.kg else 'UNKNOWN'
                    })
        
        candidates.sort(key=lambda x: x['weight'], reverse=True)
        return candidates[:20]
    
    def get_entity_path(self, source: str, target: str, max_hops: int = 3) -> List[str]:
        """获取两个实体之间的路径"""
        if source not in self.kb.kg or target not in self.kb.kg:
            return []
        
        try:
            path = nx.shortest_path(self.kb.kg, source, target)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def get_hop_entities(self, source_entities: List[str], target_entities: List[str], max_hops: int = 2) -> Dict:
        """获取跳数内的实体"""
        hop_entities = {}
        
        for src in source_entities:
            hop_entities[src] = {'neighbors': [], 'paths': {}}
            
            neighbors = self.get_entity_neighbors(src, top_k=10)
            hop_entities[src]['neighbors'] = [{'entity': n, 'weight': w} for n, w in neighbors]
            
            for tgt in target_entities:
                path = self.get_entity_path(src, tgt, max_hops)
                if path:
                    hop_entities[src]['paths'][tgt] = path
        
        return hop_entities
    
    def retrieve_by_entities(self, entities: List[str], top_k: int = 10) -> List[GraphRetrievalResult]:
        """
        基于实体检索相关文档
        
        Args:
            entities: 查询实体列表
            top_k: 返回结果数量
            
        Returns:
            GraphRetrievalResult列表
        """
        entity_scores = {}
        
        for entity in entities:
            neighbors = self.get_entity_neighbors(entity, top_k=top_k)
            for neighbor, weight in neighbors:
                if neighbor not in entity_scores:
                    entity_scores[neighbor] = {'weight': weight, 'entities': [entity]}
                else:
                    entity_scores[neighbor]['weight'] += weight
                    entity_scores[neighbor]['entities'].append(entity)
        
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1]['weight'], reverse=True)
        
        results = []
        seen_docs = set()
        for rank, (entity, info) in enumerate(sorted_entities):
            doc = self.kb.get_doc_by_id(entity)
            if doc is None or doc['id'] in seen_docs:
                continue
            seen_docs.add(doc['id'])
            
            results.append(GraphRetrievalResult(
                doc_id=doc['id'],
                question=doc.get('question', ''),
                text=doc.get('text', ''),
                score=float(info['weight']),
                rank=len(results) + 1,
                source="graph",
                bridge_entities=[{'entity': entity, 'source_entities': info['entities']}]
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def multi_hop_retrieve(self, question: str, q_type: str = "bridge", top_k: int = 10) -> List[GraphRetrievalResult]:
        """
        多跳图检索
        
        Args:
            question: 查询问题
            q_type: 问题类型（bridge/comparison）
            top_k: 返回结果数量
            
        Returns:
            GraphRetrievalResult列表
        """
        if q_type == "comparison":
            entities = self.entity_extractor.extract_comparison_entities(question)
        else:
            entities = self.entity_extractor.extract_entities(question)
        
        if not entities:
            return []
        
        if q_type == "comparison" and len(entities) >= 2:
            all_candidates = {}
            seen_docs = set()
            
            for entity in entities[:2]:
                results = self.retrieve_by_entities([entity], top_k=top_k)
                for result in results:
                    if result.doc_id not in seen_docs:
                        seen_docs.add(result.doc_id)
                        all_candidates[result.doc_id] = result
            
            final_results = list(all_candidates.values())
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:top_k]
        else:
            bridge_entities = self.find_bridge_entities(entities)
            all_entities = entities + [b['entity'] for b in bridge_entities[:10]]
            return self.retrieve_by_entities(all_entities, top_k)
