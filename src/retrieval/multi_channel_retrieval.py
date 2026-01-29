#!/usr/bin/env python3
"""Multi-Channel Retrieval with Multi-Hop Reasoning for HotpotQA

主模块：整合三路检索 + 重排序 + 生成答案
评估功能由 eval_kb.py 完成
"""

import os
import sys
import json
import pickle
import re
import time
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import numpy as np

from src.retrieval.vector_retriever import VectorRetriever, VectorRetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever, BM25RetrievalResult
from src.retrieval.graph_retriever import GraphRetriever, GraphRetrievalResult
from src.retrieval.entity_extractor import EntityExtractor
from src.models.reranker_en import RerankerClient
from src.models.ollama import OllamaClient


DEBUG = False

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


@dataclass
class FusionResult:
    doc_id: str
    question: str
    text: str
    score: float
    rank: int
    source: str
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None


class KBLoader:
    """加载知识库"""
    
    def __init__(self, kb_dir: str = "kb"):
        self.kb_dir = kb_dir
        self.docs = []
        self.doc_map = {}
        self.kg = None
        self.entity_map = {}
        self.load_all()
    
    def load_all(self):
        print(f"[Load] Loading KB from {self.kb_dir}")
        
        with open(f"{self.kb_dir}/docs.json", 'r') as f:
            self.docs = json.load(f)
        
        for doc in self.docs:
            self.doc_map[doc['id']] = doc
        
        with open(f"{self.kb_dir}/bm25.pkl", 'rb') as f:
            bm25_data = pickle.load(f)
            if isinstance(bm25_data, dict):
                self.bm25 = bm25_data.get('model')
                self.bm25_doc_ids = bm25_data.get('doc_ids', [])
            else:
                self.bm25 = bm25_data
                self.bm25_doc_ids = list(range(len(self.docs)))
        
        import faiss
        self.vector_index = faiss.read_index(f"{self.kb_dir}/vector.index")
        
        with open(f"{self.kb_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        print(f"[Load] Loading Knowledge Graph...")
        graph_pkl = f"{self.kb_dir}/graph.pkl"
        graph_graphml = f"{self.kb_dir}/graph.graphml"
        
        import networkx as nx
        
        if os.path.exists(graph_pkl):
            print(f"[Load] Loading from graph.pkl (fast)...")
            with open(graph_pkl, 'rb') as f:
                data = pickle.load(f)
            self.kg = data['graph']
        elif os.path.exists(graph_graphml):
            print(f"[Load] Loading from graph.graphml...")
            self.kg = nx.read_graphml(graph_graphml)
        else:
            raise FileNotFoundError("No graph file found (graph.pkl or graph.graphml)")
        
        for node in self.kg.nodes():
            self.entity_map[node.lower()] = node
        
        print(f"[Load] Loaded {len(self.docs)} docs, dim={self.vector_index.d}")
        print(f"[Load] KG: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
    
    def get_doc_by_id(self, doc_id: str) -> Optional[Dict]:
        if not doc_id or not doc_id.strip():
            return None
        return self.doc_map.get(doc_id)
    
    def get_entity_neighbors(self, entity: str, top_k: int = 10) -> List[Tuple[str, float]]:
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


class MultiChannelRetriever:
    """多路检索融合（整合三路检索结果）"""
    
    DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)
    
    def __init__(self, kb_loader: KBLoader, 
                 vector_weight: float = 0.4,
                 bm25_weight: float = 0.3,
                 graph_weight: float = 0.3):
        self.kb = kb_loader
        
        self.vector_retriever = VectorRetriever(kb_loader)
        self.bm25_retriever = BM25Retriever(kb_loader)
        self.graph_retriever = GraphRetriever(kb_loader)
        self.entity_extractor = EntityExtractor(use_glm=True)
        
        self.weights = (vector_weight, bm25_weight, graph_weight)
        self._normalize_scores = True
    
    def retrieve(self, query: str, top_k: int = 20, 
                 weights: Optional[Tuple[float, float, float]] = None,
                 use_normalize: bool = True) -> List[FusionResult]:
        """
        执行三路检索融合
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            weights: 三个检索通道的权重 (vector, bm25, graph)
            use_normalize: 是否对分数进行归一化
            
        Returns:
            FusionResult列表
        """
        if not query or not query.strip():
            print(f"[MultiChannelRetriever] Warning: Empty query received")
            return []
        
        w = weights or self.weights
        
        dprint(f"[Fusion] Query: {query[:50]}...")
        
        vector_results = self.vector_retriever.retrieve(query, top_k * 2)
        dprint(f"[Fusion] Vector: {len(vector_results)} results")
        
        bm25_results = self.bm25_retriever.retrieve(query, top_k * 2)
        dprint(f"[Fusion] BM25: {len(bm25_results)} results")
        
        graph_results = self.graph_retriever.retrieve_by_entities(
            self.entity_extractor.extract_entities(query), top_k=top_k * 2
        )
        dprint(f"[Fusion] Graph: {len(graph_results)} results")
        
        if not vector_results and not bm25_results and not graph_results:
            print(f"[MultiChannelRetriever] Warning: All retrievers returned empty results")
        
        score_map = self._merge_scores(
            vector_results, bm25_results, graph_results,
            w, use_normalize
        )
        
        fused_scores = sorted(score_map.items(), key=lambda x: x[1]['fused'], reverse=True)
        
        results = []
        for rank, (doc_id, info) in enumerate(fused_scores[:top_k]):
            doc = info['doc']
            results.append(FusionResult(
                doc_id=doc_id,
                question=doc.get('question', ''),
                text=doc.get('text', ''),
                score=info['fused'],
                rank=rank + 1,
                source="fusion",
                rerank_score=None,
                final_score=None
            ))
        
        return results
    
    def _merge_scores(self, 
                      vector: List[VectorRetrievalResult],
                      bm25: List[BM25RetrievalResult],
                      graph: List[GraphRetrievalResult],
                      weights: Tuple[float, float, float],
                      normalize: bool) -> Dict:
        """合并三路检索分数（融入桥接实体权重）"""
        score_map = {}
        
        for r in vector:
            score_map[r.doc_id] = {
                'vector': r.score, 'bm25': 0.0, 'graph': 0.0,
                'doc': {'id': r.doc_id, 'question': r.question, 'text': r.text},
                'bridge_weight': 0.0
            }
        
        for r in bm25:
            key = r.doc_id
            if key in score_map:
                score_map[key]['bm25'] = r.score
            else:
                score_map[key] = {
                    'vector': 0.0, 'bm25': r.score, 'graph': 0.0,
                    'doc': {'id': r.doc_id, 'question': r.question, 'text': r.text},
                    'bridge_weight': 0.0
                }
        
        for r in graph:
            key = r.doc_id
            bridge_weight = self._calculate_bridge_weight(r)
            if key in score_map:
                score_map[key]['graph'] = r.score
                score_map[key]['bridge_weight'] = max(score_map[key]['bridge_weight'], bridge_weight)
            else:
                score_map[key] = {
                    'vector': 0.0, 'bm25': 0.0, 'graph': r.score,
                    'doc': {'id': r.doc_id, 'question': r.question, 'text': r.text},
                    'bridge_weight': bridge_weight
                }
        
        if normalize:
            self._normalize_score_map(score_map)
        
        fused_results = {}
        bridge_weight_factor = 0.15
        
        for key, vals in score_map.items():
            v_score = vals['vector'] * weights[0]
            b_score = vals['bm25'] * weights[1]
            g_score = vals['graph'] * weights[2]
            bridge_boost = vals['bridge_weight'] * bridge_weight_factor
            
            fused_score = v_score + b_score + g_score + bridge_boost
            
            fused_results[key] = {
                **vals,
                'fused': fused_score,
                'bridge_boost': bridge_boost
            }
        
        return fused_results
    
    def _calculate_bridge_weight(self, graph_result: GraphRetrievalResult) -> float:
        """计算桥接实体权重"""
        if not graph_result.bridge_entities:
            return 0.0
        
        if len(graph_result.bridge_entities) == 0:
            return 0.0
        
        total_weight = 0.0
        for bridge in graph_result.bridge_entities:
            weight = bridge.get('weight', bridge.get('confidence', 1.0))
            total_weight += weight
        
        normalized_weight = total_weight / len(graph_result.bridge_entities)
        
        return min(normalized_weight * 0.1, 0.15)
    
    def _normalize_score_map(self, score_map: Dict):
        """Min-Max分数归一化"""
        all_scores = []
        for vals in score_map.values():
            all_scores.extend([vals['vector'], vals['bm25'], vals['graph']])
        
        if not all_scores:
            return
        
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        if max_score - min_score < 1e-6:
            for vals in score_map.values():
                vals['vector'] = 0.0
                vals['bm25'] = 0.0
                vals['graph'] = 0.0
            return
        
        for vals in score_map.values():
            vals['vector'] = (vals['vector'] - min_score) / (max_score - min_score)
            vals['bm25'] = (vals['bm25'] - min_score) / (max_score - min_score)
            vals['graph'] = (vals['graph'] - min_score) / (max_score - min_score)


class Reranker:
    """BGE 重排序"""
    
    def __init__(self, rerank_weight: float = 0.7, original_weight: float = 0.3):
        self.reranker = RerankerClient()
        self.rerank_weight = rerank_weight
        self.original_weight = original_weight
        self._text_cache = {}
    
    def _prepare_passage(self, candidate: Dict) -> str:
        """预处理候选文档"""
        text = candidate.get('text', '')
        relevant_text = candidate.get('relevant_text', '')
        
        if relevant_text and len(relevant_text) > 50:
            return relevant_text[:512]
        
        sentences = text.split('. ')
        truncated = ''
        for sent in sentences:
            if len(truncated) + len(sent) + 1 > 512:
                break
            truncated += ('. ' if truncated else '') + sent
        
        if not truncated:
            truncated = text[:512]
        
        cache_key = truncated[:100]
        self._text_cache[cache_key] = candidate
        
        return truncated[:512]
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """重排序"""
        if not candidates:
            return []
        
        text_to_original = {}
        processed_texts = []
        
        for c in candidates:
            text = self._prepare_passage(c)
            processed_texts.append(text)
            if text[:100] not in text_to_original:
                text_to_original[text[:100]] = c
        
        try:
            reranked = self.reranker.rerank(query, processed_texts, top_k)
        except Exception as e:
            print(f"[Reranker] Error: {e}")
            reranked = None
        
        results = []
        if reranked:
            for (text, score), orig_text in zip(reranked, processed_texts):
                orig = text_to_original.get(orig_text[:100], candidates[0])
                final_score = orig.get('score', 0) * self.original_weight + score * self.rerank_weight
                
                result = {
                    **orig,
                    'rerank_score': score,
                    'final_score': final_score
                }
                results.append(result)
        else:
            for c in candidates[:top_k]:
                result = {
                    **c,
                    'rerank_score': c.get('score', 0),
                    'final_score': c.get('score', 0)
                }
                results.append(result)
        
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return results[:top_k]
    
    def clear_cache(self):
        """清除缓存"""
        self._text_cache.clear()


class AnswerGenerator:
    """答案生成"""
    
    def __init__(self, model: str = "qwen2:7b-instruct", max_tokens: int = 128):
        self.model = model
        self.max_tokens = max_tokens
        self.client = None
        
        try:
            self.client = OllamaClient(model_name=model)
        except Exception as e:
            print(f"[AnswerGenerator] Warning: Ollama client init failed: {e}")
    
    def generate(self, question: str, contexts: List[str], 
                 max_context_len: int = 2000) -> str:
        """生成答案"""
        if self.client is None:
            return "[Generator unavailable]"
        
        context_text = "\n".join([c[:max_context_len] for c in contexts])
        
        prompt = f"""Answer the question based on the context. If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question: {question}

Answer:"""
        
        try:
            response = self.client.generate(prompt, max_tokens=self.max_tokens)
            return response.strip() if response else "[No response]"
        except Exception as e:
            print(f"[AnswerGenerator] Error: {e}")
            return "[Generation failed]"
    
    def generate_with_history(self, question: str, contexts: List[str],
                              history: List[Dict] = None, max_context_len: int = 2000) -> str:
        """带历史的答案生成"""
        if self.client is None:
            return "[Generator unavailable]"
        
        context_text = "\n".join([c[:max_context_len] for c in contexts])
        
        history_prompt = ""
        if history:
            for h in history[-3:]:
                history_prompt += f"Q: {h.get('question', '')}\nA: {h.get('answer', '')}\n\n"
        
        prompt = f"""{history_prompt}Based on the context and previous conversation, answer the current question.

Context:
{context_text}

Current Question: {question}

Answer:"""
        
        try:
            response = self.client.generate(prompt, max_tokens=self.max_tokens)
            return response.strip() if response else "[No response]"
        except Exception as e:
            print(f"[AnswerGenerator] Error: {e}")
            return "[Generation failed]"


class MultiHopReasoner:
    """多跳推理（核心编排器）"""
    
    def __init__(self, kb_loader: KBLoader, 
                 rerank_weight: float = 0.7,
                 fusion_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)):
        self.kb = kb_loader
        self.retriever = MultiChannelRetriever(kb_loader, 
                                               vector_weight=fusion_weights[0],
                                               bm25_weight=fusion_weights[1],
                                               graph_weight=fusion_weights[2])
        self.reranker = Reranker(rerank_weight=rerank_weight)
        self.entity_extractor = self.retriever.entity_extractor
        self.generator = AnswerGenerator()
    
    def search(self, question: str, q_type: str = "bridge", 
               level: str = "medium", top_k: int = 10) -> Dict:
        """搜索入口"""
        if q_type == "comparison":
            return self._search_comparison(question, top_k)
        else:
            return self._search_bridge(question, top_k)
    
    def _search_comparison(self, question: str, top_k: int) -> Dict:
        """比较型问题检索"""
        entities = self.entity_extractor.extract_comparison_entities_glm(question)
        
        if len(entities) < 2:
            entities = self.entity_extractor.extract_comparison_entities(question)
        
        if len(entities) < 2:
            entities = [e.strip() for e in re.split(r'\s+and\s+|\s+or\s+', question) if e.strip()]
            entities = entities[:2]
        
        dprint(f"[Comparison] GLM Entities: {entities}")
        
        all_candidates = {}
        seen_docs = set()
        
        for entity in entities:
            results = self.retriever.retrieve(entity, top_k * 3)
            for r in results:
                if r.doc_id not in seen_docs:
                    seen_docs.add(r.doc_id)
                    all_candidates[r.doc_id] = {
                        'doc_id': r.doc_id,
                        'text': r.text,
                        'score': r.score,
                        'entity': entity,
                        'source': f'entity:{entity[:20]}'
                    }
        
        candidates = list(all_candidates.values())
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        reranked = self.reranker.rerank(question, candidates, top_k)
        
        return {
            "question": question,
            "candidates": reranked,
            "entities": entities,
            "type": "comparison"
        }
    
    def _search_bridge(self, question: str, top_k: int) -> Dict:
        """桥接型问题检索"""
        first_entities = self.entity_extractor.extract_bridge_entities(question)
        dprint(f"[Bridge] First hop entities: {first_entities[:5]}")
        
        if not first_entities:
            all_entities = self.entity_extractor.extract_entities(question)
            first_entities = all_entities[:3]
        
        if not first_entities:
            glm_bridge = self.entity_extractor.extract_bridge_entities_glm(question)
            if glm_bridge:
                first_entities = glm_bridge[:3]
        
        if not first_entities:
            results = self.retriever.retrieve(question, top_k)
            reranked = self.reranker.rerank(question, [
                {'doc_id': r.doc_id, 'text': r.text, 'score': r.score}
                for r in results
            ], top_k)
            
            return {
                "question": question,
                "candidates": reranked,
                "entities": [],
                "type": "bridge"
            }
        
        graph_bridge = self.retriever.graph_retriever.find_bridge_entities(first_entities)
        second_queries = [b['entity'] for b in graph_bridge[:5]]
        dprint(f"[Bridge] Second-hop queries: {second_queries} (含KG: {len(graph_bridge)}个)")
        
        all_candidates = {}
        seen_docs = set()
        
        for entity in first_entities:
            results = self.retriever.retrieve(entity, top_k * 2)
            for r in results:
                if r.doc_id not in seen_docs:
                    seen_docs.add(r.doc_id)
                    all_candidates[r.doc_id] = {
                        'doc_id': r.doc_id,
                        'text': r.text,
                        'score': r.score,
                        'entity': entity,
                        'source': f'first_hop:{entity[:20]}'
                    }
        
        for entity in second_queries:
            results = self.retriever.retrieve(entity, top_k)
            for r in results:
                if r.doc_id not in seen_docs:
                    seen_docs.add(r.doc_id)
                    all_candidates[r.doc_id] = {
                        'doc_id': r.doc_id,
                        'text': r.text,
                        'score': r.score,
                        'entity': entity,
                        'source': f'bridge:{entity[:20]}'
                    }
        
        candidates = list(all_candidates.values())
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        reranked = self.reranker.rerank(question, candidates, top_k)
        
        return {
            "question": question,
            "candidates": reranked,
            "entities": first_entities,
            "bridge_entities": second_queries,
            "type": "bridge"
        }
    
    def answer(self, question: str, q_type: str = "bridge", 
               level: str = "medium", top_k: int = 5) -> Dict:
        """检索+生成答案"""
        search_result = self.search(question, q_type, level, top_k * 2)
        
        candidates = search_result.get('candidates', [])
        
        contexts = [c.get('text', '')[:2000] for c in candidates[:top_k]]
        
        answer = self.generator.generate(question, contexts)
        
        return {
            "question": question,
            "answer": answer,
            "candidates": candidates,
            "contexts": contexts,
            "type": q_type,
            "entities": search_result.get('entities', []),
            "bridge_entities": search_result.get('bridge_entities', [])
        }


def demo(kb_loader: KBLoader, sample_data: List[Dict], top_k: int = 5):
    """演示模式"""
    print("\n" + "="*80)
    print("DEMO MODE")
    print("="*80)
    
    reasoner = MultiHopReasoner(kb_loader)
    
    for idx, item in enumerate(sample_data[:5]):
        question = item.get('question', '')
        q_type = item.get('type', 'bridge')
        
        if not question:
            continue
        
        print(f"\n[Demo {idx+1}] Type: {q_type}")
        print(f"  Question: {question[:80]}...")
        
        start_time = time.time()
        result = reasoner.answer(question, q_type, top_k=top_k)
        elapsed = time.time() - start_time
        
        print(f"  Answer: {result.get('answer', '[No answer]')[:100]}...")
        print(f"  Entities: {result.get('entities', [])}")
        print(f"  Time: {elapsed*1000:.1f}ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Channel Retrieval Demo")
    parser.add_argument("--kb-dir", type=str, default="kb", help="Knowledge base directory")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--data", type=str, help="Test data file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    if args.debug:
        DEBUG = True
    
    kb_loader = KBLoader(args.kb_dir)
    
    test_data = []
    if args.data:
        import pandas as pd
        if args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
            test_data = df.to_dict('records')[:10]
        elif args.data.endswith('.json'):
            with open(args.data, 'r') as f:
                test_data = json.load(f)[:10]
    
    if args.demo:
        demo(kb_loader, test_data, args.top_k)
    else:
        print("Use --demo to run demo mode. Use --help for options.")
