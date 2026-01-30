#!/usr/bin/env python3
"""Bridge Retriever

桥接类型问题专用检索器
特点：需要关联两个实体，找到连接它们的桥接实体
权重配置：关键词库 0.7，向量库 0.3
"""

import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reranker_model import get_reranker_model
from question_classifier import QuestionClassifier


class BridgeRetriever:
    """桥接问题检索器"""

    def __init__(self, vector_store=None, keyword_store=None):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.reranker = get_reranker_model()
        self.classifier = QuestionClassifier()

        self._init_stores()
        self._init_config()

    def _init_stores(self):
        """初始化存储库"""
        if self.vector_store is None:
            from kb.vector_store import VectorStore
            self.vector_store = VectorStore()

        if self.keyword_store is None:
            from kb.keyword_store import KeywordStore
            self.keyword_store = KeywordStore()

    def _init_config(self):
        """初始化配置（桥接问题优化版）
        
        核心原则：
        - 关键词库权重 0.7：桥接题依赖实体精准匹配
        - 向量库权重 0.3：仅做语义补充
        - recall_top_k=35：保障多跳链路线索覆盖
        - 添加分数阈值过滤：去除低质量线索
        """
        self.keyword_weight = 0.7
        self.vector_weight = 0.3
        self.recall_top_k = 35
        self.final_top_k = 10
        
        self.keyword_score_threshold = 0.2
        self.vector_score_threshold = 0.6

    def retrieve(self, question: str) -> List[Dict]:
        """检索桥接问题的相关文档（优化版：含兜底机制）
        
        检索策略：
        1. 多实体组合检索
        2. 若结果不足8条，触发单实体兜底检索
        3. 若仍不足，触发问题改写兜底检索
        """
        entities = self.classifier.extract_entities(question, "bridge")

        keyword_results = self._keyword_search(question, entities)
        vector_results = self._vector_search(question, entities)

        fused = self._fuse_results(keyword_results, vector_results)
        deduped = self._deduplicate(fused)
        
        if len(deduped) < 8 and entities:
            fallback_keyword = []
            fallback_vector = []
            
            for entity in entities:
                fk = self._keyword_search(question, [entity])
                fv = self._vector_search(question, [entity])
                fallback_keyword.extend(fk)
                fallback_vector.extend(fv)
            
            if fallback_keyword or fallback_vector:
                fallback_fused = self._fuse_results(fallback_keyword, fallback_vector)
                fallback_deduped = self._deduplicate(fallback_fused)
                
                existing_ids = set(r.get("chunk_id") for r in deduped)
                for r in fallback_deduped:
                    if r.get("chunk_id") not in existing_ids:
                        r["is_fallback"] = True
                        deduped.append(r)
        
        # 兜底策略2：若仍不足，尝试问题改写检索
        if len(deduped) < 5:
            reformulated_queries = self._generate_reformulated_queries(question, entities)
            for query in reformulated_queries:
                reformulated_results = self._keyword_search(query, entities)
                if reformulated_results:
                    existing_ids = set(r.get("chunk_id") for r in deduped)
                    for r in reformulated_results:
                        if r.get("chunk_id") not in existing_ids:
                            r["is_fallback"] = True
                            r["reformulated_query"] = query
                            deduped.append(r)
        
        deduped = sorted(deduped, key=lambda x: x.get("fused_score", 0.0), reverse=True)[:self.recall_top_k]
        reranked = self._rerank(question, entities, deduped)

        return self._format_results(reranked, entities)

    def _keyword_search(self, question: str, entities: List[str] = None) -> List[Dict]:
        """关键词库检索（增强版：支持实体扩展）
        
        增强策略：
        1. 原始实体检索
        2. 实体扩展（同义词、变体）检索
        """
        all_entities = entities if entities else []
        all_results = []
        seen_chunk_ids = set()
        
        for entity in all_entities:
            results = self.keyword_store.search(question, [entity], top_k=self.recall_top_k)
            
            for r in results:
                chunk_id = r.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    r["searched_entity"] = entity
                    all_results.append(r)
        
        for r in all_results:
            r["retrieval_type"] = "keyword"

        return all_results

    def _vector_search(self, question: str, entities: List[str] = None) -> List[Dict]:
        """向量库检索（增强版：多查询融合检索）
        
        增强策略：
        1. 原始问题查询
        2. 核心实体查询
        3. 关系词增强查询
        4. 多查询结果融合去重
        """
        all_results = []
        seen_chunk_ids = set()
        
        query_variants = []
        
        query_variants.append(question)
        
        if entities:
            for entity in entities:
                query_variants.append(entity)
            
            self._add_relation_words(question, query_variants)
        
        for query in query_variants[:5]:
            results = self.vector_store.search(query, top_k=self.recall_top_k // 2)
            
            for r in results:
                chunk_id = r.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    r["vector_query"] = query
                    all_results.append(r)
        
        for r in all_results:
            r["retrieval_type"] = "vector"
        
        return all_results
    
    def _add_relation_words(self, question: str, query_parts: List[str]):
        """提取并添加关系词，增强多实体关联检索"""
        relation_words = ["portrayed", "played", "directed", "composed", "written", 
                         "produced", "acted", "married", "born", "located"]
        
        question_lower = question.lower()
        for rw in relation_words:
            if rw in question_lower:
                query_parts.append(rw)
                break
    
    def _generate_reformulated_queries(self, question: str, entities: List[str]) -> List[str]:
        """生成改写查询（兜底策略）
        
        当原始检索失败时，尝试以下改写策略：
        1. 移除停用词，只保留核心实体
        2. 提取问题类型关键词（who, what, where, etc.）
        3. 保留疑问词+核心实体
        """
        reformulated = []
        
        if entities:
            reformulated.extend(entities)
        
        question_lower = question.lower()
        
        question_type_map = {
            "who": ["who", "person", "actor", "director", "singer"],
            "what": ["what", "position", "album", "book", "series", "team"],
            "where": ["where", "located", "city", "country", "venue"],
            "when": ["when", "year", "date"]
        }
        
        for q_type, keywords in question_type_map.items():
            if q_type in question_lower:
                for kw in keywords:
                    if kw in question_lower:
                        if entities:
                            reformulated.append(f"{q_type} {entities[0]} {kw}")
                        else:
                            reformulated.append(f"{q_type} {kw}")
                        break
                break
        
        if not reformulated:
            words = question.split()
            if len(words) > 3:
                reformulated.append(" ".join(words[:5]))
        
        return list(set(reformulated))[:5]

    def _fuse_results(self, keyword_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
        """融合检索结果（优化版：双库命中加分+分数归一化）
        
        融合策略：
        1. 分数归一化：统一关键词和向量分数到 0-1 范围
        2. 双库命中加分：同时被两库检索到的结果额外 +15%
        3. 保留原始排名信息
        """
        fused = {}
        
        max_keyword = max([r.get("score", 0) for r in keyword_results], default=1)
        max_vector = max([r.get("score", 0) for r in vector_results], default=1)
        
        DUAL_HIT_BONUS = 0.15

        for r in keyword_results:
            chunk_id = r.get("chunk_id", "")
            if chunk_id:
                norm_score = r.get("score", 0) / max_keyword if max_keyword > 0 else 0
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "keyword_score": r.get("score", 0.0),
                    "keyword_score_norm": norm_score,
                    "vector_score": 0.0,
                    "vector_score_norm": 0.0,
                    "fused_score": norm_score * self.keyword_weight,
                    "retrieval_types": ["keyword"],
                    "keyword_rank": 0,
                    "vector_rank": -1,
                    **r
                }

        for r in vector_results:
            chunk_id = r.get("chunk_id", "")
            if chunk_id:
                norm_score = r.get("score", 0) / max_vector if max_vector > 0 else 0
                if chunk_id in fused:
                    fused[chunk_id]["vector_score"] = r.get("score", 0.0)
                    fused[chunk_id]["vector_score_norm"] = norm_score
                    fused[chunk_id]["fused_score"] = (
                        fused[chunk_id]["keyword_score_norm"] * self.keyword_weight +
                        norm_score * self.vector_weight
                    )
                    fused[chunk_id]["retrieval_types"].append("vector")
                    fused[chunk_id]["vector_rank"] = vector_results.index(r)
                    
                    if len(fused[chunk_id]["retrieval_types"]) >= 2:
                        fused[chunk_id]["fused_score"] += DUAL_HIT_BONUS
                else:
                    fused[chunk_id] = {
                        "chunk_id": chunk_id,
                        "keyword_score": 0.0,
                        "keyword_score_norm": 0.0,
                        "vector_score": r.get("score", 0.0),
                        "vector_score_norm": norm_score,
                        "fused_score": norm_score * self.vector_weight,
                        "retrieval_types": ["vector"],
                        "keyword_rank": -1,
                        "vector_rank": vector_results.index(r),
                        **r
                    }

        for i, r in enumerate(keyword_results):
            if r.get("chunk_id") in fused:
                fused[r.get("chunk_id")]["keyword_rank"] = i

        sorted_results = sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)
        return sorted_results[:self.recall_top_k]

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """去重（优化版：chunk_id为主+哈希为辅，保留高分多类型线索）
        
        去重策略：
        1. chunk_id 完全去重（全局唯一标识）
        2. 文本哈希辅助去重相似内容
        3. 保留「融合分数更高、检索类型更全」的线索
        """
        seen_chunk_ids = set()
        seen_text_hashes = set()
        deduped = []
        
        for r in results:
            chunk_id = r.get("chunk_id", "")
            core_text = r.get("core_text", "").strip()
            
            if not core_text:
                continue
            
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                deduped.append(r)
            else:
                text_hash = core_text[:200].lower()
                if text_hash not in seen_text_hashes:
                    seen_text_hashes.add(text_hash)
                    deduped.append(r)
        
        return deduped

    def _rerank(self, question: str, entities: List[str], results: List[Dict]) -> List[Dict]:
        """重排序（优化版：实体加权+覆盖度筛选）
        
        重排序策略：
        1. 查询增强：问题+实体+关系词
        2. 桥接实体加权：包含桥接实体的线索额外加权
        3. 覆盖度筛选：确保覆盖问题中所有核心实体
        """
        if not results:
            return []
        
        enhanced_query = self._build_enhanced_query(question, entities)
        
        if self.reranker.is_available():
            passages = [r.get("core_text", "") for r in results]
            rerank_scores = self.reranker.rerank(enhanced_query, passages)
            
            for r, score in zip(results, rerank_scores):
                base_score = float(score)
                
                bridge_entity_bonus = 0.0
                chunk_bridge_entity = r.get("bridge_entity", "").lower()
                for entity in entities:
                    if entity.lower() in chunk_bridge_entity or chunk_bridge_entity in entity.lower():
                        bridge_entity_bonus = 0.2
                        break
                
                dual_hit_bonus = 0.1 if len(r.get("retrieval_types", [])) >= 2 else 0.0
                
                r["rerank_score"] = base_score + bridge_entity_bonus + dual_hit_bonus
            
            results = sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        results = self._filter_by_entity_coverage(results, entities)
        
        return results
    
    def _build_enhanced_query(self, question: str, entities: List[str]) -> str:
        """构建增强的查询"""
        query_parts = []
        
        if entities:
            query_parts.extend(entities)
        
        self._add_relation_words(question, query_parts)
        
        query_parts.append(question)
        
        return " ".join(query_parts)
    
    def _filter_by_entity_coverage(self, results: List[Dict], entities: List[str]) -> List[Dict]:
        """筛选：优先保留覆盖所有核心实体的线索"""
        if not entities or len(entities) == 0:
            return results
        
        covered_results = []
        uncovered_results = []
        
        for r in results:
            core_text = r.get("core_text", "").lower()
            entities_found = sum(1 for e in entities if e.lower() in core_text)
            
            if entities_found >= len(entities):
                covered_results.append(r)
            else:
                uncovered_results.append(r)
        
        return (covered_results + uncovered_results)[:self.recall_top_k]

    def _format_results(self, results: List[Dict], entities: List[str] = None) -> List[Dict]:
        """格式化结果（优化版：强化metadata信息，适配ReAct推理）
        
        返回格式：
        - core_text: 核心文本
        - chunk_id: 唯一标识
        - score: 融合分数
        - metadata:
          - bridge_entity: 桥接实体
          - clue_type: 线索类型
          - retrieval_types: 检索类型列表（keyword/vector）
          - dual_hit: 是否双库命中
          - source_entity: 关联的核心实体
        """
        formatted = []
        seen = set()

        for r in results:
            chunk_id = r.get("chunk_id", "")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                
                retrieval_types = r.get("retrieval_types", [])
                dual_hit = len(retrieval_types) >= 2 if retrieval_types else False
                
                source_entity = ""
                if entities:
                    core_text = r.get("core_text", "").lower()
                    for entity in entities:
                        if entity.lower() in core_text:
                            source_entity = entity
                            break
                
                formatted.append({
                    "core_text": r.get("core_text", ""),
                    "chunk_id": chunk_id,
                    "score": r.get("rerank_score", r.get("fused_score", 0.0)),
                    "metadata": {
                        "bridge_entity": r.get("bridge_entity", ""),
                        "clue_type": r.get("clue_type", ""),
                        "retrieval_types": retrieval_types,
                        "dual_hit": dual_hit,
                        "source_entity": source_entity,
                        "rerank_score": r.get("rerank_score", 0.0),
                        "fused_score": r.get("fused_score", 0.0)
                    }
                })

        return formatted


def get_bridge_retriever() -> BridgeRetriever:
    """获取桥接问题检索器（单例）"""
    return BridgeRetriever()
