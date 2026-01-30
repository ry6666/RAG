#!/usr/bin/env python3
"""Comparison Retriever

比较类型问题专用检索器
特点：需要比较两个实体的属性或特征
权重配置：关键词库 0.6，向量库 0.4
"""

import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reranker_model import get_reranker_model
from question_classifier import QuestionClassifier


class ComparisonRetriever:
    """比较问题检索器"""

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
        """初始化配置（比较问题优化版）
        
        核心原则：
        - 关键词库权重 0.6：实体精准匹配
        - 向量库权重 0.4：语义关联同维度属性
        - recall_top_k=30：保障双实体各维度线索覆盖
        - 添加分数阈值过滤
        """
        self.keyword_weight = 0.6
        self.vector_weight = 0.4
        self.recall_top_k = 30
        self.final_top_k = 8
        
        self.keyword_score_threshold = 0.25
        self.vector_score_threshold = 0.65

    def retrieve(self, question: str) -> List[Dict]:
        """检索比较问题的相关文档（优化版：维度化检索+双实体均衡）
        
        检索策略：
        1. 提取双实体 + 对比维度
        2. 对双实体分别执行「实体+维度」检索
        3. 合并结果，保障线索占比均衡
        """
        extracted = self._extract_entities_with_dimension(question)
        entities = extracted.get("entities", [])
        dimension = extracted.get("dimension", "")
        
        if len(entities) >= 2:
            entity_a, entity_b = entities[0], entities[1]
            
            keyword_a = self._keyword_search(question, [entity_a, dimension], f"{entity_a}")
            keyword_b = self._keyword_search(question, [entity_b, dimension], f"{entity_b}")
            
            vector_a = self._vector_search(question, [entity_a, dimension])
            vector_b = self._vector_search(question, [entity_b, dimension])
            
            all_keyword = keyword_a + keyword_b
            all_vector = vector_a + vector_b
        else:
            all_keyword = self._keyword_search(question, entities, "both")
            all_vector = self._vector_search(question, entities)

        fused = self._fuse_results(all_keyword, all_vector, dimension)
        deduped = self._deduplicate(fused)
        reranked = self._rerank(question, entities, dimension, deduped)

        return self._format_results(reranked, entities, dimension)
    
    def _extract_entities_with_dimension(self, question: str) -> Dict:
        """提取双实体和对比维度"""
        entities = self.classifier.extract_entities(question, "comparison")
        
        dimension = ""
        dimension_keywords = [
            "nationality", "born", "birth", "occupation", "profession", 
            "career", "genre", "year", "time", "position", "role",
            "country", "city", "education", "award", "style"
        ]
        
        question_lower = question.lower()
        for kw in dimension_keywords:
            if kw in question_lower:
                dimension = kw
                break
        
        if not dimension:
            dimension = "general"
        
        return {
            "entities": entities,
            "dimension": dimension
        }
    
    def _keyword_search(self, question: str, entities: List[str] = None, entity_tag: str = "both") -> List[Dict]:
        """关键词库检索（带实体标签）"""
        all_entities = entities if entities else []
        
        results = self.keyword_store.search(question, all_entities, top_k=self.recall_top_k // 2 if entity_tag != "both" else self.recall_top_k)
        
        for r in results:
            r["retrieval_type"] = "keyword"
            r["entity_tag"] = entity_tag
        
        return results
    
    def _vector_search(self, question: str, entities: List[str] = None) -> List[Dict]:
        """向量库检索（维度化查询）"""
        query_parts = []
        
        if entities and len(entities) > 0:
            query_parts.extend(entities)
        
        query_parts.append(question)
        query = " ".join(query_parts)
        
        results = self.vector_store.search(query, top_k=self.recall_top_k)
        
        for r in results:
            r["retrieval_type"] = "vector"
        
        return results

    def _fuse_results(self, keyword_results: List[Dict], vector_results: List[Dict], dimension: str = "") -> List[Dict]:
        """融合检索结果（优化版：双库命中加分+分数归一化+同维度加权）
        
        融合策略：
        1. 分数归一化：统一关键词和向量分数到 0-1 范围
        2. 双库命中加分：同时被两库检索到的结果额外 +15%
        3. 同维度线索加权：与对比维度匹配的属性线索额外 ×1.1
        """
        fused = {}
        
        max_keyword = max([r.get("score", 0) for r in keyword_results], default=1)
        max_vector = max([r.get("score", 0) for r in vector_results], default=1)
        
        DUAL_HIT_BONUS = 0.15
        DIMENSION_BONUS = 1.1
        
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

        if dimension and dimension != "general":
            for chunk_id, r in fused.items():
                clue_type = r.get("clue_type", "").lower()
                if dimension.lower() in clue_type or clue_type in dimension.lower():
                    r["fused_score"] *= DIMENSION_BONUS

        sorted_results = sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)
        return sorted_results[:self.recall_top_k]

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """去重（优化版：chunk_id为主+实体维度哈希为辅，保留高分多类型线索）
        
        去重策略：
        1. chunk_id 完全去重（全局唯一标识）
        2. 文本哈希辅助去重相似内容
        3. 保留「双库命中、融合分数更高」的线索
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
    
    def _rerank(self, question: str, entities: List[str], dimension: str, results: List[Dict]) -> List[Dict]:
        """重排序（优化版：同维度加权+维度覆盖校验）
        
        重排序策略：
        1. 查询增强：双实体+对比维度+问题核心
        2. 同维度线索加权：×1.2
        3. 维度覆盖校验：确保结果包含目标维度
        """
        if not results:
            return []
        
        enhanced_query = self._build_enhanced_query(question, entities, dimension)
        
        if self.reranker.is_available():
            passages = [r.get("core_text", "") for r in results]
            rerank_scores = self.reranker.rerank(enhanced_query, passages)
            
            for r, score in zip(results, rerank_scores):
                base_score = float(score)
                
                dimension_bonus = 0.0
                clue_type = r.get("clue_type", "").lower()
                if dimension and dimension != "general":
                    if dimension.lower() in clue_type:
                        dimension_bonus = 0.2
                
                dual_hit_bonus = 0.1 if len(r.get("retrieval_types", [])) >= 2 else 0.0
                
                r["rerank_score"] = base_score + dimension_bonus + dual_hit_bonus
            
            results = sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        results = self._filter_by_dimension_coverage(results, entities, dimension)
        
        return results
    
    def _build_enhanced_query(self, question: str, entities: List[str], dimension: str) -> str:
        """构建增强的查询"""
        query_parts = []
        
        if entities and len(entities) >= 2:
            query_parts.extend([entities[0], entities[1]])
        elif entities:
            query_parts.extend(entities)
        
        if dimension and dimension != "general":
            query_parts.append(dimension)
        
        query_parts.append(question)
        
        return " ".join(query_parts)
    
    def _filter_by_dimension_coverage(self, results: List[Dict], entities: List[str], dimension: str) -> List[Dict]:
        """筛选：优先保留覆盖目标维度的线索"""
        if not dimension or dimension == "general":
            return results
        
        covered_results = []
        uncovered_results = []
        
        for r in results:
            clue_type = r.get("clue_type", "").lower()
            if dimension.lower() in clue_type:
                covered_results.append(r)
            else:
                uncovered_results.append(r)
        
        return (covered_results + uncovered_results)[:self.recall_top_k]

    def _format_results(self, results: List[Dict], entities: List[str] = None, dimension: str = "") -> List[Dict]:
        """格式化结果（优化版：添加entity_tag和attr_dimension，适配对比推理）
        
        返回格式：
        - core_text: 核心文本
        - chunk_id: 唯一标识
        - score: 融合分数
        - metadata:
          - entity_tag: 实体关联标签（entity_a/entity_b/both）
          - attr_dimension: 属性维度（对比维度）
          - retrieval_types: 检索类型列表
          - dual_hit: 是否双库命中
          - rerank_score: 重排序分数
        """
        formatted = []
        seen = set()

        for r in results:
            chunk_id = r.get("chunk_id", "")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                
                retrieval_types = r.get("retrieval_types", [])
                dual_hit = len(retrieval_types) >= 2 if retrieval_types else False
                entity_tag = r.get("entity_tag", "both")
                
                formatted.append({
                    "core_text": r.get("core_text", ""),
                    "chunk_id": chunk_id,
                    "score": r.get("rerank_score", r.get("fused_score", 0.0)),
                    "metadata": {
                        "entity_tag": entity_tag,
                        "attr_dimension": dimension,
                        "clue_type": r.get("clue_type", ""),
                        "retrieval_types": retrieval_types,
                        "dual_hit": dual_hit,
                        "rerank_score": r.get("rerank_score", 0.0),
                        "fused_score": r.get("fused_score", 0.0)
                    }
                })

        return formatted


def get_comparison_retriever() -> ComparisonRetriever:
    """获取比较问题检索器（单例）"""
    return ComparisonRetriever()
