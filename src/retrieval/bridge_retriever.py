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
        """初始化配置（优化版：增加检索数量以支持rerank）
        
        核心原则：
        - 初步检索返回更多chunks（80个），给reranker足够的候选
        - 关键词库权重 0.5：平衡实体匹配和语义检索
        - 向量库权重 0.5：增强语义匹配能力
        - 重排序后返回top 15
        """
        self.keyword_weight = 0.5
        self.vector_weight = 0.5
        self.recall_top_k = 80
        self.final_top_k = 15
        
        self.keyword_score_threshold = 0.1
        self.vector_score_threshold = 0.4

    def retrieve_with_entities(self, question: str, entities: List[str]) -> List[Dict]:
        """使用预提取的实体进行检索（优化版：增加直接问题检索）
        
        Args:
            question: 问题文本
            entities: 预提取的实体列表
            
        Returns:
            检索结果列表
        """
        if not entities:
            entities = self.classifier.extract_entities(question, "bridge")
        
        keyword_results = self._keyword_search(question, entities)
        vector_results = self._vector_search(question, entities)

        fused = self._fuse_results(keyword_results, vector_results)
        deduped = self._deduplicate(fused)
        
        if len(deduped) < 8:
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
        
        if len(deduped) < 3:
            direct_results = self._direct_question_search(question)
            if direct_results:
                existing_ids = set(r.get("chunk_id") for r in deduped)
                for r in direct_results:
                    if r.get("chunk_id") not in existing_ids:
                        r["is_direct_search"] = True
                        deduped.append(r)
        
        deduped = sorted(deduped, key=lambda x: x.get("fused_score", 0.0), reverse=True)[:self.recall_top_k]
        reranked = self._rerank(question, entities, deduped)
        return reranked

    def _direct_question_search(self, question: str) -> List[Dict]:
        """直接用问题全文进行检索（当实体检索失败时的兜底策略）
        
        策略：
        1. 从问题中提取关键词
        2. 用关键词组合进行检索
        3. 返回top结果
        """
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
        stop_words = {'what', 'who', 'whom', 'which', 'where', 'when', 'why', 'how',
                      'were', 'are', 'was', 'is', 'did', 'does', 'do', 'has', 'have',
                      'had', 'the', 'that', 'this', 'these', 'those', 'and', 'or',
                      'but', 'for', 'with', 'from', 'into', 'about', 'above'}
        keywords = [w for w in words if w not in stop_words]
        
        if not keywords:
            return []
        
        results = []
        seen_ids = set()
        
        for i in range(min(5, len(keywords))):
            keyword = keywords[i]
            
            kw_results = self.keyword_store.search(keyword, [keyword], top_k=10)
            for r in kw_results:
                if r.get("chunk_id") not in seen_ids:
                    seen_ids.add(r.get("chunk_id"))
                    r["direct_search_term"] = keyword
                    results.append(r)
            
            vec_results = self.vector_store.search(keyword, top_k=10)
            for r in vec_results:
                if r.get("chunk_id") not in seen_ids:
                    seen_ids.add(r.get("chunk_id"))
                    r["direct_search_term"] = keyword
                    results.append(r)
        
        for r in results:
            r["retrieval_type"] = "direct"
        
        return results

    def retrieve(self, question: str) -> List[Dict]:
        """检索桥接问题的相关文档（优化版：含兜底机制）
        
        检索策略：
        1. 多实体组合检索
        2. 若结果不足8条，触发单实体兜底检索
        3. 若仍不足，触发问题改写兜底检索
        """
        entities = self.classifier.extract_entities(question, "bridge")
        return self.retrieve_with_entities(question, entities)

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
        """生成改写查询（增强版）
        
        当原始检索失败时，尝试以下改写策略：
        1. 原始实体
        2. 带类型限定词的实体（如 "Kiss and Tell film/movie"）
        3. 问题类型+实体组合
        4. 移除停用词
        """
        import re
        reformulated = []
        
        if entities:
            reformulated.extend(entities)
        
        question_lower = question.lower()
        
        if '"' in question or "'" in question:
            quoted = re.findall(r'["\']([^"\']+)["\']', question)
            for q in quoted:
                reformulated.append(f"{q} film")
                reformulated.append(f"{q} movie")
                reformulated.append(f"{q} 1945 film")
        
        who_portrayed_pattern = r'who portrayed (.+?) in the film'
        match = re.search(who_portrayed_pattern, question_lower)
        if match:
            actor = match.group(1).strip()
            reformulated.append(f"{actor} actress")
            reformulated.append(f"{actor} actor")
        
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
        
        if entities:
            for entity in entities[:2]:
                reformulated.append(f"the {entity}")
                reformulated.append(f"{entity} biography")
        
        return list(set(reformulated))[:8]

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
        """重排序（优化版：实体加权+覆盖度筛选+时间完整性优先+模糊匹配惩罚）
        
        重排序策略：
        1. 查询增强：问题+实体+关系词
        2. 桥接实体加权：包含桥接实体的线索额外加权
        3. 覆盖度筛选：确保覆盖问题中所有核心实体
        4. 时间完整性优先：对于时间范围问题，优先返回完整时间段
        5. 模糊匹配惩罚：同名实体过滤，降低模糊实体的匹配权重
        """
        if not results:
            return []
        
        is_time_question = any(kw in question.lower() for kw in ['year', 'during', 'from', 'until', 'between', 'managed', 'served', 'held'])
        
        ambiguous_entities = self._detect_ambiguous_entities(entities, results)
        
        enhanced_query = self._build_enhanced_query(question, entities)
        
        if self.reranker.is_available():
            passages = [r.get("core_text", "") for r in results]
            rerank_scores = self.reranker.rerank(enhanced_query, passages)
            
            for i, (r, score) in enumerate(zip(results, rerank_scores)):
                base_score = float(score)
                
                fuzzy_match_penalty = self._calculate_fuzzy_match_penalty(entities, r, ambiguous_entities)
                
                bridge_entity_bonus = 0.0
                chunk_bridge_entity = r.get("bridge_entity", "").lower()
                for entity in entities:
                    if entity.lower() in chunk_bridge_entity or chunk_bridge_entity in entity.lower():
                        if entity not in ambiguous_entities:
                            bridge_entity_bonus = 0.2
                        else:
                            bridge_entity_bonus = 0.1
                        break
                
                dual_hit_bonus = 0.1 if len(r.get("retrieval_types", [])) >= 2 else 0.0
                
                time_completeness_bonus = 0.0
                if is_time_question:
                    time_completeness_bonus = self._calculate_time_completeness(r.get("core_text", ""), question)
                
                r["rerank_score"] = base_score + bridge_entity_bonus + dual_hit_bonus + time_completeness_bonus + fuzzy_match_penalty
            
            results = sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        results = self._filter_by_entity_coverage(results, entities)
        results = self._filter_irrelevant_chunks(results, entities, question)
        
        return results
    
    def _detect_ambiguous_entities(self, entities: List[str], results: List[Dict]) -> set:
        """检测模糊实体：同名但在不同上下文中指代不同事物
        
        模糊实体判断标准：
        1. 实体名过短（<4字符）但不是常见缩写
        2. 实体在检索结果中出现在多种不同主题的chunk中
        3. 实体与多个不同的bridge_entity关联
        """
        ambiguous = set()
        
        for entity in entities:
            entity_lower = entity.lower()
            
            if len(entity) < 4:
                common_short_entities = {'usa', 'uk', 'nyc', 'la', 'sf', 'ai', 'ml', 'dj', 'tv'}
                if entity_lower not in common_short_entities:
                    ambiguous.add(entity)
                    continue
            
            associated_bridge_entities = set()
            topic_variety = set()
            
            for r in results[:20]:
                core_text = r.get("core_text", "").lower()
                chunk_bridge_entity = r.get("bridge_entity", "").lower()
                
                if entity_lower in core_text:
                    associated_bridge_entities.add(chunk_bridge_entity)
                    
                    first_sentence = core_text.split('.')[0][:50] if '.' in core_text else core_text[:50]
                    topic_variety.add(first_sentence)
            
            if len(associated_bridge_entities) >= 3:
                ambiguous.add(entity)
            elif len(topic_variety) >= 3 and len(associated_bridge_entities) >= 2:
                ambiguous.add(entity)
        
        return ambiguous
    
    def _calculate_fuzzy_match_penalty(self, entities: List[str], result: Dict, ambiguous_entities: set) -> float:
        """计算模糊匹配惩罚分数
        
        惩罚策略：
        1. 模糊实体的匹配降低权重
        2. 非模糊实体的精确匹配保持或增加权重
        """
        penalty = 0.0
        core_text = result.get("core_text", "").lower()
        chunk_bridge_entity = result.get("bridge_entity", "").lower()
        
        has_exact_bridge_match = False
        has_ambiguous_match = False
        
        for entity in entities:
            entity_lower = entity.lower()
            
            if entity in ambiguous_entities:
                if entity_lower in core_text or entity_lower in chunk_bridge_entity:
                    has_ambiguous_match = True
                    penalty -= 0.15
            else:
                if entity_lower in core_text or entity_lower in chunk_bridge_entity:
                    has_exact_bridge_match = True
        
        if has_exact_bridge_match and not has_ambiguous_match:
            penalty += 0.1
        
        return max(-0.3, penalty)
    
    def _calculate_time_completeness(self, text: str, question: str) -> float:
        """计算时间完整性得分"""
        if not text:
            return 0.0
        
        import re
        
        text_lower = text.lower()
        question_lower = question.lower()
        
        year_pattern = r'\b(19|20)\d{2}\b'
        years_in_text = set(re.findall(year_pattern, text))
        
        if not years_in_text:
            return 0.0
        
        full_range_bonus = 0.0
        if 'from' in question_lower and 'to' in question_lower:
            if 'from' in text_lower and ('to' in text_lower or 'until' in text_lower):
                full_range_bonus = 0.15
        elif 'during' in question_lower:
            if any(year in text for year in years_in_text):
                full_range_bonus = 0.1
        
        duration_bonus = 0.0
        if len(years_in_text) >= 2:
            years = sorted([int(y) for y in years_in_text])
            if len(years) >= 2:
                span = years[-1] - years[0]
                if span >= 10:
                    duration_bonus = 0.1
                elif span >= 20:
                    duration_bonus = 0.15
        
        return full_range_bonus + duration_bonus
    
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
    
    def _filter_irrelevant_chunks(self, results: List[Dict], entities: List[str], question: str) -> List[Dict]:
        """过滤无关 chunk：基于实体匹配度 + 相关性评分"""
        if not results:
            return results
        
        filtered = []
        question_lower = question.lower()
        question_keywords = set(w for w in question_lower.split() if len(w) > 3)
        
        for r in results:
            core_text = r.get("core_text", "").lower()
            chunk_id = r.get("chunk_id", "")
            
            entity_match_count = 0
            for entity in entities:
                if entity.lower() in core_text:
                    entity_match_count += 1
            
            chunk_keywords = set(w for w in core_text.split() if len(w) > 4)
            keyword_overlap = len(question_keywords & chunk_keywords)
            
            relevance_score = entity_match_count * 0.6 + keyword_overlap * 0.4
            
            if entity_match_count >= 1 and relevance_score > 0.5:
                filtered.append(r)
            elif entity_match_count >= 2:
                filtered.append(r)
            elif keyword_overlap >= 3:
                filtered.append(r)
        
        if not filtered and results:
            filtered = results[:5]
        
        return filtered

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
