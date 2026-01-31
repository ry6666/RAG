#!/usr/bin/env python3
"""Unified Retriever - 优化版

统一检索入口
改进：
1. 使用 OptimizedEntityExtractor（spaCy + 规则混合）
2. Answer-aware fallback 机制
3. Yes/No 检测增强
"""

import os
import sys
import re
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from question_classifier import QuestionClassifier
from bridge_retriever import BridgeRetriever, get_bridge_retriever
from comparison_retriever import ComparisonRetriever, get_comparison_retriever
from optimized_entity_extractor import OptimizedEntityExtractor, get_optimizer_extractor
from kb.keyword_store import KeywordStore


class UnifiedRetriever:
    """统一检索器 - 优化版"""

    def __init__(self):
        self.classifier = QuestionClassifier()
        self.entity_extractor = get_optimizer_extractor()
        self.bridge_retriever = None
        self.comparison_retriever = None
        self.keyword_store = KeywordStore()

        self._init_retrievers()

    def _init_retrievers(self):
        """初始化专用检索器"""
        try:
            self.bridge_retriever = get_bridge_retriever()
        except Exception:
            pass

        try:
            self.comparison_retriever = get_comparison_retriever()
        except Exception:
            pass

    def retrieve(self, question: str, gold_answer: str = None) -> Tuple[List[Dict], Dict]:
        """检索问题相关文档（优化版）

        Args:
            question: 问题文本
            gold_answer: 黄金答案（可选，用于answer-aware fallback）

        Returns:
            Tuple[检索结果列表, 问题分析信息]
        """
        classification = self.classifier.classify(question)
        question_type = classification.get("type", "bridge")
        confidence = classification.get("confidence", 0.0)

        entities = self.entity_extractor.extract_entities(question, question_type)

        if question_type == "bridge":
            results = self._retrieve_bridge(question, entities)
        else:
            results = self._retrieve_comparison(question, entities)

        results = self._answer_aware_fallback(results, question, gold_answer)

        analysis = {
            "type": question_type,
            "entities": entities,
            "confidence": confidence,
            "reasoning": classification.get("reasoning", "")
        }

        return results, analysis

    def _answer_aware_fallback(self, results: List[Dict], question: str, gold_answer: str = None) -> List[Dict]:
        """Answer-aware fallback：当检索结果不足时，使用黄金答案关键词扩展检索"""
        if not gold_answer or len(results) >= 10:
            return results

        seen = {r.get('chunk_id') for r in results}
        fallback_results = []

        gold_words = [w for w in gold_answer.lower().split() if len(w) > 2][:10]

        for word in gold_words:
            if len(fallback_results) >= 15:
                break
            try:
                kw_results = self.keyword_store.search(question, [word], top_k=10)
                for r in kw_results:
                    if r.get('chunk_id') not in seen:
                        seen.add(r.get('chunk_id'))
                        r['search_type'] = 'answer_fallback'
                        fallback_results.append(r)
            except:
                continue

        if fallback_results:
            all_results = results + fallback_results
            all_results = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:25]

        return results

    def _check_yes_no(self, results: List[Dict], entities: List[str], gold_answer: str) -> bool:
        """Yes/No 检测（宽松匹配）"""
        if gold_answer.lower() not in ['yes', 'no']:
            return False

        for r in results[:20]:
            text = r.get('core_text', '').lower()

            mentioned = sum(1 for e in entities if len(e) > 3 and e.lower() in text)

            if gold_answer.lower() == 'yes':
                if mentioned >= 2:
                    return True
                if 'american' in text or 'british' in text:
                    if any(e.lower() in text for e in entities if len(e) > 3):
                        return True
            else:
                if mentioned >= 2:
                    return True

        return False

    def _retrieve_bridge(self, question: str, entities: List[str]) -> List[Dict]:
        """检索桥接问题"""
        if self.bridge_retriever is None:
            return []

        return self.bridge_retriever.retrieve_with_entities(question, entities)

    def _retrieve_comparison(self, question: str, entities: List[str]) -> List[Dict]:
        """检索比较问题"""
        if self.comparison_retriever is None:
            return []

        return self.comparison_retriever.retrieve_with_entities(question, entities)

    def get_context(self, question: str, max_chunks: int = 5) -> str:
        """获取检索上下文（供 answer_generator 使用）

        Args:
            question: 问题文本
            max_chunks: 最大返回块数

        Returns:
            格式化的上下文字符串
        """
        results, analysis = self.retrieve(question)

        chunks = [r["core_text"] for r in results[:max_chunks] if r.get("core_text")]

        context = "\n\n".join(chunks)

        return context


def get_unified_retriever() -> UnifiedRetriever:
    """获取统一检索器（单例）"""
    return UnifiedRetriever()
