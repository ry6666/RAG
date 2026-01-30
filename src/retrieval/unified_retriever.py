#!/usr/bin/env python3
"""Unified Retriever

统一检索入口
负责：
1. 判断问题类型（桥接/比较）
2. 调用相应的专用检索器
3. 返回统一格式的检索结果
"""

import os
import sys
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from question_classifier import QuestionClassifier
from bridge_retriever import BridgeRetriever, get_bridge_retriever
from comparison_retriever import ComparisonRetriever, get_comparison_retriever


class UnifiedRetriever:
    """统一检索器"""

    def __init__(self):
        self.classifier = QuestionClassifier()
        self.bridge_retriever = None
        self.comparison_retriever = None

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

    def retrieve(self, question: str) -> Tuple[List[Dict], Dict]:
        """检索问题相关文档

        Args:
            question: 问题文本

        Returns:
            Tuple[检索结果列表, 问题分析信息]
            检索结果格式: [{"core_text": "...", "chunk_id": "...", "score": 0.0, "metadata": {...}}]
            分析信息格式: {"type": "...", "entities": [...], "confidence": 0.0}
        """
        classification = self.classifier.classify(question)
        question_type = classification.get("type", "bridge")
        confidence = classification.get("confidence", 0.0)
        entities = self.classifier.extract_entities(question, question_type)

        if question_type == "bridge":
            results = self._retrieve_bridge(question)
        else:
            results = self._retrieve_comparison(question)

        analysis = {
            "type": question_type,
            "entities": entities,
            "confidence": confidence,
            "reasoning": classification.get("reasoning", "")
        }

        return results, analysis

    def _retrieve_bridge(self, question: str) -> List[Dict]:
        """检索桥接问题"""
        if self.bridge_retriever is None:
            return []

        return self.bridge_retriever.retrieve(question)

    def _retrieve_comparison(self, question: str) -> List[Dict]:
        """检索比较问题"""
        if self.comparison_retriever is None:
            return []

        return self.comparison_retriever.retrieve(question)

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
