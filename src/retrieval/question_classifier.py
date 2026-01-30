#!/usr/bin/env python3
"""Question Type Classifier

使用util.py中的HotpotQARuleProcessor纯规则引擎进行实体提取
问题类型直接使用数据集标注，无需判断

不再依赖任何大模型调用
"""

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util import HotpotQARuleProcessor


class QuestionClassifier:
    """基于纯规则的问题类型分类器和实体提取器"""

    def __init__(self):
        self.processor = HotpotQARuleProcessor()

    def classify(self, question: str) -> Dict:
        """使用纯规则判断问题类型（桥接/比较）

        Args:
            question: Question text

        Returns:
            Dict with type and confidence
        """
        if not question:
            return {
                "type": "bridge",
                "confidence": 0.0,
                "reasoning": "empty question"
            }

        result = self.processor.judge_question_type(question)
        return {
            "type": result["type"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"]
        }

    def extract_entities(self, question: str, question_type: str) -> List[str]:
        """使用纯规则根据问题类型提取实体

        Args:
            question: Question text
            question_type: "bridge" or "comparison" (from dataset annotation)

        Returns:
            List of extracted entities
        """
        if not question:
            return []

        return self.processor.extract_entities_by_type(question, question_type)

    def process(self, question: str, question_type: str = None) -> Dict:
        """全流程处理：实体提取

        Args:
            question: Question text
            question_type: Optional question type (comparison/bridge) from dataset

        Returns:
            Dict with type and entities
        """
        return self.processor.process(question, question_type)


def get_question_classifier() -> QuestionClassifier:
    """获取分类器实例（单例模式）"""
    return QuestionClassifier()
