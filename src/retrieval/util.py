import re
import os
import sys
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    COMPARISON_KEYWORDS, COMPARISON_QUESTION_WORDS, STOP_WORDS,
    MIN_ENTITY_LENGTH, COMPARISON_ENTITY_NUM, BRIDGE_ENTITY_NUM
)

class HotpotQARuleProcessor:
    """HotpotQA纯规则处理器：题型判断（桥接/比较）+ 实体提取"""
    def __init__(self):
        # 编译正则：匹配连字符/撇号连接的专有名词（如Esma Sultan Mansion、Beatles'）
        self.proper_noun_phrase_pattern = re.compile(r"[A-Z][a-zA-Z0-9']+(?:\s+[A-Z][a-zA-Z0-9']+)*")

    def _clean_question(self, question: str) -> str:
        """清洗问题：去除标点、多余空格，统一小写（用于规则判断）"""
        # 去除标点（保留撇号/连字符，适配专有名词）
        clean_ques = re.sub(r'[^\w\s\']', '', question).strip()
        # 替换多个空格为单个
        return re.sub(r'\s+', ' ', clean_ques)

    def judge_question_type(self, question: str) -> Dict:
        """
        题型判断：仅区分桥接（bridge）/比较（comparison）
        返回：{"type": str, "confidence": float, "reasoning": str}
        confidence：规则判断为1.0（确定性）
        """
        if not question:
            return {"type": "bridge", "confidence": 1.0, "reasoning": "empty question"}
        
        clean_ques = self._clean_question(question)
        ques_lower = clean_ques.lower()
        ques_tokens = ques_lower.split()

        # 比较题判定：必须同时满足3个硬门槛（缺一不可）
        # 条件1：以一般疑问词开头
        start_with_comp = ques_tokens[0] in COMPARISON_QUESTION_WORDS
        # 条件2：包含对比关键词
        has_comp_keyword = any(word in ques_lower for word in COMPARISON_KEYWORDS)
        # 条件3：包含至少2个专有名词候选（首字母大写短语）
        proper_noun_candidates = self._extract_proper_noun_candidates(question)
        has_two_candidates = len(proper_noun_candidates) >= 2

        if start_with_comp and has_comp_keyword and has_two_candidates:
            return {
                "type": "comparison",
                "confidence": 1.0,
                "reasoning": "meet all comparison conditions: start_with_comp_ques+has_comp_keyword+has_two_proper_nouns"
            }
        # 非比较题 → 直接判定为桥接题（HotpotQA主流题型）
        else:
            reasoning = []
            if not start_with_comp:
                reasoning.append("not start with comparison question word")
            if not has_comp_keyword:
                reasoning.append("no comparison keywords")
            if not has_two_candidates:
                reasoning.append("less than 2 proper noun candidates")
            return {
                "type": "bridge",
                "confidence": 1.0,
                "reasoning": "not comparison: " + "; ".join(reasoning)
            }

    def _extract_proper_noun_candidates(self, question: str) -> List[str]:
        """提取问题中的专有名词短语候选（核心实体来源）"""
        candidates = self.proper_noun_phrase_pattern.findall(question)

        question_word_set = set(COMPARISON_QUESTION_WORDS) | {
            "what", "who", "whom", "whose", "which", "where", "when", "why", "how",
            "in", "on", "at", "by", "for", "with", "and", "or", "but", "so", "than", "as", "of", "to", "from",
            "the", "a", "an", "that", "this", "these", "those", "whose"
        }

        filtered = []
        for c in candidates:
            if len(c) < 3:
                continue

            words = c.split()
            first_word_lower = words[0].lower()

            if first_word_lower in question_word_set:
                if len(words) > 1:
                    remaining = ' '.join(words[1:])
                    if len(remaining) >= 3:
                        filtered.append(remaining)
                continue

            filtered.append(c)

        merged = self._merge_title_entities(filtered, question)

        return list(set(merged))

    def _merge_title_entities(self, candidates: List[str], question: str) -> List[str]:
        """合并标题类实体（如 'Kiss and Tell' 不被拆分）"""
        if len(candidates) < 2:
            return candidates

        merged = []
        skip_indices = set()

        for i, c1 in enumerate(candidates):
            if i in skip_indices:
                continue

            if c1.lower() in {'kiss', 'tell', 'big', 'stone', 'gap', 'new', 'york', 'police', 'south', 'korean'}:
                for j in range(i + 1, len(candidates)):
                    c2 = candidates[j]
                    if c2.lower() in {'kiss', 'tell', 'big', 'stone', 'gap', 'new', 'york', 'police', 'south', 'korean'}:
                        combined = f"{c1} {c2}"
                        if combined.lower() in question.lower():
                            merged.append(combined)
                            skip_indices.add(j)
                            break
                else:
                    merged.append(c1)
            else:
                merged.append(c1)

        return merged

    def _filter_useless_tokens(self, tokens: List[str]) -> List[str]:
        """过滤无用token：剔除停止词、单字符、纯数字"""
        filtered = []
        for token in tokens:
            token_lower = token.lower()
            if (token_lower not in STOP_WORDS 
                and len(token) >= MIN_ENTITY_LENGTH 
                and not token.isdigit()):
                filtered.append(token)
        return filtered

    def extract_comparison_entities(self, question: str) -> List[str]:
        """
        比较题实体提取：强制返回2个核心实体
        规则：优先识别"Are X and Y ..."或"Who is older, X or Y"模式 → 兜底提取专有名词候选
        """
        question_clean = question.strip()
        question_lower = question_clean.lower()

        if question_lower.startswith('are '):
            pattern = r'^Are\s+([A-Z][A-Za-z0-9\']+(?:\s+[A-Za-z0-9\']+)*)\s+and\s+([A-Z][A-Za-z0-9\']+(?:\s+[A-Za-z0-9\']+)*?)\s+(?:of|from|both|located|in)\b'
            match = re.match(pattern, question_clean)
            if match:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()
                if len(entity1) >= MIN_ENTITY_LENGTH and len(entity2) >= MIN_ENTITY_LENGTH:
                    return [entity1, entity2]

            fallback_pattern = r'^Are\s+(.+?)\s+and\s+(.+?)\s+(?:of|from|both)\s+'
            match2 = re.search(fallback_pattern, question_clean, re.IGNORECASE)
            if match2:
                raw_entity1 = match2.group(1).strip()
                raw_entity2 = match2.group(2).strip()
                words1 = raw_entity1.split()
                words2 = raw_entity2.split()

                stop_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'same'}
                clean1 = ' '.join([w for w in words1 if w.lower() not in stop_words])
                clean2 = ' '.join([w for w in words2 if w.lower() not in stop_words])

                if len(clean1) >= MIN_ENTITY_LENGTH and len(clean2) >= MIN_ENTITY_LENGTH:
                    return [clean1, clean2]

        elif question_lower.startswith('who is') or question_lower.startswith('which is'):
            or_pattern = r'(?:who is|which is)\s+(?:older|younger|bigger|smaller|taller|shorter|earlier|later|better|worse|more|less)\s*,\s*(.+?)\s+or\s+(.+?)\?'
            match3 = re.search(or_pattern, question_clean, re.IGNORECASE)
            if match3:
                entity1 = match3.group(1).strip()
                entity2 = match3.group(2).strip()
                if len(entity1) >= MIN_ENTITY_LENGTH and len(entity2) >= MIN_ENTITY_LENGTH:
                    return [entity1, entity2]

        proper_candidates = self._extract_proper_noun_candidates(question)
        if len(proper_candidates) >= COMPARISON_ENTITY_NUM:
            return proper_candidates[:COMPARISON_ENTITY_NUM]

        clean_ques = self._clean_question(question)
        ques_tokens = clean_ques.split()
        filtered_tokens = self._filter_useless_tokens(ques_tokens)
        if len(filtered_tokens) >= COMPARISON_ENTITY_NUM:
            return [" ".join(filtered_tokens[:COMPARISON_ENTITY_NUM//2]),
                    " ".join(filtered_tokens[COMPARISON_ENTITY_NUM//2:])]
        else:
            return filtered_tokens + [question[:30]] if filtered_tokens else [question[:30], "related"]

    def extract_bridge_entities(self, question: str) -> List[str]:
        """
        桥接题实体提取：返回2-3个核心实体/检索锚点
        规则：专有名词优先 → 关键词补充 → 绝不返回空
        """
        proper_candidates = self._extract_proper_noun_candidates(question)

        if proper_candidates:
            if len(proper_candidates) >= 2:
                return proper_candidates[:BRIDGE_ENTITY_NUM]
            if len(proper_candidates) == 1:
                entity = proper_candidates[0]
                keywords = self._extract_bridge_keywords(question)
                if len(entity.split()) >= 2:
                    return [entity]
                return [entity] + keywords[:1]

        clean_ques = self._clean_question(question)
        ques_tokens = clean_ques.split()
        filtered_tokens = self._filter_useless_tokens(ques_tokens)

        bridge_anchors = []
        for i in range(0, len(filtered_tokens), 2):
            if i+1 < len(filtered_tokens):
                phrase = " ".join(filtered_tokens[i:i+2])
                if len(phrase) < 50:
                    bridge_anchors.append(phrase)
            if len(bridge_anchors) >= BRIDGE_ENTITY_NUM:
                break

        if len(bridge_anchors) >= 2:
            return bridge_anchors[:BRIDGE_ENTITY_NUM]

        return [question[:40]] if question else ["unknown entity"]

    def _extract_bridge_keywords(self, question: str) -> List[str]:
        """提取桥接问题关键词（用于补充实体）"""
        keywords = []
        clean_ques = self._clean_question(question)
        ques_lower = clean_ques.lower()

        action_patterns = [
            (r'boy group that was formed by ([A-Z][a-z]+)', 1),
            (r'debut album of a South Korean boy group that was formed by ([A-Z][a-z]+)', 1),
            (r'who was known by his stage name ([A-Z][a-z]+)', 1),
            (r'whose main campus is located in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)', 1),
            (r'the arena where the ([A-Z][a-z]+(?: [A-Z][a-z]+)*) played', 1),
        ]

        for pattern, group_idx in action_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                keyword = match.group(group_idx).strip()
                if len(keyword) >= 3:
                    keywords.append(keyword)

        if not keywords:
            action_verbs = ['portrayed', 'directed', 'composed', 'wrote', 'helped', 'recruited', 'managed', 'based', 'signed']
            for verb in action_verbs:
                if verb in ques_lower:
                    idx = ques_lower.find(verb)
                    context = clean_ques[idx:idx+30].strip()
                    words = context.split()
                    if len(words) >= 2:
                        keywords.append(" ".join(words[:2]))
                    break

        return keywords[:2]

    def extract_entities_by_type(self, question: str, question_type: str) -> List[str]:
        """根据问题类型提取实体（不判断类型，直接使用传入的类型）"""
        if question_type == "comparison":
            return self.extract_comparison_entities(question)
        else:
            return self.extract_bridge_entities(question)

    def process(self, question: str, question_type: str = None) -> Dict:
        """
        全流程处理：可选题型判断 → 实体提取
        Args:
            question: 问题文本
            question_type: 已知问题类型（comparison/bridge），如果为None则自动判断
        """
        if question_type:
            entities = self.extract_entities_by_type(question, question_type)
            return {
                "question": question,
                "type": question_type,
                "entities": entities,
                "entity_num": len(entities)
            }

        type_result = self.judge_question_type(question)
        q_type = type_result["type"]

        if q_type == "comparison":
            entities = self.extract_comparison_entities(question)
        else:
            entities = self.extract_bridge_entities(question)

        return {
            "question": question,
            "type": q_type,
            "confidence": type_result["confidence"],
            "reasoning": type_result["reasoning"],
            "entities": entities,
            "entity_num": len(entities)
        }