#!/usr/bin/env python3
"""Entity Extraction module for HotpotQA"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("[EntityExtractor] Warning: spacy not available, NER disabled")

try:
    from src.models.glm import GLMClient, get_glm_client
    GLM_AVAILABLE = True
except ImportError:
    GLM_AVAILABLE = False
    print("[EntityExtractor] Warning: GLM client not available")


@dataclass
class EntityExtractionResult:
    entities: List[str]
    bridge_entities: List[str]
    comparison_entities: List[str]
    source: str


class EntityExtractor:
    """实体抽取（支持GLM增强）"""
    
    _glm_client = None
    
    def __init__(self, use_glm: bool = True):
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[EntityExtractor] Warning: en_core_web_sm not found, NER disabled")
        
        self.glm_client = None
        if use_glm and GLM_AVAILABLE:
            if EntityExtractor._glm_client is None:
                try:
                    EntityExtractor._glm_client = GLMClient()
                    if EntityExtractor._glm_client.client is None:
                        EntityExtractor._glm_client = None
                except Exception:
                    EntityExtractor._glm_client = None
            self.glm_client = EntityExtractor._glm_client
        
        self.comp_patterns = [
            r'(?P<e1>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:and|or)\s+(?P<e2>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:difference|similarity|between)\s+(?P<e1>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+and\s+(?P<e2>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?P<e1>What\s+is\s+(?:the\s+)?(?:name|identity|who)\s+of\s+(?:\w+\s+)?(?P<e2>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        self._entity_cache = {}
    
    def _clean_entity(self, entity: str) -> str:
        """清洗实体"""
        entity = entity.strip()
        entity = re.sub(r'^\W+|\W+$', '', entity)
        return entity
    
    def _extract_by_ner(self, text: str) -> List[str]:
        """使用SpaCy NER抽取实体"""
        if self.nlp is None:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"]:
                    cleaned = self._clean_entity(ent.text)
                    if len(cleaned) > 1:
                        entities.append(cleaned)
            return entities
        except Exception as e:
            print(f"[EntityExtractor] NER error: {e}")
            return []
    
    def _extract_by_regex(self, text: str) -> List[str]:
        """使用正则表达式抽取实体"""
        patterns = [
            r'(?:is|are|was|were|been being)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:of|for|in|at|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'"([^"]+)"',
            r"'([^']+)'",
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                cleaned = self._clean_entity(match)
                if len(cleaned) > 2:
                    entities.append(cleaned)
        return entities
    
    def extract_entities(self, text: str, use_cache: bool = True) -> List[str]:
        """
        抽取问题中的实体（优先使用GLM）
        
        Args:
            text: 问题文本
            use_cache: 是否使用缓存
            
        Returns:
            实体列表
        """
        if not text or not text.strip():
            return []
        
        cache_key = f"entities:{text}"
        if use_cache and cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        
        entities = []
        
        if self.glm_client is not None:
            entities = self._extract_by_glm(text)
            if entities:
                entities = list(dict.fromkeys(entities))
                if use_cache:
                    self._entity_cache[cache_key] = entities
                return entities
        
        if self.nlp:
            try:
                entities.extend(self._extract_by_ner(text))
            except Exception as e:
                print(f"[EntityExtractor] NER extraction error: {e}")
        
        entities.extend(self._extract_by_regex(text))
        
        entities = list(dict.fromkeys(entities))
        
        if use_cache:
            self._entity_cache[cache_key] = entities
        
        return entities
    
    def _extract_by_glm(self, text: str) -> List[str]:
        """使用GLM抽取实体"""
        prompt = f"""
Extract all named entities from this question. Focus on:
- Person names
- Organization names  
- Location names
- Work titles (films, books, etc.)
- Event names

Question: {text}

Return only entity names, one per line. If no entities found, return empty.
"""
        
        try:
            response = self.glm_client.generate(prompt, max_tokens=100)
            if response:
                entities = []
                for line in response.split('\n'):
                    cleaned = self._clean_entity(line)
                    if cleaned and len(cleaned) > 2:
                        entities.append(cleaned)
                return entities
        except Exception as e:
            print(f"[EntityExtractor] GLM extraction error: {e}")
        
        return []
    
    def extract_bridge_entities(self, text: str) -> List[str]:
        """
        抽取桥接实体（用于多跳推理）
        
        Args:
            text: 问题文本
            
        Returns:
            桥接实体列表
        """
        if not text or not text.strip():
            return []
        
        all_entities = self.extract_entities(text)
        
        question_words = {'what', 'who', 'which', 'where', 'when', 'how', 'why', 'is', 'are', 'was', 'were'}
        words = text.lower().split()
        
        bridge_entities = []
        for entity in all_entities:
            entity_words = entity.lower().split()
            if not any(w in question_words for w in entity_words):
                bridge_entities.append(entity)
        
        return bridge_entities[:5]
    
    def extract_bridge_entities_glm(self, text: str) -> List[str]:
        """使用GLM抽取桥接实体"""
        if self.glm_client is None:
            return []
        
        prompt = f"""
Extract the bridge entities needed for multi-hop reasoning in this question.
A bridge entity is the key entity that connects to the second-hop information.

Question: {text}

Example:
Q: "What government position was held by the woman who portrayed Ella Black in the film?"
A: "Ella Black"

Return only the bridge entity names, one per line.
"""
        
        try:
            response = self.glm_client.generate(prompt, max_tokens=100)
            if response:
                entities = []
                for line in response.split('\n'):
                    cleaned = self._clean_entity(line)
                    if cleaned and len(cleaned) > 2:
                        entities.append(cleaned)
                return entities[:3]
        except Exception:
            pass
        
        return []
    
    def extract_comparison_entities(self, text: str) -> List[str]:
        """
        抽取比较型问题的两个实体
        
        Args:
            text: 问题文本
            
        Returns:
            两个比较实体的列表
        """
        if not text or not text.strip():
            return []
        
        entities = []
        
        for pattern in self.comp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                for key in ['e1', 'e2']:
                    if key in match.group():
                        entity = match.group(key)
                        cleaned = self._clean_entity(entity)
                        if cleaned and len(cleaned) > 2:
                            entities.append(cleaned)
                if len(entities) >= 2:
                    break
        
        if len(entities) < 2:
            all_entities = self.extract_entities(text)
            entities = [e for e in all_entities if len(e) > 3][:2]
        
        return entities[:2]
    
    def extract_comparison_entities_glm(self, text: str) -> List[str]:
        """使用GLM抽取比较实体"""
        if self.glm_client is None:
            return []
        
        prompt = f"""
Extract the two main entities being compared in this question. 
Return only the entity names, separated by "|||".

Question: {text}

Example output: "New York|||Los Angeles"
"""
        
        try:
            response = self.glm_client.generate(prompt, max_tokens=100)
            if response:
                parts = response.split('|||')
                entities = [self._clean_entity(p.strip()) for p in parts if p.strip()]
                return [e for e in entities if len(e) > 2]
        except Exception as e:
            print(f"[EntityExtractor] GLM error: {e}")
        
        return []
    
    def extract_entities_glm(self, text: str) -> List[str]:
        """使用GLM抽取实体"""
        if self.glm_client is None:
            return []
        
        prompt = f"""
Extract all named entities (person names, organizations, locations, etc.) from this question.
Return only the entity names, one per line.

Question: {text}
"""
        
        try:
            response = self.glm_client.generate(prompt, max_tokens=200)
            if response:
                entities = []
                for line in response.split('\n'):
                    cleaned = self._clean_entity(line)
                    if cleaned and len(cleaned) > 2:
                        entities.append(cleaned)
                return entities
        except Exception as e:
            print(f"[EntityExtractor] GLM error: {e}")
        
        return []
    
    def extract(self, text: str, q_type: str = "bridge") -> EntityExtractionResult:
        """
        综合实体抽取
        
        Args:
            text: 问题文本
            q_type: 问题类型
            
        Returns:
            EntityExtractionResult
        """
        entities = self.extract_entities(text)
        
        if q_type == "comparison":
            if len(entities) < 2 and self.glm_client:
                glm_entities = self.extract_comparison_entities_glm(text)
                if len(glm_entities) >= 2:
                    entities = glm_entities
            
            comparison = self.extract_comparison_entities(text)
            if len(comparison) >= 2:
                entities = comparison
            
            return EntityExtractionResult(
                entities=entities,
                bridge_entities=[],
                comparison_entities=entities[:2],
                source="comparison"
            )
        else:
            bridge = self.extract_bridge_entities(text)
            if len(bridge) < 2 and self.glm_client:
                glm_bridge = self.extract_bridge_entities_glm(text)
                if glm_bridge:
                    bridge = glm_bridge
            
            return EntityExtractionResult(
                entities=entities,
                bridge_entities=bridge,
                comparison_entities=[],
                source="bridge"
            )
    
    def clear_cache(self):
        """清除缓存"""
        self._entity_cache.clear()
