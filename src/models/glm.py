#!/usr/bin/env python3
"""GLM Client - Entity Extraction and Answer Generation via ZhipuAI API"""

import os
import sys
import json
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from zhipuai import ZhipuAI
    ZHIPUAI_AVAILABLE = True
except ImportError:
    ZHIPUAI_AVAILABLE = False


_glm_client_instance = None


def get_glm_client(api_key: str = None, model: str = "glm-4") -> 'GLMClient':
    """获取 GLM 客户端单例"""
    global _glm_client_instance
    if _glm_client_instance is None:
        _glm_client_instance = GLMClient(api_key, model)
    return _glm_client_instance


class GLMClient:
    _instance = None
    _initialized = False
    
    def __new__(cls, api_key: str = None, model: str = "glm-4"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, api_key: str = None, model: str = "glm-4"):
        if GLMClient._initialized:
            return
        
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(proj_root, '.env')
        
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path)
        except ImportError:
            pass
        
        self.api_key = api_key or os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key and ZHIPUAI_AVAILABLE:
            self._init_api_client()
        else:
            if not self.api_key:
                pass
            elif not ZHIPUAI_AVAILABLE:
                pass
        
        GLMClient._initialized = True
    
    def _init_api_client(self):
        try:
            self.client = ZhipuAI(api_key=self.api_key)
        except Exception as e:
            self.client = None

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self.client:
            return self._generate_api(prompt, max_tokens)
        else:
            return ""

    def _generate_api(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[GLM] API error: {e}")
            return ""

    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        prompt = """You are a knowledge graph expert. Extract knowledge triplets from the following text in JSON format.

Text: {text}

Triplet format: {{"subject": "...", "predicate": "...", "object": "..."}}

Relation types:
- establish/create/found (A establishes B)
- replace/overthrow/defeat (A replaces B)
- capital/move capital (A makes B capital)
- rule/rule during (A rules during period B)
- battle/fight/defeat (A battles/fights/defeats B)
- invent/create (A invents/creates B)
- belong/originate from (A belongs to B, A originates from B)
- location/situated at (A is located at B)
- inherit/succeed (A inherits from B)

Return only the JSON array, nothing else.
[
    {{"subject": "Xia Dynasty", "predicate": "established", "object": "Qi"}},
    {{"subject": "Qi", "predicate": "established", "object": "Xia Dynasty"}}
]""".format(text=text)

        result = self.generate(prompt, max_tokens=512)
        try:
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.startswith("["):
                result = result
            else:
                result = result[result.find("["):]
            if result.endswith("```"):
                result = result[:-3]
            triplets = json.loads(result)
            if isinstance(triplets, dict) and "triplets" in triplets:
                triplets = triplets["triplets"]
            return [dict(t) for t in triplets]
        except json.JSONDecodeError as e:
            print(f"[GLM] Failed to parse triplets: {e}")
            return []

    def answer_question(self, question: str, context: List[str]) -> str:
        context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context)])
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't know".

Context:
{context_text}

Question: {question}

Answer:"""

        return self.generate(prompt, max_tokens=256)


if __name__ == "__main__":
    client = GLMClient()

    if client.client:
        test_text = "Xia Dynasty was established by Yu the Great. Qi was founded by Tang of Shang."
        triplets = client.extract_triplets(test_text)
        print(f"Extracted {len(triplets)} triplets:")
        for t in triplets:
            print(f"  {t}")
    else:
        print("GLM client not available - check ZHIPUAI_API_KEY")
