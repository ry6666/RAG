import os
import json
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class ZhipuAIClient:
    def __init__(self, api_key: str = None, base_url: str = "https://open.bigmodel.cn/api/paas/v4"):
        self.api_key = api_key or os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not self.api_key:
            raise ValueError("GLM_API_KEY/ZHIPU_API_KEY not found in environment or arguments")
        self.base_url = base_url
        self.default_model = "glm-4.7-flash"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            response = requests.get(
                f"{self.base_url.replace('/chat/completions', '')}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, model: str = None, temperature: float = 0.1, max_tokens: int = 512) -> str:
        model = model or self.default_model
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "").strip()
                if content:
                    return content
                reasoning = message.get("reasoning_content", "").strip()
                if reasoning:
                    return self._extract_final_answer(reasoning)
                return ""
            else:
                print(f"[ZhipuAI] Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[ZhipuAI] Request error: {e}")
        return ""
    
    def _extract_final_answer(self, reasoning: str) -> str:
        """从推理内容中提取最终答案"""
        if not reasoning:
            return ""
        
        import re
        
        lines = reasoning.strip().split('\n')
        
        final_answer_patterns = [
            r'["\']*(Yes|No|UNKNOWN|Unknown)["\']*\.?\s*$',
            r'Construct Output:?\s*["\']*(Yes|No|UNKNOWN|Unknown)["\']*',
            r'Answer is:?\s*["\']*(Yes|No|UNKNOWN|Unknown)["\']*',
            r'Final Answer:?\s*["\']*(Yes|No|UNKNOWN|Unknown)["\']*',
        ]
        
        for pattern in final_answer_patterns:
            for line in reversed(lines):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        last_line = lines[-1].strip() if lines else ""
        if last_line and len(last_line) < 50:
            last_line = re.sub(r'^[\d.\s\*]+', '', last_line).strip()
            last_line = re.sub(r'^[*\s]+', '', last_line).strip()
            if last_line:
                return last_line
        
        sentences = [s.strip() for s in reasoning.split('.') if s.strip() and len(s.strip()) < 100]
        if sentences:
            for sent in reversed(sentences):
                if re.match(r'^(Yes|No|Unknown|UNKNOWN)', sent, re.IGNORECASE):
                    return re.split(r'[,:\s]', sent, 1)[0].strip()
        
        return ""

    def generate_with_history(self, messages: List[Dict[str, str]], model: str = None, temperature: float = 0.1, max_tokens: int = 512) -> str:
        model = model or self.default_model
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "").strip()
                if content:
                    return content
                reasoning = message.get("reasoning_content", "").strip()
                if reasoning:
                    return self._extract_final_answer(reasoning)
                return ""
            else:
                print(f"[ZhipuAI] Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[ZhipuAI] Request error: {e}")
        return ""

_instance = None

def get_zhipu_model() -> ZhipuAIClient:
    global _instance
    if _instance is None:
        _instance = ZhipuAIClient()
    return _instance
