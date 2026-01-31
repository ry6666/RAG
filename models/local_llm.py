#!/usr/bin/env python3
"""Local LLM Interface

统一本地LLM调用接口
支持: Ollama
"""

import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.ollama_model import OllamaModel


class LocalLLM:
    """本地LLM统一接口"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.ollama = OllamaModel()
        self._available = self.ollama._connected

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        """生成文本"""
        if not self._available:
            return ""
        
        try:
            return self.ollama.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception:
            return ""

    def is_available(self) -> bool:
        return self._available


_local_llm_instance = None

def get_local_llm() -> LocalLLM:
    """获取本地LLM实例"""
    global _local_llm_instance
    if _local_llm_instance is None:
        _local_llm_instance = LocalLLM()
    return _local_llm_instance
