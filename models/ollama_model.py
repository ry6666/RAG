#!/usr/bin/env python3
"""Ollama Model Manager

Manages Ollama generation models
"""

import requests
import json

class OllamaModel:
    """Ollama model manager"""

    _instance = None
    _base_url = None

    def __new__(cls, base_url: str = "http://localhost:11434"):
        if cls._instance is None or cls._base_url != base_url:
            cls._instance = super().__new__(cls)
            cls._instance._init(base_url)
        return cls._instance

    def _init(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.default_model = "qwen2:7b-instruct"
        self._connected = False
        self._check_connection()

    def _check_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                available = [m['name'] for m in data.get('models', [])]
                self._connected = True
            else:
                self._connected = False
        except Exception as e:
            self._connected = False

    def generate(self, prompt: str, model: str = None, temperature: float = 0.1, max_tokens: int = 256) -> str:
        model = model or self.default_model
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60, stream=True)
            if response.status_code == 200:
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data:
                            content = data['message'].get('content', '')
                            full_content += content
                return full_content
        except Exception as e:
            pass
        return ""

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


_ollama_instance = None

def get_ollama_model(base_url: str = "http://localhost:11434") -> OllamaModel:
    global _ollama_instance
    _ollama_instance = OllamaModel(base_url)
    return _ollama_instance
