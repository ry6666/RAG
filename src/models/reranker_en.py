#!/usr/bin/env python3
"""Reranker Model Client - BGE Reranker Base"""

import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['PYTHONMALLOC'] = 'default'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(1)

_MODEL_CACHE = {}


class RerankerClient:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        local_model_path = "/Users/xry/.cache/modelscope/hub/models/BAAI/bge-reranker-base"
        self.model_path = local_model_path
        self.tokenizer = None
        self.model = None
        self._init_model()

    def _init_model(self):
        cache_key = f"{self.model_name}"
        if cache_key in _MODEL_CACHE:
            print(f"[Reranker] Use cached model: {self.model_name}")
            self.tokenizer, self.model = _MODEL_CACHE[cache_key]
            return

        print(f"[Reranker] Load from local: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32
        )
        self.model.eval()
        _MODEL_CACHE[cache_key] = (self.tokenizer, self.model)
        print("[Reranker] Ready")

    def rerank(self, query, passages, top_k: int = 5):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        scores = []
        for passage in passages:
            with torch.no_grad():
                inputs = self.tokenizer(
                    query, passage,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    padding=True
                )
                outputs = self.model(**inputs)
                score = torch.sigmoid(outputs.logits[0][0]).item()
                scores.append(score)
        
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(passages[i], s) for i, s in indexed]


if __name__ == "__main__":
    client = RerankerClient()
    results = client.rerank("what is deep learning", ["Deep learning is...", "Machine learning is..."])
    for p, s in results:
        print(f"[{s:.4f}] {p[:50]}...")
