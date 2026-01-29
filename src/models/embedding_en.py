#!/usr/bin/env python3
"""Embedding Model Client - BGE English"""

import os
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = {}


class EmbeddingClient:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        local_model_path = "/Users/xry/.cache/modelscope/hub/models/BAAI/bge-base-en-v1.5"
        self.model_path = local_model_path
        self.model = self._get_model()

    def _get_model(self):
        cache_key = f"{self.model_name}"
        if cache_key in _MODEL_CACHE:
            print(f"[Embedding] Use cached model: {self.model_name}")
            return _MODEL_CACHE[cache_key]

        print(f"[Embedding] Load from local: {self.model_path}")
        model = SentenceTransformer(self.model_path)
        _MODEL_CACHE[cache_key] = model
        print(f"[Embedding] Ready")
        return model

    def encode(self, texts, normalize: bool = True):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=normalize)

    def similarity(self, texts1, texts2):
        emb1 = self.encode(texts1, normalize=True)
        emb2 = self.encode(texts2, normalize=True)
        return emb1 @ emb2.T

    def cosine_similarity(self, query, documents):
        q_emb = self.encode([query], normalize=True)
        d_emb = self.encode(documents, normalize=True)
        return (q_emb @ d_emb.T)[0].tolist()


if __name__ == "__main__":
    client = EmbeddingClient()
    emb = client.encode(["hello world", "test"])
    print(f"Dim: {emb.shape}")
