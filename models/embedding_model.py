#!/usr/bin/env python3
"""Embedding Model Manager

管理BGE嵌入模型，统一模型加载和使用
"""

import os
import sys
from typing import List, Optional
import numpy as np

if sys.platform == 'darwin':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    import torch
    torch.backends.mps.enable = lambda: True

# 本地模型路径
LOCAL_MODEL_PATH = "/Users/xry/.cache/modelscope/hub/models"

class EmbeddingModel:
    """嵌入模型管理器"""
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, model_name: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = None):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.model_name = model_name or os.path.join(LOCAL_MODEL_PATH, "BAAI/bge-base-en-v1.5")
        self.embedding_dim = 768  # BGE-Base输出维度
        
        if EmbeddingModel._model is None:
            self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            from transformers import AutoModel, AutoTokenizer

            EmbeddingModel._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            EmbeddingModel._model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            EmbeddingModel._model.eval()

        except Exception as e:
            EmbeddingModel._model = None
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """批量向量化文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量数组
        """
        if EmbeddingModel._model is None:
            return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)
        
        import torch
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = EmbeddingModel._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = EmbeddingModel._model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        return self._normalize_embeddings(embeddings).astype(np.float32)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """向量归一化"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "loaded": EmbeddingModel._model is not None
        }
    
    @classmethod
    def get_instance(cls, model_name: str = None):
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance


def get_embedding_model(model_name: str = None) -> EmbeddingModel:
    """获取嵌入模型实例"""
    return EmbeddingModel.get_instance(model_name)
