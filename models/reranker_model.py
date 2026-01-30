#!/usr/bin/env python3
"""Reranker Model Manager

管理BGE重排模型，统一模型加载和使用
"""

import os
import sys
import numpy as np
from typing import List, Optional

if sys.platform == 'darwin':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    import torch
    torch.backends.mps.enable = lambda: True

# 本地模型路径
LOCAL_MODEL_PATH = "/Users/xry/.cache/modelscope/hub/models"

class RerankerModel:
    """重排模型管理器"""
    
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
        
        self.model_name = model_name or os.path.join(LOCAL_MODEL_PATH, "BAAI/bge-reranker-base")
        
        if RerankerModel._model is None:
            self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            RerankerModel._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            RerankerModel._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            RerankerModel._model.eval()

        except Exception as e:
            RerankerModel._model = None
    
    def rerank(self, query: str, passages: List[str], top_k: int = None) -> List[float]:
        """重排序
        
        Args:
            query: 查询文本
            passages: 文本段落列表
            top_k: 返回数量
            
        Returns:
            得分列表
        """
        try:
            if not RerankerModel._model or not passages:
                return [0.0] * len(passages)
            
            valid_passages = []
            for p in passages:
                if p and isinstance(p, str) and len(p.strip()) > 0:
                    valid_passages.append(p.strip()[:1000])
            
            if not valid_passages:
                return [0.0] * len(passages)
            
            import torch
            
            pairs = [[query, passage] for passage in valid_passages]
            
            batch_size = 32
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                inputs = RerankerModel._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = RerankerModel._model(**inputs)
                    batch_scores = outputs.logits.squeeze().cpu().numpy()
                
                batch_scores = np.array(batch_scores)
                if batch_scores.ndim == 0:
                    all_scores.append(float(batch_scores))
                else:
                    all_scores.extend(batch_scores.tolist())
            
            if top_k is not None and top_k < len(valid_passages):
                scores_with_idx = [(s, i) for i, s in enumerate(all_scores)]
                scores_with_idx.sort(key=lambda x: x[0], reverse=True)
                top_indices = [idx for _, idx in scores_with_idx[:top_k]]
                return [1.0 if i in top_indices else 0.0 for i in range(len(valid_passages))]
            
            return all_scores
            
        except Exception as e:
            print(f"[RerankerModel] Rerank error: {e}")
            return [0.0] * len(passages)
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return RerankerModel._model is not None
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "loaded": RerankerModel._model is not None
        }
    
    @classmethod
    def get_instance(cls, model_name: str = None):
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance


def get_reranker_model(model_name: str = None) -> RerankerModel:
    """获取重排模型实例"""
    return RerankerModel.get_instance(model_name)
