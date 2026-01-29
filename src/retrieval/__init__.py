"""Multi-Channel Retrieval Package

Modules:
- vector_retriever: 向量检索 (BGE + FAISS)
- bm25_retriever: BM25关键词检索
- graph_retriever: 知识图谱检索
- entity_extractor: 实体抽取
- multi_channel_retrieval: 多路融合 + 重排序 + 生成答案
"""

from .vector_retriever import VectorRetriever, VectorRetrievalResult
from .bm25_retriever import BM25Retriever, BM25RetrievalResult
from .graph_retriever import GraphRetriever, GraphRetrievalResult
from .entity_extractor import EntityExtractor, EntityExtractionResult
from .multi_channel_retrieval import (
    KBLoader,
    MultiChannelRetriever,
    MultiHopReasoner,
    Reranker,
    AnswerGenerator,
    FusionResult,
    demo,
)

__all__ = [
    "VectorRetriever",
    "VectorRetrievalResult",
    "BM25Retriever", 
    "BM25RetrievalResult",
    "GraphRetriever",
    "GraphRetrievalResult",
    "EntityExtractor",
    "EntityExtractionResult",
    "KBLoader",
    "MultiChannelRetriever",
    "MultiHopReasoner",
    "Reranker",
    "AnswerGenerator",
    "FusionResult",
    "demo",
]
