#!/usr/bin/env python3
"""Project Configuration File

按照用户要求的配置：
1. 知识库构建：仅用支持文档，细粒度切分，双库存储
2. 检索策略：按问题类型针对性设计
3. 生成策略：保准确率，无无依据生成
4. 评估策略：双场景测试
"""

import os
from typing import Dict, List, Tuple

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DATA_DIR = os.path.join(DATA_DIR, "validation")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# 知识库目录
KB_DIR = os.path.join(PROJECT_ROOT, "kb")
VECTOR_INDEX_DIR = os.path.join(KB_DIR, "vector_index")
INDEX_DIR = os.path.join(KB_DIR, "index")

# 模型配置
MODEL_CONFIG = {
    "embedding": {
        "model_name": "/Users/xry/.cache/modelscope/hub/models/BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "normalize": True,
        "device": "cpu"
    },
    "reranker": {
        "model_name": "/Users/xry/.cache/modelscope/hub/models/BAAI/bge-reranker-base",
        "device": "cpu"
    },
    "llm": {
        "model_name": "qwen2:7b-instruct",
        "base_url": "http://localhost:11434/api/generate",
        "max_tokens": 512,
        "temperature": 0.1
    }
}
# ==================== 可配置参数（核心关键词/规则）====================
# 比较题必备：对比关键词表（HotpotQA高频）
COMPARISON_KEYWORDS = {
    "same", "different", "alike", "similar", "identical", "equivalent",
    "both", "either", "neither", "compare", "older", "younger", "bigger",
    "smaller", "taller", "shorter", "earlier", "later", "higher", "lower",
    "larger", "heavier", "lighter", "more", "less", "vs", "versus"
}
# 比较题必备：一般疑问词开头（判定是否为一般疑问句）
COMPARISON_QUESTION_WORDS = {"are", "is", "was", "were", "do", "does", "did", "have", "has", "had", "who", "which"}
# 全局过滤词：功能词/虚词（提取实体时彻底剔除）
STOP_WORDS = {
    # 语法功能词/疑问词/介词/连词
    "the", "a", "an", "what", "who", "whom", "whose", "which", "where", "when", "why", "how",
    "in", "on", "at", "by", "for", "with", "and", "or", "but", "so", "than", "as", "of", "to", "from",
    "that", "this", "these", "those", "it", "they", "he", "she", "we", "you", "i",
    # 通用名词（无检索价值）
    "series", "book", "film", "song", "painting", "museum", "person", "people", "man", "woman",
    "position", "job", "career", "place", "location", "city", "country", "year", "time", "story"
}
# 实体判定规则：首字母大写的词/短语为核心候选（HotpotQA实体均为专有名词）
def is_proper_noun_token(token: str) -> bool:
    """判断单个词是否为专有名词候选（首字母大写+非单个字母）"""
    return len(token) > 1 and token[0].isupper() and token[1:].islower() is False

# 最小实体长度：剔除过短的无意义实体
MIN_ENTITY_LENGTH = 2
# 实体数量约束
COMPARISON_ENTITY_NUM = 2  # 比较题强制2个实体
BRIDGE_ENTITY_NUM = 3      # 桥接题最多3个实体（不足则取所有，最少1个）
# 知识库构建配置
KB_CONFIG = {
    "min_chunk_length": 50,
    "max_chunk_length": 500,
    "context_window": 1,  # 前后各1句上下文
    "batch_size": 128,
    "vector_index_type": "faiss",
    "index_chunk_size": 10000
}

# 检索配置
RETRIEVAL_CONFIG = {
    "top_k": {
        "bridge": 10,
        "comparison": 8,
        "general": 10
    },
    "weights": {
        "bridge": {
            "index": 0.7,
            "vector": 0.3
        },
        "comparison": {
            "index": 0.6,
            "vector": 0.4
        },
        "general": {
            "index": 0.5,
            "vector": 0.5
        }
    },
    "thresholds": {
        "similarity": 0.5,
        "entity_match": 0.8
    }
}

# 问题类型判定配置
QUESTION_TYPE_CONFIG = {
    "bridge_indicators": [
        r'也', r'还', r'曾', r'既.*又.*', r'谁.*', r'什么.*关联',
        r'关联', r'连接', r'联系', r'有关', r'相关'
    ],
    "comparison_indicators": [
        r'更', r'谁比谁', r'是否相同', r'差异', r'区别',
        r'比较', r'对比', r'相比', r'不同于', r'不同于'
    ]
}

# 生成配置
GENERATION_CONFIG = {
    "max_context_length": 4000,
    "bridge_prompt": """You are a knowledge reasoning assistant. Follow these steps to answer bridge questions:

1. First, explicitly identify the bridge entity that connects the clues
2. Then, connect the clues using this bridge entity
3. Finally, derive the answer based on the connected clues

If no valid bridge entity or clues are found, respond with "I cannot answer this question based on the provided information."

Question: {question}

Clues:
{clues}

Bridge Entity:
Answer:
""",
    "comparison_prompt": """You are a knowledge reasoning assistant. Follow these steps to answer comparison questions:

1. First, clearly identify the comparison dimension
2. Then, integrate features of both targets
3. Finally, objectively output the comparison conclusion

If no valid comparison dimension or features are found, respond with "I cannot answer this question based on the provided information."

Question: {question}

Clues:
{clues}

Comparison Dimension:
Answer:
""",
    "react_prompt": """You are a multi-hop reasoning expert for HotpotQA bridge questions. Based on the provided retrieval clues, reason step-by-step in the following format to give an accurate answer:
1. Thought: Clearly state current known information, information gaps, and the next bridge entity/clue to find;
2. Action: Choose only "Retrieve[entity/keyword to search]" or "Integrate" - select Integrate if no additional retrieval is needed;
3. Observation: Record the result of the Action (retrieval clues for Retrieve, summary of existing clues for Integrate);
4. Maximum 3 cycles. Output Answer when information is sufficient, or "Insufficient bridge clues to answer" when no valid clues.

Initial retrieval clues: {clues}
Question: {question}
"""
}

# ReAct配置
REACT_CONFIG = {
    "max_cycles": 3,
    "action_types": ["Retrieve", "Integrate"],
    "max_retrieve_tokens": 100,
    "integration_threshold": 0.7,
    "confidence_threshold": 0.6
}

# 评估配置
EVALUATION_CONFIG = {
    "top_k_values": [5, 10],
    "interference_ratio": 5,  # 5:1 干扰比例
    "metrics": ["recall", "mrr", "accuracy"],
    "batch_size": 10
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(LOGS_DIR, "app.log")
}

# 确保目录存在
for dir_path in [
    DATA_DIR, TRAIN_DATA_DIR, VALIDATION_DATA_DIR,
    OUTPUT_DIR, RESULTS_DIR, LOGS_DIR,
    KB_DIR, VECTOR_INDEX_DIR, INDEX_DIR
]:
    os.makedirs(dir_path, exist_ok=True)

# 导出配置
def get_config() -> Dict:
    """获取完整配置"""
    return {
        "project_root": PROJECT_ROOT,
        "data_dir": DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "kb_dir": KB_DIR,
        "model_config": MODEL_CONFIG,
        "kb_config": KB_CONFIG,
        "retrieval_config": RETRIEVAL_CONFIG,
        "question_type_config": QUESTION_TYPE_CONFIG,
        "generation_config": GENERATION_CONFIG,
        "evaluation_config": EVALUATION_CONFIG,
        "log_config": LOG_CONFIG
    }

if __name__ == "__main__":
    # 验证配置
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Project Root: {config['project_root']}")
    print(f"Data Dir: {config['data_dir']}")
    print(f"KB Dir: {config['kb_dir']}")
    print(f"Embedding Model: {config['model_config']['embedding']['model_name']}")
    print(f"LLM Model: {config['model_config']['llm']['model_name']}")
