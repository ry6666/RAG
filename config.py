# config.py - 项目全局配置文件（三路检索+多跳推理）
# 原则：按模块分组 + 全大写命名 + 仅存常量/参数/路径

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================
# 1. 三路检索配置（核心：向量+BM25+DPR）
# ==============================================
RETRIEVAL_SINGLE_TOP_K = 20
RETRIEVAL_NORMALIZE_SCORE = True
RETRIEVAL_FUSION_TOP_K = 10
THREE_WEIGHTS = (0.4, 0.3, 0.3)
SUPPORTED_RETRIEVERS = ["vector", "bm25", "dpr"]
VECTOR_INDEX_PATH = os.path.join(PROJECT_ROOT, "kb/vector.index")
BM25_MODEL_PATH = os.path.join(PROJECT_ROOT, "kb/bm25.pkl")
DPR_EMBEDDING_DIM = 1024

# ==============================================
# 2. 重排序配置
# ==============================================
RERANK_TOP_K = 5
RERANK_SCORE_WEIGHT = (0.3, 0.7)
RERANK_TEXT_TRUNCATION = 512
RERANK_BATCH_SIZE = 16

# ==============================================
# 3. 多跳推理配置（知识图谱相关）
# ==============================================
REASONING_MAX_HOPS = 2
ENTITY_NEIGHBOR_TOP_K = 10
BRIDGE_ENTITY_TOP_K = 20
KG_GRAPH_PATH = os.path.join(PROJECT_ROOT, "kb/graph.pkl")
ENTITY_LINK_THRESHOLD = 0.75

# ==============================================
# 4. 模型配置（嵌入/重排序/本地大模型）
# ==============================================
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_NORMALIZE = True
RERANK_MODEL = "BAAI/bge-reranker-base"
OLLAMA_MODEL = "qwen2:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"

# ==============================================
# 5. 评估配置
# ==============================================
EVAL_TOP_K_LIST = [5, 10]
EVAL_BATCH_SIZE = 10
EVAL_METRICS = ["hit", "mrr", "precision", "ndcg"]
EVAL_SAVE_RESULT = True
EVAL_RESULT_PATH = os.path.join(PROJECT_ROOT, "results")

# ==============================================
# 6. 路径配置（数据/输出/日志）
# ==============================================
KB_DIR = os.path.join(PROJECT_ROOT, "kb")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for dir_path in [KB_DIR, DATA_DIR, OUTPUT_DIR, LOG_DIR, RESULTS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# ==============================================
# 7. 运行配置（开发/生产环境）
# ==============================================
ENV = "development"

if ENV == "development":
    DEBUG = True
    TRAIN_BATCH_SIZE = 8
    TEST_SAMPLE_NUM = 100
else:
    DEBUG = False
    TRAIN_BATCH_SIZE = 32
    TEST_SAMPLE_NUM = -1
