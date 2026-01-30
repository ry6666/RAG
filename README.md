# RAG 多跳问答系统

基于检索增强生成的多跳问答系统，支持双库检索、BGE重排序，适用于 HotpotQA 风格的多跳推理问题。

## 系统架构

```
RAG/
├── config/                    # 配置管理
│   └── config.py              # 全局配置参数
├── models/                    # 模型管理
│   ├── __init__.py            # 模块导出
│   ├── embedding_model.py     # BGE嵌入模型（向量生成）
│   ├── reranker_model.py      # BGE重排模型（结果重排序）
│   └── ollama_model.py        # Ollama生成模型（答案生成）
├── src/
│   ├── kb/                    # 知识库模块
│   │   ├── parquet_processor.py   # Parquet数据预处理
│   │   ├── vector_store.py        # FAISS向量库
│   │   ├── keyword_store.py       # Whoosh关键词库
│   │   └── validator.py           # 知识库验证
│   ├── retrieval/             # 检索模块
│   │   ├── dual_retriever.py      # 双库检索器（关键词+向量）
│   │   ├── differential_retriever.py  # 差异化检索策略
│   │   └── question_classifier.py # 问题类型分类
│   └── generation/            # 生成模块
│       └── answer_generator.py    # 答案生成器
├── scripts/                   # 脚本工具
│   └── extract_test_samples.py   # 测试样本提取
├── test/                      # 测试评估
│   ├── samples/               # 测试样本
│   │   └── validation_samples.json
│   ├── evaluate_rag.py        # RAG系统评估脚本
│   └── results/               # 评估报告输出
├── kb/                        # 知识库存储
│   ├── vector_index/          # FAISS向量索引
│   └── keyword_index/         # Whoosh关键词索引
├── data/                      # 数据集
│   ├── train/                 # 训练集Parquet
│   └── validation/            # 验证集Parquet
└── DEPENDENCIES.md            # 依赖说明
```

## 核心模块说明

### 1. 模型层（models/）

| 文件 | 功能描述 |
|------|----------|
| [embedding_model.py](file:///Users/xry/Desktop/python/projects/RAG/models/embedding_model.py) | 管理BGE-Base-en-v1.5嵌入模型，负责将文本转换为768维向量。采用单例模式，支持本地缓存加载。 |
| [reranker_model.py](file:///Users/xry/Desktop/python/projects/RAG/models/reranker_model.py) | 管理BGE-Reranker-base重排模型，对召回结果进行精细排序。使用XLM-RoBERTa架构。 |
| [ollama_model.py](file:///Users/xry/Desktop/python/projects/RAG/models/ollama_model.py) | 管理Ollama本地LLM服务，用于答案生成。支持chat和generate两种模式，默认模型llama3.2。 |

### 2. 知识库层（src/kb/）

| 文件 | 功能描述 |
|------|----------|
| [parquet_processor.py](file:///Users/xry/Desktop/python/projects/RAG/kb/parquet_processor.py) | Parquet数据批处理：读取训练集、提取核心字段（question、answer、supporting_facts）、结构化分块（支持句+上下文）、生成chunk_id和元信息。 |
| [vector_store.py](file:///Users/xry/Desktop/python/projects/RAG/kb/vector_store.py) | FAISS向量库管理：使用BGE模型生成向量，支持IndexFlatL2和IndexIVFFlat两种索引，实现向量相似度检索。 |
| [keyword_store.py](file:///Users/xry/Desktop/python/projects/RAG/kb/keyword_store.py) | Whoosh关键词库管理：构建倒排索引，使用BM25评分（k1=1.2, b=0.75），桥接实体权重×2，支持增量更新。 |
| [validator.py](file:///Users/xry/Desktop/python/projects/RAG/kb/validator.py) | 知识库验证工具：检查chunk完整性、索引覆盖率、重复数据检测，支持知识库维护和清理。 |

### 3. 检索层（src/retrieval/）

| 文件 | 功能描述 |
|------|----------|
| [dual_retriever.py](file:///Users/xry/Desktop/python/projects/RAG/src/retrieval/dual_retriever.py) | **核心检索器**：双库召回+结果融合+BGE重排序+过滤。支持问题类型差异化权重配置（桥接题0.7/0.3，比较题0.6/0.4）。 |
| [differential_retriever.py](file:///Users/xry/Desktop/python/projects/RAG/src/retrieval/differential_retriever.py) | 差异化检索策略：根据问题类型（桥接/比较/一般）自动调整检索参数和重排策略。 |
| [question_classifier.py](file:///Users/xry/Desktop/python/projects/RAG/src/retrieval/question_classifier.py) | 问题类型分类器：识别桥接题（also、还、曾等模式）、比较题（more、谁比谁等模式）、一般问题，提取实体（大写短语）。 |

### 4. 生成层（src/generation/）

| 文件 | 功能描述 |
|------|----------|
| [answer_generator.py](file:///Users/xry/Desktop/python/projects/RAG/src/generation/answer_generator.py) | 答案生成器：整合检索结果和Ollama模型，生成最终答案，支持ReAct推理模式。 |

### 5. 测试评估（test/）

| 文件 | 功能描述 |
|------|----------|
| [evaluate_rag.py](file:///Users/xry/Desktop/python/projects/RAG/test/evaluate_rag.py) | RAG系统评估脚本：加载测试样本、执行检索、提取答案、计算准确率、生成JSON和Markdown报告。 |

## 使用方法

### 1. 提取测试样本

```bash
python scripts/extract_test_samples.py 10
```

### 2. 运行评估

```bash
python test/evaluate_rag.py
```

评估结果保存在 `test/results/` 目录下：
- `evaluation_results.json` - 详细结果数据
- `evaluation_report.md` - 可读报告

## 配置说明

### 检索权重配置

```python
# dual_retriever.py
self.weights = {
    "bridge": {"keyword": 0.7, "vector": 0.3},      # 桥接题：关键词优先
    "comparison": {"keyword": 0.6, "vector": 0.4},  # 比较题：关键词偏重
    "general": {"keyword": 0.5, "vector": 0.5}      # 一般题：均衡
}
```

### BM25参数

```python
# keyword_store.py
BM25F(k1=1.2, b=0.75, field_weights={"bridge_entity": 2.0, "clue_type": 1.5})
```

## 依赖环境

- Python 3.8+
- faiss-cpu / faiss-gpu
- whoosh
- pyarrow（Parquet读写）
- transformers（模型加载）
- requests（Ollama通信）
- numpy, tqdm

## 检索流程

```
用户问题 → 问题分类 → 实体提取 → 双库检索（关键词+向量）
                                    ↓
                              结果融合（加权）
                                    ↓
                              BGE重排序
                                    ↓
                              过滤去重
                                    ↓
                              返回Top-K结果
```

## 输出示例

评估报告包含：
- 整体准确率
- 按问题类型统计
- 每个样本的预测答案与标准答案对比
- 检索结果详情
