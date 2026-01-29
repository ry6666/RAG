#!/usr/bin/env python3
"""Build Knowledge Base from Local Parquet Dataset (Fixed Version)"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 环境变量配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# 项目根路径配置（更鲁棒）
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# 核心依赖导入+缺失检查
REQUIRED_PACKAGES = ['faiss', 'numpy', 'pandas', 'networkx', 'spacy', 'rank_bm25', 'tqdm']
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[Error] Missing required package: {pkg}, please install it first!")
        sys.exit(1)

import json
import pickle
import faiss
import numpy as np
import pandas as pd
import networkx as nx
import spacy
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import re
from tqdm import tqdm  # 更友好的进度条
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# 导入自定义模块（增加异常捕获）
try:
    from src.models.embedding_en import EmbeddingClient
except ImportError as e:
    print(f"[Warning] EmbeddingClient import failed: {e}, please check src/models/embedding_en.py")
    sys.exit(1)

# 全局配置
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'not', 'only', 'just', 'also', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'it', 'its'
}

def tokenize_text(text: str) -> List[str]:
    """分词函数（全局函数，支持多进程pickle）"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return [w for w in words if w and len(w) > 1 and w not in STOPWORDS]

def load_parquet_data(split: str = "train", sample: Optional[int] = None, data_dir: str = "dataset", use_head: bool = False) -> pd.DataFrame:
    """Load local parquet dataset (修复路径硬编码+数据加载容错)"""
    try:
        if split == "train":
            file1 = os.path.join(data_dir, "train-00000-of-00002.parquet")
            file2 = os.path.join(data_dir, "train-00001-of-00002.parquet")
            if not os.path.exists(file1) or not os.path.exists(file2):
                raise FileNotFoundError(f"Train parquet files not found in {data_dir}")
            df1 = pd.read_parquet(file1)
            df2 = pd.read_parquet(file2)
            df = pd.concat([df1, df2], ignore_index=True)
        else:
            val_file = os.path.join(data_dir, "validation-00000-of-00001.parquet")
            if not os.path.exists(val_file):
                raise FileNotFoundError(f"Validation parquet file not found in {data_dir}")
            df = pd.read_parquet(val_file)
        
        # 样本采样（更安全）
        if sample and sample > 0 and sample < len(df):
            if use_head:
                df = df.head(sample)
            else:
                df = df.sample(n=sample, random_state=42)
        
        print(f"[Data] Loaded {split}: {len(df)} samples")
        return df
    except Exception as e:
        print(f"[Error] Failed to load parquet data: {e}")
        sys.exit(1)

def _prepare_doc_row(row_data: Tuple[int, Dict]) -> Dict:
    """处理单行数据（用于多进程）"""
    idx, row = row_data
    from collections import defaultdict
    
    doc_id = str(row.get('id', idx))
    question = str(row.get('question', '')).strip()
    answer = str(row.get('answer', '')).strip()
    q_type = str(row.get('type', '')).strip()
    level = str(row.get('level', '')).strip()
    
    ctx = row.get('context', {})
    titles = []
    sentences = []
    if isinstance(ctx, dict):
        raw_titles = ctx.get('title', [])
        raw_sentences = ctx.get('sentences', [])
        if isinstance(raw_titles, (list, np.ndarray)) and isinstance(raw_sentences, (list, np.ndarray)):
            n = min(len(raw_titles), len(raw_sentences))
            for i in range(n):
                title = str(raw_titles[i]).strip() if i < len(raw_titles) else ""
                sents = raw_sentences[i]
                if isinstance(sents, (list, np.ndarray)):
                    sents_clean = [str(s).strip() for s in sents if isinstance(s, str) and str(s).strip()]
                else:
                    sents_clean = []
                if title and sents_clean:
                    titles.append(title)
                    sentences.append(sents_clean)
    
    supporting_facts = row.get('supporting_facts', {})
    support_map = defaultdict(list)
    _raw_supporting_facts = {}
    if isinstance(supporting_facts, dict):
        sf_titles = supporting_facts.get('title', [])
        sf_sent_ids = supporting_facts.get('sent_id', [])
        _raw_supporting_facts = {
            'title': [str(t) for t in sf_titles],
            'sent_id': [
                int(s) if isinstance(s, (int, float)) and not (isinstance(s, float) and np.isnan(s)) else -1
                for s in sf_sent_ids
            ]
        }
        if isinstance(sf_titles, (list, np.ndarray)) and isinstance(sf_sent_ids, (list, np.ndarray)):
            for i in range(min(len(sf_titles), len(sf_sent_ids))):
                sf_title = str(sf_titles[i]).strip()
                try:
                    sf_sent_idx = int(sf_sent_ids[i])
                except (ValueError, TypeError):
                    sf_sent_idx = -1
                if sf_title and sf_sent_idx >= 0:
                    support_map[sf_title].append(sf_sent_idx)
    
    full_text_parts = []
    relevant_text_parts = []
    supporting_titles = []
    supporting_sentences = []
    
    for title, sents in zip(titles, sentences):
        supporting_titles.append(title)
        supporting_sentences.append(sents)
        
        if title and sents:
            sent_str = ' '.join(sents)
            full_text_parts.append(f"[[ENTITY]] {title}: {sent_str}")
        
        if title in support_map:
            for sent_idx in support_map[title]:
                if 0 <= sent_idx < len(sents):
                    relevant_text_parts.append(f"{title}: {sents[sent_idx]}")
    
    entity_summary = ' '.join([f"[[ENTITY]] {t}" for t in supporting_titles])
    full_text = f"{entity_summary} {' '.join(full_text_parts)}".strip()
    relevant_text = ' '.join(relevant_text_parts).strip()
    
    return {
        "id": doc_id,
        "question": question,
        "answer": answer,
        "type": q_type,
        "level": level,
        "text": full_text,
        "relevant_text": relevant_text,
        "supporting_titles": supporting_titles,
        "supporting_sentences": supporting_sentences,
        "_raw_supporting_facts": _raw_supporting_facts
    }

def prepare_documents(df: pd.DataFrame, n_workers: int = None) -> List[Dict[str, Any]]:
    """Prepare documents from dataframe with multi-processing"""
    import multiprocessing
    
    n_workers = n_workers or max(1, multiprocessing.cpu_count() - 1)
    n = len(df)
    
    if n < 100 or n_workers <= 1:
        docs = []
        for idx, row in tqdm(df.iterrows(), total=n, desc="Preparing documents"):
            docs.append(_prepare_doc_row((idx, dict(row))))
        return docs
    
    print(f"[Data] Multi-processing with {n_workers} workers...")
    rows = list(df.iterrows())
    
    docs = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_prepare_doc_row, row_data): i for i, row_data in enumerate(rows)}
        for future in tqdm(as_completed(futures), total=n, desc="Preparing documents"):
            try:
                doc = future.result()
                docs.append(doc)
            except Exception as e:
                print(f"[Warning] Failed to process row: {e}")
    
    return docs

def _build_subgraph_from_chunk(chunk_file: str) -> int:
    """Worker function: build subgraph from a chunk file and save to file (optimized)"""
    import networkx as nx
    import pickle
    from collections import Counter, defaultdict
    import os
    import traceback
    import time
    
    chunk_idx = int(chunk_file.split('_')[-2]) if '_' in chunk_file else 0
    output_file = chunk_file.replace('.pkl', '_subgraph.pkl')
    
    try:
        print(f"[Worker {os.getpid()}] Processing chunk {chunk_idx}...")
        start_time = time.time()
        
        print(f"[Worker {os.getpid()}] Loading {chunk_file}...")
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
        print(f"[Worker {os.getpid()}] Loaded in {time.time() - start_time:.2f}s")
        
        chunk_entities = chunk_data['entities_list']
        chunk_docs = chunk_data['doc_mapping']
        print(f"[Worker {os.getpid()}] Processing {len(chunk_entities)} docs...")
        
        # 优化：使用字典累积，最后一次性构建图
        nodes = {}  # {ent_text: ent_type}
        edge_counter = Counter()  # {(ent1, ent2): count}
        entity_counter = Counter()
        
        for idx, (doc, entities) in enumerate(zip(chunk_docs, chunk_entities)):
            if idx % 2000 == 0:
                print(f"[Worker {os.getpid()}] Progress: {idx}/{len(chunk_entities)} ({100*idx/len(chunk_entities):.1f}%)")
            
            unique_ents = list(set(entities))
            if not unique_ents:
                continue
            
            # 记录节点
            for ent_text, ent_type in unique_ents:
                nodes[ent_text] = ent_type
                entity_counter[ent_text] += 1
            
            # 累积边（使用 frozenset 保证顺序无关）
            for i in range(len(unique_ents)):
                ent1 = unique_ents[i][0]
                for j in range(i + 1, len(unique_ents)):
                    ent2 = unique_ents[j][0]
                    if ent1 != ent2:
                        edge = tuple(sorted([ent1, ent2]))
                        edge_counter[edge] += 1
        
        print(f"[Worker {os.getpid()}] Final: {len(chunk_entities)}/{len(chunk_entities)} (100.0%)")
        
        # 过滤低权重边（只保留出现>=2次的边）
        min_edge_weight = 2
        filtered_edges = {k: v for k, v in edge_counter.items() if v >= min_edge_weight}
        print(f"[Worker {os.getpid()}] Filtering edges: {len(edge_counter)} -> {len(filtered_edges)} (min_weight={min_edge_weight})")
        print(f"[Worker {os.getpid()}] Building graph from {len(nodes)} nodes, {len(filtered_edges)} edges...")
        
        # 一次性构建图
        G = nx.Graph()
        for ent_text, ent_type in nodes.items():
            G.add_node(ent_text, type=ent_type)
        for (ent1, ent2), weight in filtered_edges.items():
            G.add_edge(ent1, ent2, weight=weight)
        
        print(f"[Worker {os.getpid()}] Saving {G.number_of_nodes()} nodes, {G.number_of_edges()} edges...")
        with open(output_file, 'wb') as f:
            pickle.dump({
                'graph': G,
                'counter': entity_counter,
                'chunk_idx': chunk_idx
            }, f, protocol=4)
        
        elapsed = time.time() - start_time
        print(f"[Worker {os.getpid()}] Completed chunk {chunk_idx} in {elapsed:.1f}s")
        
        return chunk_idx
        
    except Exception as e:
        print(f"[Worker Error] {chunk_file}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0

def _merge_graphs(chunk_files: List[str]) -> Tuple[nx.Graph, Counter]:
    """Merge multiple subgraph files into one with progress display"""
    import networkx as nx
    from collections import Counter

    merged_G = nx.Graph()
    merged_counter = Counter()

    subgraph_files = [cf.replace('.pkl', '_subgraph.pkl') for cf in chunk_files]
    valid_subgraph_files = [sf for sf in sorted(subgraph_files) if os.path.exists(sf)]

    print(f"[Merge] Found {len(valid_subgraph_files)} subgraph files to merge")

    for i, sf in enumerate(valid_subgraph_files):
        chunk_name = os.path.basename(sf).replace('_subgraph.pkl', '')
        print(f"[Merge] [{i+1}/{len(valid_subgraph_files)}] Loading {chunk_name}...")

        with open(sf, 'rb') as f:
            data = pickle.load(f)

        G = data['graph']
        counter = data['counter']

        nodes_count = G.number_of_nodes()
        edges_count = G.number_of_edges()

        for node, attrs in G.nodes(data=True):
            if node not in merged_G:
                merged_G.add_node(node, **attrs)

        for u, v, data_edge in G.edges(data=True):
            if merged_G.has_edge(u, v):
                merged_G[u][v]['weight'] += data_edge.get('weight', 1)
            else:
                merged_G.add_edge(u, v, **data_edge)

        for ent, count in counter.items():
            merged_counter[ent] += count

        print(f"[Merge] [{i+1}/{len(valid_subgraph_files)}] Merged {nodes_count} nodes, {edges_count} edges (total: {merged_G.number_of_nodes()} nodes, {merged_G.number_of_edges()} edges)")

    print(f"[Merge] Complete! Final graph: {merged_G.number_of_nodes()} nodes, {merged_G.number_of_edges()} edges")
    return merged_G, merged_counter

class LocalKBBuilder:
    def __init__(self, kb_dir: str = "kb", split: str = "train", sample: Optional[int] = None, data_dir: str = "dataset", use_head: bool = False):
        self.kb_dir = kb_dir
        self.split = split
        self.data_dir = data_dir
        
        # 创建目录（容错）
        os.makedirs(kb_dir, exist_ok=True)
        print(f"[KB] Initialize Local KB Builder (dir={kb_dir}, split={split})")
        
        # 加载数据
        self.df = load_parquet_data(split, sample, data_dir, use_head)
        self.docs = prepare_documents(self.df)
        print(f"[KB] Prepared {len(self.docs)} valid documents")
        
        # 初始化嵌入模型（增加异常捕获）
        try:
            self.embedding = EmbeddingClient()
            # 测试嵌入模型是否可用
            test_emb = self.embedding.encode(["test"], normalize=True)
            if test_emb is None or len(test_emb) == 0:
                raise ValueError("EmbeddingClient returns empty embeddings")
        except Exception as e:
            print(f"[Error] Failed to initialize EmbeddingClient: {e}")
            sys.exit(1)
        
        # 初始化Spacy（加载失败时降级）
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])  # 禁用parser提升速度
        except OSError:
            print("[Warning] en_core_web_sm not found, downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])

    def build_vector_kb(self):
        """Build FAISS vector index (修复批处理+异常捕获)"""
        print(f"[Build] Vector KB ({len(self.docs)} docs)...")
        
        # 过滤空文本
        valid_docs = [(doc['id'], doc['text']) for doc in self.docs if doc['text'].strip()]
        if not valid_docs:
            print("[Error] No valid text for vector index")
            return
        
        doc_ids, texts = zip(*valid_docs)
        total = len(texts)
        batch_size = 128  # 增大batch size加速
        all_embeddings = []
        
        # 批处理编码（用tqdm显示进度）
        for i in tqdm(range(0, total, batch_size), desc="Encoding embeddings"):
            batch = texts[i:i+batch_size]
            try:
                emb = self.embedding.encode(batch, normalize=True)
                if emb is not None and len(emb) > 0:
                    all_embeddings.append(emb)
                else:
                    print(f"[Warning] Empty embeddings for batch {i//batch_size + 1}")
            except Exception as e:
                print(f"[Warning] Failed to encode batch {i//batch_size + 1}: {e}")
                continue
        
        if not all_embeddings:
            print("[Error] No embeddings generated")
            return
        
        # 合并嵌入向量
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        dim = embeddings.shape[1]
        
        # 构建FAISS索引（支持余弦相似度）
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        # 保存索引和文档映射
        faiss.write_index(index, os.path.join(self.kb_dir, "vector.index"))
        
        # 保存文档ID映射（新增：关联向量索引和文档ID）
        with open(os.path.join(self.kb_dir, "doc_ids.json"), 'w', encoding='utf-8') as f:
            json.dump(doc_ids, f, ensure_ascii=False)
        
        print(f"[Build] Vector KB: {index.ntotal} vectors, dim={dim}")

    def build_keyword_kb(self, n_workers: int = None):
        """Build BM25 keyword index (优化分词+容错)"""
        import multiprocessing
        print("[Build] Keyword KB (BM25)...")
        
        valid_docs = [(doc['id'], doc['text'].strip()) for doc in self.docs if doc['text'].strip()]
        if not valid_docs:
            print("[Error] No valid text for BM25 index")
            return
        
        doc_ids, texts = zip(*valid_docs) if valid_docs else ([], [])
        valid_texts = list(texts) if texts else []
        
        n_workers = n_workers or max(1, multiprocessing.cpu_count() - 1)
        
        if len(valid_texts) >= 1000 and n_workers > 1:
            print(f"[Build] Tokenizing with {n_workers} workers...")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                tokenized = list(tqdm(executor.map(tokenize_text, valid_texts), total=len(valid_texts), desc="Tokenizing for BM25"))
        else:
            tokenized = [tokenize_text(text) for text in tqdm(valid_texts, desc="Tokenizing for BM25")]
        
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi(tokenized)
            
            bm25_data = {
                'model': bm25,
                'doc_ids': list(doc_ids),
                'doc_count': len(tokenized)
            }
            
            with open(os.path.join(self.kb_dir, "bm25.pkl"), 'wb') as f:
                pickle.dump(bm25_data, f, protocol=4)
            
            print(f"[Build] Keyword KB: {len(tokenized)} docs")
        except ImportError:
            print("[Warn] rank_bm25 not installed, run: pip install rank_bm25")
        except Exception as e:
            print(f"[Error] Failed to build BM25 index: {e}")

    def build_graph_kb(self, sample_size: Optional[int] = None):
        """Build entity graph KB (优化性能+容错)"""
        docs_to_process = self.docs[:sample_size] if sample_size else self.docs
        n = len(docs_to_process)
        print(f"[Build] Graph KB (all={len(self.docs)}, process={n})")
        
        G = nx.MultiDiGraph()
        entity_counter = Counter()
        
        def extract_entities_batch(texts: List[str]) -> List[List[tuple]]:
            """批处理提取实体（提升速度）"""
            entities_list = []
            for doc in self.nlp.pipe(texts, batch_size=64):
                entities = []
                for ent in doc.ents:
                    ent_text = ent.text.strip()
                    ent_type = ent.label_
                    if ent_text and len(ent_text) > 1 and ent_text.lower() not in STOPWORDS:
                        entities.append((ent_text, ent_type))
                entities_list.append(entities)
            return entities_list
        
        # 准备待处理文本
        texts_to_process = []
        doc_mapping = []
        for doc in docs_to_process:
            combined = f"{doc['question']} {doc['text']}"[:50000]  # 合理截断（避免内存溢出）
            if combined.strip():
                texts_to_process.append(combined)
                doc_mapping.append(doc)
        
        # 批处理提取实体（分批保存，每10000个样本一个文件）
        print("[Build] Extracting entities with spaCy...")
        
        entities_chunk_size = 10000
        chunk_files = []
        
        # 首先检查是否已有 chunk 文件（支持断点续传）
        chunk_files = []
        for chunk_start in range(0, len(texts_to_process), entities_chunk_size):
            chunk_end = min(chunk_start + entities_chunk_size, len(texts_to_process))
            chunk_file = os.path.join(self.kb_dir, f"entities_chunk_{chunk_start:06d}_{chunk_end:06d}.pkl")
            if os.path.exists(chunk_file):
                chunk_files.append(chunk_file)
            else:
                # 有缺失的 chunk，需要提取
                chunk_texts = texts_to_process[chunk_start:chunk_end]
                chunk_docs = doc_mapping[chunk_start:chunk_end]
                
                print(f"[Chunk] Processing {chunk_start}-{chunk_end}...")
                
                # 处理当前块（带进度条）
                batch_size = 64
                chunk_entities = []
                total_batches = (len(chunk_texts) + batch_size - 1) // batch_size
                for i in tqdm(range(0, len(chunk_texts), batch_size), total=total_batches, desc=f"  Chunk {chunk_start//10000}"):
                    batch = chunk_texts[i:i+batch_size]
                    batch_entities = extract_entities_batch(batch)
                    chunk_entities.extend(batch_entities)
                
                # 保存块
                with open(chunk_file, 'wb') as f:
                    pickle.dump({
                        'entities_list': chunk_entities,
                        'doc_mapping': chunk_docs,
                        'chunk_range': (chunk_start, chunk_end)
                    }, f, protocol=4)
                print(f"[Chunk] Saved {chunk_file} ({len(chunk_entities)} docs)")
                chunk_files.append(chunk_file)
        
        print(f"[Chunk] Total {len(chunk_files)} chunks ready")

        print("[Build] Checking existing subgraphs...")

        subgraph_files = [cf.replace('.pkl', '_subgraph.pkl') for cf in chunk_files]
        existing_subgraphs = [sf for sf in sorted(subgraph_files) if os.path.exists(sf)]
        missing_subgraphs = [sf for sf in sorted(subgraph_files) if not os.path.exists(sf)]

        if existing_subgraphs:
            print(f"[Build] Found {len(existing_subgraphs)} existing subgraph files, skipping rebuild")

        if missing_subgraphs:
            print(f"[Build] Building {len(missing_subgraphs)} missing subgraphs...")

            for chunk_file in sorted(chunk_files):
                subgraph_file = chunk_file.replace('.pkl', '_subgraph.pkl')
                if os.path.exists(subgraph_file):
                    continue

                chunk_idx = _build_subgraph_from_chunk(chunk_file)
                completed_chunks.append(chunk_idx)

        print(f"[Build] All subgraph files ready ({len([sf for sf in subgraph_files if os.path.exists(sf)])}/{len(subgraph_files)})")

        print("[Build] Merging subgraphs with progress...")
        G, entity_counter = _merge_graphs(chunk_files)
        
        # 清理子图文件（注释掉，保留子图文件）
        # for cf in chunk_files:
        #     subgraph_file = cf.replace('.pkl', '_subgraph.pkl')
        #     if os.path.exists(subgraph_file):
        #         os.remove(subgraph_file)
        
        # 清理内存
        import gc
        gc.collect()
        
        # 保存图和实体统计
        nx.write_graphml(G, os.path.join(self.kb_dir, "graph.graphml"))
        
        all_entities = [
            {"entity": k, "type": G.nodes[k].get("type", "UNKNOWN"), "count": v}
            for k, v in entity_counter.most_common(10000)  # 限制实体数量避免文件过大
        ]
        
        with open(os.path.join(self.kb_dir, "entities.json"), 'w', encoding='utf-8') as f:
            json.dump(all_entities, f, ensure_ascii=False, indent=2)
        
        print(f"[Build] Graph KB: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(all_entities)} top entities")

    def save_metadata(self):
        """Save metadata and test data (优化格式+容错)"""
        meta = {
            "source": "local_parquet",
            "split": self.split,
            "data_dir": self.data_dir,
            "num_samples_total": len(self.df),
            "num_samples_valid": len(self.docs),
            "fields": ["id", "question", "answer", "type", "level", "text", "relevant_text", "supporting_titles", "supporting_sentences"],
            "files": {
                "docs": "docs.json",
                "doc_ids": "doc_ids.json",
                "vector_index": "vector.index",
                "bm25": "bm25.pkl",
                "graph": "graph.graphml",
                "entities": "entities.json",
                "test_data": "test_data.json"
            },
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        # 保存元数据
        try:
            with open(os.path.join(self.kb_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[KB] Metadata saved")
        except Exception as e:
            print(f"[Warning] Failed to save metadata: {e}")
        
        # 保存测试数据
        try:
            test_data = []
            for doc in self.docs:
                if not doc["question"].strip():
                    continue
                
                # 提取supporting_facts格式: List[Tuple[str, int]]
                # 原始格式: dict with 'title' and 'sent_id' arrays
                sf_list = []
                raw_sf = doc.get('_raw_supporting_facts', {})
                if isinstance(raw_sf, dict):
                    titles = raw_sf.get('title', [])
                    sent_ids = raw_sf.get('sent_id', [])
                    for i in range(min(len(titles), len(sent_ids))):
                        try:
                            sf_list.append((str(titles[i]), int(sent_ids[i])))
                        except (ValueError, TypeError):
                            pass
                
                test_data.append({
                    "id": doc["id"],
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "type": doc["type"],
                    "level": doc["level"],
                    "supporting_facts": sf_list
                })
            
            with open(os.path.join(self.kb_dir, "test_data.json"), 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"[KB] Test data saved ({len(test_data)} valid samples)")
        except Exception as e:
            print(f"[Warning] Failed to save test data: {e}")
        
        # 保存完整文档
        try:
            with open(os.path.join(self.kb_dir, "docs.json"), 'w', encoding='utf-8') as f:
                json.dump(self.docs, f, ensure_ascii=False, indent=2)
            print(f"[KB] Full documents saved")
        except Exception as e:
            print(f"[Warning] Failed to save docs: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build Knowledge Base from Local Data (Fixed Version)')
    parser.add_argument('--kb', type=str, default='kb', help='Knowledge base directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation'], help='Dataset split')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Parquet dataset directory')
    parser.add_argument('--vector', action='store_true', help='Build vector index')
    parser.add_argument('--keyword', action='store_true', help='Build keyword index')
    parser.add_argument('--graph', action='store_true', help='Build graph index')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for data loading')
    parser.add_argument('--graph_sample', type=int, default=None, help='Sample size for graph construction')
    parser.add_argument('--use_head', action='store_true', help='Use first N samples instead of random sampling')
    args = parser.parse_args()
    
    # 初始化构建器
    builder = LocalKBBuilder(args.kb, args.split, args.sample, args.data_dir, args.use_head)
    
    # 构建索引（默认全构建）
    build_all = not any([args.vector, args.keyword, args.graph])
    if args.vector or build_all:
        builder.build_vector_kb()
    
    if args.keyword or build_all:
        builder.build_keyword_kb()
    
    if args.graph or build_all:
        builder.build_graph_kb(args.graph_sample)
    
    # 保存元数据
    builder.save_metadata()
    print("\n[KB] Knowledge base build complete!")

if __name__ == "__main__":
    main()