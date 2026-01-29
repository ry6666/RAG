#!/usr/bin/env python3
"""
Knowledge Base Evaluation Script

Usage:
    python eval_kb.py                    # Quick demo (5 samples from train)
    python eval_kb.py --demo             # Demo mode
    python eval_kb.py --eval             # Full evaluation on test set
    python eval_kb.py --eval --samples 10 # Evaluate on 10 samples
    python eval_kb.py --generation       # Evaluate answer generation (requires Ollama)
    python eval_kb.py --validation       # Use validation set (first 10 samples)
    python eval_kb.py --validation --eval # Full evaluation on validation set
"""

import os
import sys
import json
import time
import re
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, DefaultDict
from difflib import SequenceMatcher
from collections import defaultdict

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.retrieval.multi_channel_retrieval import KBLoader, MultiHopReasoner, Reranker


def load_from_parquet(parquet_path: str, num: int = None) -> List[Dict]:
    """从parquet文件加载测试样本"""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"[Error] Parquet file not found: {parquet_path}")
    
    print(f"[Data] Loading from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    test_data = []
    seen_ids = set()
    
    for idx, row in df.iterrows():
        if num and len(test_data) >= num:
            break
        
        question = str(row.get('question', '')).strip() if pd.notna(row.get('question')) else ''
        if not question:
            continue
            
        answer = str(row.get('answer', '')).strip() if pd.notna(row.get('answer')) else ''
        if not answer:
            continue
            
        sf_raw = row.get('_raw_supporting_facts', {})
        supporting_facts = []
        
        if isinstance(sf_raw, dict):
            sf_titles = sf_raw.get('title', [])
            sf_sent_ids = sf_raw.get('sent_id', [])
            if hasattr(sf_titles, 'tolist'):
                sf_titles = sf_titles.tolist()
            if hasattr(sf_sent_ids, 'tolist'):
                sf_sent_ids = sf_sent_ids.tolist()
            supporting_facts = [(str(sf_titles[i]).strip() if i < len(sf_titles) else '', 
                                int(sf_sent_ids[i]) if i < len(sf_sent_ids) else 0) 
                               for i in range(min(len(sf_titles), len(sf_sent_ids)))]
        elif isinstance(sf_raw, list):
            supporting_facts = sf_raw
        
        q_id = str(row.get('id', f'val_{idx}'))
        if q_id in seen_ids:
            q_id = f"{q_id}_{idx}"
        seen_ids.add(q_id)
        
        test_data.append({
            'id': q_id,
            'question': question,
            'answer': answer,
            'type': str(row.get('type', 'bridge')).strip(),
            'level': str(row.get('level', 'medium')).strip(),
            'supporting_facts': supporting_facts,
            'doc_id': q_id
        })
    
    print(f"[Data] Loaded {len(test_data)} valid samples from parquet")
    return test_data


def load_from_json(json_path: str, num: int = None) -> List[Dict]:
    """从JSON文件加载测试样本"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[Error] JSON file not found: {json_path}")
    
    print(f"[Data] Loading from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    valid_data = []
    for doc in test_data:
        if num and len(valid_data) >= num:
            break
        question = str(doc.get('question', '')).strip()
        answer = str(doc.get('answer', '')).strip()
        if not question or not answer:
            continue
        valid_data.append(doc)
    
    print(f"[Data] Loaded {len(valid_data)} valid samples from JSON")
    return valid_data


def parse_supporting_facts(supporting_facts) -> List[str]:
    """解析 supporting_facts，统一提取标题列表"""
    if not supporting_facts:
        return []
    
    supporting_titles = []
    
    if isinstance(supporting_facts, dict):
        sf_titles = supporting_facts.get('title', [])
        if hasattr(sf_titles, 'tolist'):
            sf_titles = sf_titles.tolist()
        supporting_titles = [str(t).strip() for t in sf_titles if t and str(t).strip()]
    elif isinstance(supporting_facts, (list, tuple)):
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                title = str(item[0]).strip()
                if title:
                    supporting_titles.append(title)
            elif isinstance(item, dict):
                title = str(item.get('title', '')).strip()
                if title:
                    supporting_titles.append(title)
            elif isinstance(item, str) and item.strip():
                supporting_titles.append(item.strip())
    
    # 去重并过滤空值
    return list(filter(None, list(set(supporting_titles))))


def normalize_text(text: str) -> str:
    """统一标准化文本（问题/答案/标题），全局唯一规则"""
    if not text or not isinstance(text, str):
        return ""
    # 转小写、去多余空格、移除特殊符号
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


# 统一标准化别名，保持向下兼容
normalize_answer = normalize_text
normalize_title = normalize_text


def load_test_samples(kb_dir: str, num: int = None, use_validation: bool = False, 
                      data_dir: str = "dataset", validation_file: str = None) -> List[Dict]:
    """加载测试样本"""
    if use_validation or validation_file:
        if validation_file:
            validation_path = validation_file
        else:
            validation_path = os.path.join(data_dir, "validation-00000-of-00001.parquet")
        
        if not os.path.exists(validation_path):
            raise FileNotFoundError(f"[Error] Validation file not found: {validation_path}")
        
        if validation_path.endswith('.json'):
            return load_from_json(validation_path, num)
        else:
            return load_from_parquet(validation_path, num)
    
    test_data_path = f"{kb_dir}/test_data.json"
    
    if os.path.exists(test_data_path):
        print(f"[Data] Loading from {test_data_path}...")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        valid_data = []
        for doc in test_data:
            question = str(doc.get('question', '')).strip()
            answer = str(doc.get('answer', '')).strip()
            if not question or not answer:
                continue
            doc['supporting_facts'] = parse_supporting_facts(doc.get('supporting_facts', {}))
            valid_data.append(doc)
            if num and len(valid_data) >= num:
                break
        
        print(f"[Data] Loaded {len(valid_data)} valid samples from test_data.json")
        return valid_data
    
    docs_path = f"{kb_dir}/docs.json"
    if os.path.exists(docs_path):
        print(f"[Warning] {test_data_path} not found, generating from docs.json...")
        with open(docs_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        test_data = []
        for doc in docs:
            question = str(doc.get('question', '')).strip()
            answer = str(doc.get('answer', '')).strip()
            if not question or not answer:
                continue
            
            test_data.append({
                'id': doc.get('id', str(len(test_data))),
                'question': question,
                'answer': answer,
                'type': doc.get('type', 'bridge').strip(),
                'level': doc.get('level', 'medium').strip(),
                'supporting_facts': parse_supporting_facts(doc.get('supporting_facts', {})),
                'doc_id': doc.get('id', str(len(test_data)))
            })
            if num and len(test_data) >= num:
                break
        
        print(f"[Info] Generated {len(test_data)} valid test samples from docs.json")
        return test_data
    
    raise FileNotFoundError(f"[Error] No test data found in {kb_dir}")


def calculate_dcg(relevant_ranks: List[int], k: int) -> float:
    """计算 DCG (Discounted Cumulative Gain) - 标准实现"""
    dcg = 0.0
    for rank in relevant_ranks:
        if 1 <= rank <= k:
            dcg += 1.0 / np.log2(rank + 1)
    return dcg


def calculate_idcg(k: int, rel_count: int = 1) -> float:
    """计算理想 DCG - 基于实际相关文档数"""
    idcg = 0.0
    for i in range(min(k, rel_count)):
        idcg += 1.0 / np.log2((i + 1) + 1)
    return idcg if idcg > 0 else 1e-6


def calculate_ap(relevant_rank: int, k: int) -> float:
    """计算 Average Precision (AP) - 标准实现，范围[0,1]"""
    if relevant_rank <= 0 or relevant_rank > k:
        return 0.0
    return 1.0 / relevant_rank


def validate_metrics(total_samples: int, hit_count: int, metric_name: str):
    """指标校验 - 杜绝计数超过样本数"""
    if hit_count > total_samples:
        print(f"[Warning] {metric_name} count {hit_count} > total samples {total_samples}, auto corrected to {total_samples}")
        return total_samples
    return hit_count


def evaluate_retrieval(kb_loader: KBLoader, test_data: List[Dict], 
                       top_k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    评估检索效果 - 修复版
    核心修正：指标计算标准化、数据一致性校验、实体过滤优化、图谱检索关联
    计算指标：Recall, Precision, F1, MAP, NDCG, MRR, Hit Rate
    """
    from src.retrieval.multi_channel_retrieval import (
        MultiChannelRetriever, Reranker, EntityExtractor
    )
    
    reasoner = MultiHopReasoner(kb_loader)
    retriever = reasoner.retriever
    reranker = reasoner.reranker
    entity_extractor = reasoner.entity_extractor
    
    max_k = max(top_k_values)
    total_test_samples = len(test_data)
    
    # 初始化统计 - 按类型分类
    type_stats = {
        "bridge": {"total": 0, "hits": {k: 0 for k in top_k_values}, 
                   "mrr": 0.0, "mrr_count": 0,
                   "map_scores": {k: 0.0 for k in top_k_values},
                   "ndcg_scores": {k: 0.0 for k in top_k_values},
                   "retrieval_times": []}, 
        "comparison": {"total": 0, "hits": {k: 0 for k in top_k_values}, 
                       "mrr": 0.0, "mrr_count": 0,
                       "map_scores": {k: 0.0 for k in top_k_values},
                       "ndcg_scores": {k: 0.0 for k in top_k_values},
                       "retrieval_times": []}
    }
    
    recall_details = {"bridge": [], "comparison": []}
    
    # 单检索器指标
    single_retriever_metrics = {
        "vector": {"hits": {k: 0 for k in top_k_values}, "mrr": 0.0, "mrr_count": 0,
                   "map_scores": {k: 0.0 for k in top_k_values}, "retrieval_times": []},
        "bm25": {"hits": {k: 0 for k in top_k_values}, "mrr": 0.0, "mrr_count": 0,
                 "map_scores": {k: 0.0 for k in top_k_values}, "retrieval_times": []},
        "graph": {"hits": {k: 0 for k in top_k_values}, "mrr": 0.0, "mrr_count": 0,
                  "map_scores": {k: 0.0 for k in top_k_values}, "retrieval_times": []}
    }
    
    # 性能拆解 - 正确初始化
    performance_breakdown = {
        "overall": {"entity_extract": 0.0, "vector_retrieve": 0.0, "bm25_retrieve": 0.0,
                    "graph_retrieve": 0.0, "fusion": 0.0, "rerank": 0.0},
        "bridge": {"entity_extract": 0.0, "vector_retrieve": 0.0, "bm25_retrieve": 0.0,
                   "graph_retrieve": 0.0, "fusion": 0.0, "rerank": 0.0},
        "comparison": {"entity_extract": 0.0, "vector_retrieve": 0.0, "bm25_retrieve": 0.0,
                       "graph_retrieve": 0.0, "fusion": 0.0, "rerank": 0.0}
    }
    
    # 知识库覆盖度
    kb_coverage = {
        "overall": {"entity_in_kb_count": 0, "answer_in_kb_count": 0, "total": 0},
        "bridge": {"entity_in_kb_count": 0, "answer_in_kb_count": 0, "total": 0},
        "comparison": {"entity_in_kb_count": 0, "answer_in_kb_count": 0, "total": 0}
    }
    
    print(f"\nEvaluating {total_test_samples} samples (max_k={max_k})...")
    
    # 文档缓存 - 提升性能
    doc_cache = {}
    def get_doc(doc_id):
        if doc_id not in doc_cache:
            doc_cache[doc_id] = kb_loader.get_doc_by_id(doc_id) or {}
        return doc_cache[doc_id]
    
    # 预加载知识库实体映射 - 提升实体校验速度
    kb_entity_set = set()
    for doc in kb_loader.docs:
        doc_text = normalize_text(doc.get('text', ''))
        for word in doc_text.split():
            if len(word) > 2:  # 过滤短词
                kb_entity_set.add(word)
    
    for i, item in enumerate(test_data):
        q_id = item['id']
        question = item['question']
        q_type = item.get('type', 'bridge') if item.get('type', 'bridge') in ['bridge', 'comparison'] else 'bridge'
        level = item.get('level', 'medium')
        
        # 样本数统计
        type_stats[q_type]["total"] += 1
        kb_coverage[q_type]["total"] += 1
        kb_coverage["overall"]["total"] += 1
        
        # 标准化黄金答案和支持事实
        gold_answer = normalize_answer(item.get('answer', ''))
        supporting_titles = parse_supporting_facts(item.get('supporting_facts', []))
        norm_support_titles = [normalize_title(t) for t in supporting_titles if t]
        
        # 实体抽取 + 核心优化：过滤无效/冗余实体
        raw_entities = entity_extractor.extract_entities(question)
        # 过滤规则：非空、长度>2、去重、仅保留知识库中存在的实体
        extracted_entities = list(filter(
            lambda e: e and len(e.strip()) > 2 and normalize_text(e) in kb_entity_set,
            list(set(raw_entities))  # 去重
        ))[:5]  # 最多保留5个核心实体
        
        # 知识库实体覆盖度
        if extracted_entities:
            kb_coverage[q_type]["entity_in_kb_count"] += 1
            kb_coverage["overall"]["entity_in_kb_count"] += 1
        
        # 知识库答案覆盖度
        answer_in_kb = any(gold_answer in normalize_answer(doc.get('answer', '')) for doc in kb_loader.docs[:1000])
        if answer_in_kb:
            kb_coverage[q_type]["answer_in_kb_count"] += 1
            kb_coverage["overall"]["answer_in_kb_count"] += 1
        
        # 各环节耗时统计
        t_entity = t_vector = t_bm25 = t_graph = t_fusion = t_rerank = 0.0
        
        # 1. 实体抽取耗时（已统计，此处补0保持结构）
        t_entity = 0.0
        
        # 2. 单路检索
        vector_results = bm25_results = graph_results = []
        try:
            t0 = time.time()
            vector_results = retriever.vector_retriever.retrieve(question, max_k * 2)
            t_vector = time.time() - t0
        except Exception as e:
            print(f"[Warning] Vector retrieval error: {e}")
        
        try:
            t0 = time.time()
            bm25_results = retriever.bm25_retriever.retrieve(question, max_k * 2)
            t_bm25 = time.time() - t0
        except Exception as e:
            print(f"[Warning] BM25 retrieval error: {e}")
        
        try:
            t0 = time.time()
            # 图谱检索优化：使用过滤后的核心实体，确保与文本关联
            graph_results = retriever.graph_retriever.retrieve_by_entities(extracted_entities, top_k=max_k * 2)
            t_graph = time.time() - t0
        except Exception as e:
            print(f"[Warning] Graph retrieval error: {e}")
        
        # 3. 三路融合
        try:
            t0 = time.time()
            score_map = retriever._merge_scores(
                vector_results, bm25_results, graph_results,
                retriever.weights, retriever._normalize_scores
            )
            # 融合结果排序 + 去重
            fused_scores = sorted(score_map.items(), key=lambda x: x[1]['fused'], reverse=True)[:max_k]
            fusion_candidates = []
            for doc_id, info in fused_scores:
                doc = info['doc']
                fusion_candidates.append({
                    'doc_id': doc_id,
                    'text': doc.get('text', '')[:500],  # 截断过长文本
                    'score': round(info['fused'], 4),
                    'source': 'fusion'
                })
            t_fusion = time.time() - t0
        except Exception as e:
            print(f"[Warning] Fusion error: {e}")
            fusion_candidates = []
        
        # 4. 重排序
        reranked = []
        try:
            t0 = time.time()
            reranked = reranker.rerank(question, fusion_candidates, max_k) if fusion_candidates else []
            t_rerank = time.time() - t0
        except Exception as e:
            print(f"[Warning] Rerank error: {e}")
            reranked = fusion_candidates
        
        # 总耗时
        total_elapsed = t_entity + t_vector + t_bm25 + t_graph + t_fusion + t_rerank
        type_stats[q_type]["retrieval_times"].append(total_elapsed)
        
        # 性能拆解累加 - 修复版
        perf_update = {
            "entity_extract": t_entity,
            "vector_retrieve": t_vector,
            "bm25_retrieve": t_bm25,
            "graph_retrieve": t_graph,
            "fusion": t_fusion,
            "rerank": t_rerank
        }
        for k, v in perf_update.items():
            performance_breakdown[q_type][k] += v
            performance_breakdown["overall"][k] += v
        
        # 候选结果
        candidates = reranked if reranked else fusion_candidates
        
        # 匹配正确结果 - 标准化逻辑
        relevant_ranks = []
        for rank, c in enumerate(candidates, 1):
            doc = get_doc(c['doc_id'])
            if not doc:
                continue
            doc_text = normalize_title(doc.get('text', ''))
            doc_answer = normalize_answer(doc.get('answer', ''))
            
            # 匹配规则：支持事实标题存在 或 黄金答案存在
            title_match = any(gt in doc_text for gt in norm_support_titles if gt)
            answer_match = gold_answer and gold_answer in doc_answer
            
            if title_match or answer_match:
                relevant_ranks.append(rank)
        
        # 核心排名和正确性
        relevant_ranks = sorted(list(set(relevant_ranks)))  # 去重排序
        rank = relevant_ranks[0] if relevant_ranks else (max_k + 1)
        is_correct = rank <= max_k
        
        # 预测答案
        pred_answer = ""
        if candidates:
            top_doc = get_doc(candidates[0]['doc_id'])
            pred_answer = top_doc.get('answer', '').strip() if top_doc else ""
        
        # 计算类型级指标
        current_type_total = type_stats[q_type]["total"]
        for k in top_k_values:
            # Hit计数校验
            hit = 1 if any(r <= k for r in relevant_ranks) else 0
            type_stats[q_type]["hits"][k] = validate_metrics(
                current_type_total, type_stats[q_type]["hits"][k] + hit,
                f"{q_type}_Hit@{k}"
            )
            
            # MAP@k
            ap = calculate_ap(relevant_ranks[0] if relevant_ranks else 0, k)
            type_stats[q_type]["map_scores"][k] += ap
            
            # NDCG@k - 标准计算
            dcg = calculate_dcg(relevant_ranks, k)
            idcg = calculate_idcg(k, len(relevant_ranks))
            ndcg = min(dcg / idcg, 1.0)  # 确保不超过1
            type_stats[q_type]["ndcg_scores"][k] += ndcg
        
        # MRR计算 - 标准实现，范围[0,1]
        if is_correct and rank <= max_k:
            mrr_score = min(1.0 / rank, 1.0)
            type_stats[q_type]["mrr"] += mrr_score
            type_stats[q_type]["mrr_count"] = validate_metrics(
                current_type_total, type_stats[q_type]["mrr_count"] + 1,
                f"{q_type}_MRR_count"
            )
        
        # 打印进度
        status = "✓" if is_correct else "✗"
        gold_preview = item.get('answer', '')[:30] + "..." if len(item.get('answer', '')) > 30 else item.get('answer', '')
        pred_preview = pred_answer[:30] + "..." if pred_answer and len(pred_answer) > 30 else pred_answer or "N/A"
        rank_str = f" (rank={rank})" if is_correct else ""
        print(f"[{i+1:3d}/{total_test_samples}] {status} | {q_type:10} | Gold: {gold_preview:35} | Pred: {pred_preview:35}{rank_str}")
        
        # 单检索器指标计算
        for retriever_name, results in [("vector", vector_results), ("bm25", bm25_results), ("graph", graph_results)]:
            t0 = time.time()
            rel_ranks = []
            for r_rank, r in enumerate(results[:max_k], 1):
                doc = get_doc(getattr(r, 'doc_id', '')) if hasattr(r, 'doc_id') else {}
                if not doc:
                    continue
                doc_text = normalize_title(doc.get('text', ''))
                doc_answer = normalize_answer(doc.get('answer', ''))
                title_match = any(gt in doc_text for gt in norm_support_titles if gt)
                answer_match = gold_answer and gold_answer in doc_answer
                if title_match or answer_match:
                    rel_ranks.append(r_rank)
            
            # MRR
            if rel_ranks:
                single_retriever_metrics[retriever_name]["mrr"] += min(1.0 / rel_ranks[0], 1.0)
                single_retriever_metrics[retriever_name]["mrr_count"] = validate_metrics(
                    total_test_samples, single_retriever_metrics[retriever_name]["mrr_count"] + 1,
                    f"{retriever_name}_MRR_count"
                )
            
            # Hit@k and MAP@k
            for k in top_k_values:
                hit = 1 if any(r <= k for r in rel_ranks) else 0
                single_retriever_metrics[retriever_name]["hits"][k] = validate_metrics(
                    total_test_samples, single_retriever_metrics[retriever_name]["hits"][k] + hit,
                    f"{retriever_name}_Hit@{k}"
                )
                ap = calculate_ap(rel_ranks[0] if rel_ranks else 0, k)
                single_retriever_metrics[retriever_name]["map_scores"][k] += ap
            
            # 耗时
            single_retriever_metrics[retriever_name]["retrieval_times"].append(time.time() - t0)
        
        # 失效原因判定 - 优化版
        failure_reason = None
        if not is_correct:
            if not extracted_entities:
                failure_reason = "entity_extract_failure"
            else:
                # 检查融合结果中是否有黄金相关内容
                gold_in_fusion = any(
                    any(gt in normalize_title(c.get('text', '')) for gt in norm_support_titles if gt)
                    for c in fusion_candidates
                )
                if not gold_in_fusion:
                    failure_reason = "entity_retrieval_failure"
                else:
                    # 检查重排序是否把正确结果排到后面
                    gold_in_reranked = any(
                        any(gt in normalize_title(c.get('text', '')) for gt in norm_support_titles if gt)
                        for c in reranked[:max_k]
                    )
                    if not gold_in_reranked:
                        failure_reason = "ranking_failure"
                    else:
                        failure_reason = "fusion_failure"
        
        # Top3候选结果处理
        top3_candidates = []
        for c in candidates[:3]:
            text = c.get('text', '')
            top3_candidates.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "score": round(c.get('final_score', c.get('score', 0)), 4),
                "source": c.get('source', 'unknown')
            })
        
        # 保存详细结果 - 修复q_id映射
        recall_details[q_type].append({
            'id': q_id,  # 修复：原代码用q_id，保存为id
            'question': question,
            'gold_answer': item.get('answer', ''),
            'pred_answer': pred_answer,
            'supporting_titles': supporting_titles,
            'is_correct': is_correct,
            'rank': rank if is_correct else None,
            'relevant_ranks': relevant_ranks[:5],
            'retrieval_time': total_elapsed,
            'extracted_entities': extracted_entities,
            'top3_candidates': top3_candidates,
            'failure_reason': failure_reason
        })
    
    # 计算全局指标
    overall_stats = {"total": 0, "hits": {k: 0 for k in top_k_values}, 
                     "mrr": 0.0, "mrr_count": 0,
                     "map_scores": {k: 0.0 for k in top_k_values},
                     "ndcg_scores": {k: 0.0 for k in top_k_values},
                     "retrieval_times": []}
    
    for q_type in ["bridge", "comparison"]:
        stats = type_stats[q_type]
        total = stats["total"]
        if total == 0:
            continue
        
        overall_stats["total"] += total
        for k in top_k_values:
            overall_stats["hits"][k] += stats["hits"][k]
            overall_stats["map_scores"][k] += stats["map_scores"][k]
            overall_stats["ndcg_scores"][k] += stats["ndcg_scores"][k]
        overall_stats["mrr"] += stats["mrr"]
        overall_stats["mrr_count"] += stats["mrr_count"]
        overall_stats["retrieval_times"].extend(stats["retrieval_times"])
    
    # 打印类型级结果
    print("\n" + "="*80)
    print("RETRIEVAL EVALUATION RESULTS (BY QUESTION TYPE)")
    print("="*80)
    for q_type in ["bridge", "comparison"]:
        stats = type_stats[q_type]
        total = stats["total"]
        if total == 0:
            continue
        
        mrr = stats["mrr"] / stats["mrr_count"] if stats["mrr_count"] > 0 else 0.0
        avg_time = sum(stats["retrieval_times"]) / total * 1000 if total > 0 else 0
        
        print(f"\n[{q_type.upper()} QUESTIONS] ({total} samples)")
        print("-" * 70)
        print(f"[Core Metrics]")
        for k in top_k_values:
            hits = stats["hits"][k]
            recall = (hits / total) * 100 if total > 0 else 0
            precision = (hits / min(k, total)) * 100 if total > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            map_k = (stats["map_scores"][k] / total) * 100 if total > 0 else 0
            ndcg = (stats["ndcg_scores"][k] / total) * 100 if total > 0 else 0
            print(f"  Top-{k}: Recall={recall:.1f}% | Precision={precision:.1f}% | F1={f1:.1f}%")
            print(f"         MAP@{k}={map_k:.1f}% | NDCG@{k}={ndcg:.1f}%")
        
        print(f"\n[Ranking Metrics]")
        print(f"  MRR: {mrr*100:.1f}% (based on {stats['mrr_count']} valid samples)")
        hit_5 = (stats['hits'][5] / total) * 100 if total > 0 else 0
        hit_10 = (stats['hits'][10] / total) * 100 if total > 0 else 0
        print(f"  Hit@5: {hit_5:.1f}% | Hit@10: {hit_10:.1f}%")
        
        print(f"\n[Performance]")
        print(f"  Avg Retrieval Time: {avg_time:.1f}ms")
    
    # 打印全局结果
    print("\n" + "="*80)
    print("OVERALL RETRIEVAL RESULTS")
    print("="*80)
    total = overall_stats["total"]
    if total > 0:
        mrr = overall_stats["mrr"] / overall_stats["mrr_count"] if overall_stats["mrr_count"] > 0 else 0.0
        avg_time = sum(overall_stats["retrieval_times"]) / total * 1000 if total > 0 else 0
        
        print(f"Total Valid Samples: {total}")
        print(f"\n[Core Metrics]")
        for k in top_k_values:
            hits = overall_stats["hits"][k]
            recall = (hits / total) * 100
            precision = (hits / min(k, total)) * 100
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            map_k = (overall_stats["map_scores"][k] / total) * 100
            ndcg = (overall_stats["ndcg_scores"][k] / total) * 100
            print(f"  Top-{k}: Recall={recall:.1f}% | Precision={precision:.1f}% | F1={f1:.1f}%")
            print(f"         MAP@{k}={map_k:.1f}% | NDCG@{k}={ndcg:.1f}%")
        
        print(f"\n[Ranking Metrics]")
        print(f"  MRR: {mrr*100:.1f}% (based on {overall_stats['mrr_count']} samples)")
        hit_5 = (overall_stats['hits'][5] / total) * 100 if total > 0 else 0
        hit_10 = (overall_stats['hits'][10] / total) * 100 if total > 0 else 0
        print(f"  Hit@5: {hit_5:.1f}% | Hit@10: {hit_10:.1f}%")
        
        print(f"\n[Performance]")
        print(f"  Avg Retrieval Time: {avg_time:.1f}ms")
    
    # 打印单检索器结果
    print("\n" + "="*80)
    print("SINGLE RETRIEVER METRICS")
    print("="*80)
    for retriever_name in ["vector", "bm25", "graph"]:
        stats = single_retriever_metrics[retriever_name]
        mrr = stats["mrr"] / stats["mrr_count"] if stats["mrr_count"] > 0 else 0.0
        avg_time = sum(stats["retrieval_times"]) / total_test_samples * 1000 if total_test_samples > 0 else 0
        print(f"\n[{retriever_name.upper()}]")
        print(f"  MRR: {mrr*100:.1f}% (based on {stats['mrr_count']} samples)")
        for k in top_k_values:
            hits = stats["hits"][k]
            hit_rate = (hits / total_test_samples) * 100 if total_test_samples > 0 else 0
            map_k = (stats["map_scores"][k] / total_test_samples) * 100 if total_test_samples > 0 else 0
            print(f"  Hit@{k}: {hit_rate:.1f}% | MAP@{k}: {map_k:.1f}%")
        print(f"  Avg Inference Time: {avg_time:.3f}ms")
    
    # 打印性能拆解
    print("\n" + "="*80)
    print("PERFORMANCE BREAKDOWN (ms, %)")
    print("="*80)
    for category in ["overall", "bridge", "comparison"]:
        breakdown = performance_breakdown[category]
        total_time = sum(breakdown.values())
        if total_time < 1e-6:
            continue
        total_time_ms = total_time * 1000
        print(f"\n[{category.upper()}] Total: {total_time_ms:.1f}ms")
        for component, t in breakdown.items():
            t_ms = t * 1000
            pct = (t / total_time) * 100 if total_time > 0 else 0
            print(f"  {component:15}: {t_ms:.1f}ms ({pct:.1f}%)")
    
    # 打印知识库覆盖度
    print("\n" + "="*80)
    print("KNOWLEDGE BASE COVERAGE")
    print("="*80)
    for category in ["overall", "bridge", "comparison"]:
        cov = kb_coverage[category]
        total = cov["total"]
        if total == 0:
            continue
        entity_rate = (cov["entity_in_kb_count"] / total) * 100
        answer_rate = (cov["answer_in_kb_count"] / total) * 100
        print(f"\n[{category.upper()}] ({total} samples)")
        print(f"  Entity in KB: {entity_rate:.1f}% ({cov['entity_in_kb_count']}/{total})")
        print(f"  Answer in KB: {answer_rate:.1f}% ({cov['answer_in_kb_count']}/{total})")
    
    # 标准化返回结果
    return {
        'type_stats': type_stats,
        'recall_details': recall_details,
        'overall': {
            'total': total,
            'hits': {k: overall_stats["hits"][k] for k in top_k_values},
            'mrr': mrr,
            'mrr_count': overall_stats["mrr_count"],
            'map_scores': {k: overall_stats["map_scores"][k] / total if total > 0 else 0.0 for k in top_k_values},
            'ndcg_scores': {k: overall_stats["ndcg_scores"][k] / total if total > 0 else 0.0 for k in top_k_values},
            'avg_retrieval_time': sum(overall_stats["retrieval_times"]) / total if total > 0 else 0.0
        },
        'single_retriever_metrics': single_retriever_metrics,
        'performance_breakdown': performance_breakdown,
        'kb_coverage': kb_coverage
    }


def compute_rouge_l(reference: str, candidate: str) -> float:
    """计算 ROUGE-L (Longest Common Subsequence) - 标准实现"""
    ref_tokens = normalize_text(reference).split()
    cand_tokens = normalize_text(candidate).split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs = dp[m][n]
    return lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0


def compute_bleu4(reference: str, candidate: str) -> float:
    """计算 BLEU-4 - 带平滑，标准实现"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return 0.0
    
    ref = [normalize_text(reference).split()]
    cand = normalize_text(candidate).split()
    
    if not cand or len(cand) < 4:
        return 0.0
    
    smoothie = SmoothingFunction().method4
    return sentence_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)


def compute_f1_score(reference: str, candidate: str) -> float:
    """计算词级别 F1 分数 - 标准实现"""
    ref = set(normalize_text(reference).split())
    cand = set(normalize_text(candidate).split())
    
    if not ref or not cand:
        return 0.0
    
    intersection = ref & cand
    if not intersection:
        return 0.0
    
    precision = len(intersection) / len(cand)
    recall = len(intersection) / len(ref)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(reference: str, candidate: str) -> float:
    """计算 Exact Match (EM) - 标准化后精确匹配"""
    return 1.0 if normalize_text(reference) == normalize_text(candidate) else 0.0


def evaluate_generation(kb_loader: KBLoader, test_data: List[Dict],
                        model_name: str = "qwen2:7b-instruct",
                        top_k: int = 5,
                        max_context_length: int = 4000) -> Dict:
    """评估答案生成质量 - 修复版，增加容错"""
    from src.models.ollama import OllamaClient
    
    try:
        ollama = OllamaClient(model_name)
        print(f"[Generation] Using model: {model_name}")
    except Exception as e:
        print(f"[Error] Ollama initialization failed: {e}")
        return {'error': f"Ollama not available: {str(e)}"}
    
    reasoner = MultiHopReasoner(kb_loader)
    total_samples = len(test_data)
    results = []
    
    # 初始化指标
    exact_match = 0.0
    total_f1 = 0.0
    total_bleu = 0.0
    total_rouge = 0.0
    
    print(f"\n{'='*80}")
    print(f"GENERATION EVALUATION (Model: {model_name})")
    print(f"{'='*80}")
    print(f"Top-K: {top_k} | Max Context: {max_context_length} | Samples: {total_samples}")
    print()
    
    for i, item in enumerate(test_data):
        q_id = item['id']
        question = item['question']
        gold_answer = item.get('answer', '').strip()
        q_type = item.get('type', 'bridge')
        
        # 进度打印
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{total_samples}] Processing: {question[:50]}...")
        else:
            print(f"[{i+1}/{total_samples}] Processing: {question[:50]}...", end='\r')
        
        # 检索上下文
        try:
            search_result = reasoner.search(question, q_type, 'medium', top_k=top_k)
            candidates = search_result.get('candidates', [])[:top_k]
        except Exception as e:
            print(f"[Warning] Search failed for {q_id}: {e}")
            candidates = []
        
        # 构建上下文（截断过长内容）
        contexts = []
        current_len = 0
        for c in candidates:
            doc = kb_loader.get_doc_by_id(c['doc_id']) or {}
            text = doc.get('text', '').strip()
            text_len = len(text)
            if current_len + text_len <= max_context_length and text:
                contexts.append(text)
                current_len += text_len
            elif current_len < max_context_length:
                remaining = max_context_length - current_len
                contexts.append(text[:remaining])
                current_len = max_context_length
            else:
                break
        
        # 生成答案
        pred_answer = ""
        try:
            pred_answer = ollama.generate_answer(question, contexts) if contexts else ollama.generate_answer(question, [])
            pred_answer = pred_answer.strip()
        except Exception as e:
            print(f"[Warning] Generation failed for {q_id}: {e}")
        
        # 计算指标
        em = compute_exact_match(gold_answer, pred_answer)
        f1 = compute_f1_score(gold_answer, pred_answer)
        bleu = compute_bleu4(gold_answer, pred_answer)
        rouge = compute_rouge_l(gold_answer, pred_answer)
        
        # 累加指标
        exact_match += em
        total_f1 += f1
        total_bleu += bleu
        total_rouge += rouge
        
        # 保存详细结果
        results.append({
            'id': q_id,
            'question': question,
            'gold_answer': gold_answer,
            'pred_answer': pred_answer,
            'em': em,
            'f1': f1,
            'bleu4': bleu,
            'rouge_l': rouge,
            'context_count': len(contexts)
        })
        
        # 打印详细结果（每5个样本）
        if (i + 1) % 5 == 0:
            status = "✓" if em == 1.0 else "✗"
            gold_pre = gold_answer[:40] + "..." if len(gold_answer) > 40 else gold_answer
            pred_pre = pred_answer[:40] + "..." if len(pred_answer) > 40 else pred_answer or "N/A"
            print(f"  {status} Gold: {gold_pre}")
            print(f"     Pred: {pred_pre}")
            print(f"     EM={em:.1f} | F1={f1:.2f} | BLEU-4={bleu:.2f} | ROUGE-L={rouge:.2f}")
            print()
    
    # 计算平均指标
    if total_samples == 0:
        avg_metrics = {'avg_f1': 0.0, 'avg_bleu4': 0.0, 'avg_rouge_l': 0.0, 'em_rate': 0.0}
    else:
        avg_metrics = {
            'avg_f1': total_f1 / total_samples,
            'avg_bleu4': total_bleu / total_samples,
            'avg_rouge_l': total_rouge / total_samples,
            'em_rate': (exact_match / total_samples) * 100
        }
    
    # 打印生成结果
    print(f"\n{'='*80}")
    print("GENERATION EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"  Exact Match Rate: {avg_metrics['em_rate']:.1f}% ({exact_match:.0f}/{total_samples})")
    print(f"  Average F1 Score: {avg_metrics['avg_f1']*100:.1f}%")
    print(f"  Average BLEU-4: {avg_metrics['avg_bleu4']*100:.1f}%")
    print(f"  Average ROUGE-L: {avg_metrics['avg_rouge_l']*100:.1f}%")
    
    return {
        'model': model_name,
        'total': total_samples,
        'exact_match': exact_match,
        'em_rate': avg_metrics['em_rate'],
        'avg_f1': avg_metrics['avg_f1'],
        'avg_bleu4': avg_metrics['avg_bleu4'],
        'avg_rouge_l': avg_metrics['avg_rouge_l'],
        'details': results
    }


def demo(kb_loader: KBLoader, test_data: List[Dict]):
    """演示模式 - 展示多跳推理过程"""
    print("\n" + "="*60)
    print("多跳推理检索演示 (Multi-Hop Reasoning Demo)")
    print("="*60)
    
    reasoner = MultiHopReasoner(kb_loader)
    demo_samples = test_data[:5] if len(test_data) >=5 else test_data
    
    print(f"\nDemo with {len(demo_samples)} questions...\n")
    
    for i, item in enumerate(demo_samples):
        q_id = item['id']
        question = item['question']
        q_type = item.get('type', 'bridge')
        level = item.get('level', 'medium')
        gold_answer = item.get('answer', '')
        
        print(f"[Q{i+1}] {question}")
        print(f"  Type: {q_type} | Level: {level} | Gold Answer: {gold_answer}")
        
        # 多跳检索
        try:
            result = reasoner.search(question, q_type, level, top_k=5)
        except Exception as e:
            print(f"  [Error] Search failed: {e}")
            continue
        
        # 打印实体和检索结果
        if q_type == "comparison":
            entities = result.get('entities', [])
            print(f"  Extracted Entities: {entities if entities else 'None'}")
        else:
            first_ents = result.get('first_entities', [])[:3]
            second_queries = result.get('second_queries', [])[:2]
            print(f"  First Hop Entities: {first_ents if first_ents else 'None'}")
            print(f"  Second Hop Queries: {second_queries if second_queries else 'None'}")
        
        # 打印Top5候选
        candidates = result.get('candidates', [])[:5]
        print(f"  Top-5 Retrieved Candidates:")
        for j, c in enumerate(candidates):
            doc_id = c.get('doc_id', 'N/A')[:8]
            score = c.get('rerank_score', c.get('score', 0))
            text = c.get('text', '')[:50] + "..." if len(c.get('text', ''))>50 else c.get('text', '')
            print(f"    [{j+1}] {doc_id} | Score: {score:.3f} | {text}")
        
        print("-" * 80 + "\n")


def save_results(results: Dict, output_dir: str = "results") -> str:
    """保存评估结果到JSON文件 - 修复版，确保字段映射正确"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    
    # 检索配置 - 固定三路检索配置
    retrieval_config = {
        'retriever_types': ['vector', 'bm25', 'graph'],
        'fusion_weights': [0.4, 0.3, 0.3],
        'score_normalize': True,
        'rerank_switch': True,
        'rerank_top_k': 5,
        'entity_extract_switch': True,
        'embedding_model': 'BAAI/bge-base-en-v1.5'
    }
    
    # 初始化优化后的结果结构
    optimized_results = {
        'timestamp': results.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S")),
        'mode': results.get('mode', 'unknown'),
        'data_source': results.get('data_source', ''),
        'num_samples': results.get('num_samples', 0),
        'config': {
            'top_k_values': results.get('config', {}).get('top_k_values', [5,10,20]),
            'model': results.get('config', {}).get('model'),
            'context_length': results.get('config', {}).get('context_length', 4000),
            'retrieval_config': retrieval_config
        },
        'kb_stats': results.get('kb_stats', {}),
        'retrieval_eval': None,
        'generation_eval': None,
        'error_samples': [],
        'summary': {}
    }
    
    # 处理检索评估结果
    if 'retrieval_eval' in results:
        eval_data = results['retrieval_eval']
        overall = eval_data.get('overall', {})

        # 调试：打印overall内容
        import json
        print(f"\n[DEBUG] overall keys: {list(overall.keys())}")
        print(f"[DEBUG] overall['total']: {overall.get('total', 'NOT_FOUND')}")

        # 验证total的一致性
        bridge_total = eval_data.get('type_stats', {}).get('bridge', {}).get('total', 0)
        comp_total = eval_data.get('type_stats', {}).get('comparison', {}).get('total', 0)
        expected_total = bridge_total + comp_total
        actual_total = overall.get('total', 0)
        print(f"[DEBUG] type_stats total: bridge={bridge_total}, comp={comp_total}, sum={expected_total}")
        print(f"[DEBUG] overall['total']: {actual_total}")

        if expected_total != actual_total:
            print(f"[WARNING] Total mismatch! Using expected_total={expected_total} instead of actual_total={actual_total}")
            overall['total'] = expected_total

        type_stats = eval_data.get('type_stats', {})
        recall_details = eval_data.get('recall_details', {})
        single_retriever = eval_data.get('single_retriever_metrics', {})
        performance = eval_data.get('performance_breakdown', {})
        kb_coverage = eval_data.get('kb_coverage', {})
        
        optimized_results['retrieval_eval'] = {
            'total': overall.get('total', 0),
            'hits': overall.get('hits', {}),
            'mrr': overall.get('mrr', 0.0),
            'mrr_count': overall.get('mrr_count', 0),
            'map_scores': overall.get('map_scores', {}),
            'ndcg_scores': overall.get('ndcg_scores', {}),
            'avg_retrieval_time': overall.get('avg_retrieval_time', 0.0),
            'type_stats': {},
            'single_retriever_metrics': {},
            'performance_breakdown': performance,
            'kb_coverage': {},
            'error_samples': []
        }
        
        # 处理类型统计
        for q_type in ['bridge', 'comparison']:
            if q_type not in type_stats:
                continue
            ts = type_stats[q_type]
            total = ts.get('total', 0)
            avg_time = sum(ts.get('retrieval_times', [])) / total if total > 0 else 0.0
            mrr_val = ts.get('mrr', 0.0)
            mrr_count = ts.get('mrr_count', 0)
            optimized_results['retrieval_eval']['type_stats'][q_type] = {
                'total': total,
                'hits': ts.get('hits', {}),
                'mrr': mrr_val / mrr_count if mrr_count > 0 else 0.0,
                'mrr_count': mrr_count,
                'avg_retrieval_time': avg_time
            }
        
        # 处理单检索器指标
        for ret_name in ['vector', 'bm25', 'graph']:
            if ret_name not in single_retriever:
                continue
            rs = single_retriever[ret_name]
            total_t = sum(rs.get('retrieval_times', [0.0]))
            avg_t = total_t / len(rs.get('retrieval_times', [1])) if rs.get('retrieval_times') else 0.0
            mrr_val = rs.get('mrr', 0.0)
            mrr_count = rs.get('mrr_count', 0)
            optimized_results['retrieval_eval']['single_retriever_metrics'][ret_name] = {
                'hit_10': rs.get('hits', {}).get(10, 0),
                'mrr': mrr_val / mrr_count if mrr_count > 0 else 0.0,
                'mrr_count': mrr_count,
                'avg_time': avg_t
            }
        
        # 处理知识库覆盖度
        for cat in ['overall', 'bridge', 'comparison']:
            if cat not in kb_coverage:
                continue
            cov = kb_coverage[cat]
            total = cov.get('total', 1)
            optimized_results['retrieval_eval']['kb_coverage'][cat] = {
                'entity_in_kb_rate': (cov.get('entity_in_kb_count', 0) / total) * 100 if total > 0 else 0.0,
                'answer_in_kb_rate': (cov.get('answer_in_kb_count', 0) / total) * 100 if total > 0 else 0.0
            }
        
        # 收集错误样本 - 修复字段映射（id -> q_id）
        error_samples = []
        for q_type in ['bridge', 'comparison']:
            for detail in recall_details.get(q_type, []):
                if not detail.get('is_correct', False):
                    error_samples.append({
                        'q_id': detail.get('id', ''),  # 修复：从detail['id']获取q_id
                        'question_preview': detail.get('question', '')[:100],
                        'gold_answer_preview': detail.get('gold_answer', '')[:50],
                        'rank': detail.get('rank'),
                        'retrieval_time': round(detail.get('retrieval_time', 0.0), 4),
                        'extracted_entities': detail.get('extracted_entities', []),
                        'top3_candidates': detail.get('top3_candidates', []),
                        'failure_reason': detail.get('failure_reason')
                    })
        optimized_results['retrieval_eval']['error_samples'] = error_samples
        
        # 生成摘要指标
        total = overall.get('total', 0)
        hits_5 = overall.get('hits', {}).get(5, 0)
        hits_10 = overall.get('hits', {}).get(10, 0)

        # 安全计算：recall不能超过100%
        recall_5 = min((hits_5 / total) * 100, 100.0) if total > 0 else 0.0
        recall_10 = min((hits_10 / total) * 100, 100.0) if total > 0 else 0.0

        optimized_results['summary'] = {
            'recall_5': round(recall_5, 2),
            'recall_10': round(recall_10, 2),
            'mrr': round(overall.get('mrr', 0.0), 6)
        }
    
    # 处理生成评估结果
    if 'generation_eval' in results:
        gen_data = results['generation_eval']
        optimized_results['generation_eval'] = {
            'model': gen_data.get('model', ''),
            'total': gen_data.get('total', 0),
            'exact_match': gen_data.get('exact_match', 0.0),
            'em_rate': gen_data.get('em_rate', 0.0),
            'avg_f1': gen_data.get('avg_f1', 0.0),
            'avg_bleu4': gen_data.get('avg_bleu4', 0.0),
            'avg_rouge_l': gen_data.get('avg_rouge_l', 0.0),
            'error_samples': []
        }
        
        # 收集生成错误样本
        gen_error_samples = []
        for detail in gen_data.get('details', []):
            if detail.get('em', 0.0) == 0.0:
                gen_error_samples.append({
                    'q_id': detail.get('id', ''),
                    'question_preview': detail.get('question', '')[:100],
                    'gold_answer_preview': detail.get('gold_answer', '')[:50],
                    'pred_answer_preview': detail.get('pred_answer', '')[:50] if detail.get('pred_answer') else 'N/A',
                    'f1': round(detail.get('f1', 0.0), 4),
                    'bleu4': round(detail.get('bleu4', 0.0), 4)
                })
        optimized_results['generation_eval']['error_samples'] = gen_error_samples
        
        # 更新摘要指标
        optimized_results['summary'].update({
            'em_rate': round(gen_data.get('em_rate', 0.0), 2),
            'avg_f1': round(gen_data.get('avg_f1', 0.0) * 100, 2),
            'avg_bleu4': round(gen_data.get('avg_bleu4', 0.0) * 100, 2),
            'avg_rouge_l': round(gen_data.get('avg_rouge_l', 0.0) * 100, 2)
        })
    
    # 保存JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(optimized_results, f, ensure_ascii=False, indent=2, default=lambda x: str(x))
    
    print(f"\n[Results] Evaluation report saved to: {output_path}")
    print(f"[Summary] Key Metrics: {json.dumps(optimized_results['summary'], indent=2)}")
    return output_path


def analyze_metrics_quality(stats: Dict, total: int, q_type: str) -> Dict:
    """分析指标质量并返回诊断信息 - 基于HotpotQA行业基线"""
    if total == 0:
        return {'issues': ['No samples for this type'], 'suggestions': []}
    
    recall_5 = (stats['hits'][5] / total) * 100
    recall_20 = (stats['hits'][20] / total) * 100 if 20 in stats['hits'] else 0.0
    mrr = stats['mrr'] / stats['mrr_count'] if stats['mrr_count'] > 0 else 0.0
    mrr_pct = mrr * 100
    
    analysis = {
        'recall_5': round(recall_5, 1),
        'recall_20': round(recall_20, 1),
        'mrr': round(mrr_pct, 1),
        'issues': [],
        'suggestions': []
    }
    
    # 基于HotpotQA多跳问答基线的判定标准
    if recall_5 < 20:
        analysis['issues'].append(f"{q_type}: Recall@5 is too low ({recall_5:.1f}%)")
        analysis['suggestions'].append("Optimize first-hop entity extraction and retrieval recall")
    elif 20 <= recall_5 < 50:
        analysis['issues'].append(f"{q_type}: Recall@5 is moderate ({recall_5:.1f}%)")
        analysis['suggestions'].append("Tune fusion weights and reranker model")
    else:
        analysis['issues'].append(f"{q_type}: Recall@5 is good ({recall_5:.1f}%)")
    
    if mrr_pct < 15:
        analysis['issues'].append(f"{q_type}: MRR is too low ({mrr_pct:.1f}%)")
        analysis['suggestions'].append("Optimize ranking model to push correct answers higher")
    elif 15 <= mrr_pct < 40:
        analysis['issues'].append(f"{q_type}: MRR is moderate ({mrr_pct:.1f}%)")
        analysis['suggestions'].append("Improve candidate reranking strategy")
    
    if 10 in stats['hits'] and stats['hits'][5] == stats['hits'][10]:
        analysis['issues'].append(f"{q_type}: No improvement from Top-5 to Top-10")
        analysis['suggestions'].append("Check retrieval pipeline for 6-10 rank candidate quality")
    
    return analysis


def validate_calculation(type_stats: Dict, test_data: List[Dict]) -> Dict:
    """验证指标计算过程是否正确 - 数据一致性校验"""
    total_samples = len(test_data)
    type_total = type_stats['bridge']['total'] + type_stats['comparison']['total']
    
    validation = {
        'total_samples': total_samples,
        'type_aggregated_total': type_total,
        'matching_logic': 'answer match OR supporting facts title match',
        'potential_issues': [],
        'checks': []
    }
    
    # 样本数校验
    if type_total != total_samples:
        validation['potential_issues'].append(f"Type total ({type_total}) != actual total ({total_samples})")
        validation['checks'].append({
            'name': 'Sample Count Match',
            'status': '✗',
            'detail': f"Type sum {type_total} != test data {total_samples}"
        })
    else:
        validation['checks'].append({
            'name': 'Sample Count Match',
            'status': '✓',
            'detail': f"Type sum {type_total} = test data {total_samples}"
        })
    
    # Hit计数校验
    total_hit_5 = type_stats['bridge']['hits'][5] + type_stats['comparison']['hits'][5]
    if total_hit_5 > total_samples:
        validation['potential_issues'].append(f"Total Hit@5 ({total_hit_5}) > total samples ({total_samples})")
    validation['checks'].append({
        'name': 'Hit@5 Count Validation',
        'status': '✓' if total_hit_5 <= total_samples else '✗',
        'detail': f"Total Hit@5 {total_hit_5} ≤ {total_samples}"
    })
    
    # MRR计数校验
    total_mrr_count = type_stats['bridge']['mrr_count'] + type_stats['comparison']['mrr_count']
    if total_mrr_count > total_samples:
        validation['potential_issues'].append(f"Total MRR count ({total_mrr_count}) > total samples ({total_samples})")
    validation['checks'].append({
        'name': 'MRR Count Validation',
        'status': '✓' if total_mrr_count <= total_samples else '✗',
        'detail': f"Total MRR count {total_mrr_count} ≤ {total_samples}"
    })
    
    # 空答案样本校验
    empty_answer = sum(1 for item in test_data if not item.get('answer', '').strip())
    validation['checks'].append({
        'name': 'Empty Gold Answer',
        'status': '✓' if empty_answer == 0 else '⚠',
        'detail': f"{empty_answer}/{total_samples} samples have empty gold answer"
    })
    
    return validation


def print_metric_analysis(all_stats: Dict, test_data: List[Dict]):
    """打印指标质量分析和计算验证"""
    print("\n" + "="*80)
    print("METRIC QUALITY ANALYSIS & VALIDATION")
    print("="*80)
    
    # 计算验证
    validation = validate_calculation(all_stats['type_stats'], test_data)
    print("\n[Calculation Validation]")
    for check in validation['checks']:
        print(f"  {check['status']} {check['name']}: {check['detail']}")
    
    if validation['potential_issues']:
        print("\n[Potential Calculation Issues]")
        for issue in validation['potential_issues']:
            print(f"  ⚠ {issue}")
    
    # 指标质量评估
    print("\n[Metric Quality Assessment (vs HotpotQA Baseline)]")
    for q_type in ["bridge", "comparison"]:
        stats = all_stats['type_stats'][q_type]
        total = stats['total']
        if total == 0:
            continue
        
        analysis = analyze_metrics_quality(stats, total, q_type)
        print(f"\n  [{q_type.upper()} - {total} samples]")
        print(f"    Recall@5:  {analysis['recall_5']:.1f}% | Recall@20: {analysis['recall_20']:.1f}% | MRR: {analysis['mrr']:.1f}%")
        
        if analysis['issues']:
            print(f"    [Key Issues]")
            for issue in analysis['issues']:
                print(f"      • {issue}")
        if analysis['suggestions']:
            print(f"    [Optimization Suggestions]")
            for sugg in analysis['suggestions']:
                print(f"      → {sugg}")


def main():
    """主函数 - 解析参数并执行对应模式"""
    parser = argparse.ArgumentParser(
        description='Knowledge Base Evaluation for Multi-Hop Reasoning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (5 samples from train)
  python eval_kb.py --demo
  
  # Evaluate retrieval on 10 validation samples
  python eval_kb.py --validation --eval --samples 10
  
  # Evaluate retrieval on 100 train samples
  python eval_kb.py --eval --samples 100
  
  # Evaluate answer generation (requires Ollama)
  python eval_kb.py --generation --samples 5
  
  # Full evaluation with custom parameters
  python eval_kb.py --validation --eval --samples 100 --top-k 5,10,20 --model qwen2:7b-instruct
  
  # Custom fusion weights and reranking
  python eval_kb.py --eval --samples 50 --weights 0.4,0.3,0.3 --rerank-top-k 5 --graph-top-k 20
        """
    )
    
    # 模式选择
    parser.add_argument('--demo', action='store_true', help='Demo mode (5 samples from train)')
    parser.add_argument('--eval', action='store_true', help='Evaluate retrieval performance')
    parser.add_argument('--generation', action='store_true', help='Evaluate answer generation (requires Ollama)')
    
    # 数据参数
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to evaluate (default: all)')
    parser.add_argument('--kb_dir', type=str, default='kb', help='Knowledge base directory (default: kb)')
    parser.add_argument('--validation', action='store_true', help='Use validation set instead of train set')
    parser.add_argument('--validation_file', type=str, default=None, help='Custom validation file path')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory (default: dataset)')
    parser.add_argument('--output_dir', type=str, default='test/results', help='Output directory (default: test/results)')
    
    # 评估参数
    parser.add_argument('--top_k', type=str, default='5,10,20', help='Comma-separated top-k values (default: 5,10,20)')
    parser.add_argument('--model', type=str, default='qwen2:7b-instruct', help='Ollama model for generation')
    parser.add_argument('--context_len', type=int, default=4000, help='Max context length for generation')
    
    # 检索配置
    parser.add_argument('--weights', type=str, default='0.4,0.3,0.3', help='Fusion weights (vector,bm25,graph)')
    parser.add_argument('--rerank_top_k', type=int, default=5, help='Rerank top-k candidates')
    parser.add_argument('--graph_top_k', type=int, default=20, help='Graph retrieval top-k')
    parser.add_argument('--batch_size', type=int, default=10, help='Evaluation batch size')
    
    # 日志和种子
    parser.add_argument('--quiet', action='store_true', help='Reduce logging verbosity')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 解析参数
    top_k_values = sorted([int(k) for k in args.top_k.split(',') if k.strip().isdigit()])
    fusion_weights = [float(w) for w in args.weights.split(',') if w.strip().replace('.', '').isdigit()]
    # 确保融合权重为3个
    if len(fusion_weights) != 3:
        fusion_weights = [0.4, 0.3, 0.3]
        print(f"[Warning] Invalid fusion weights, use default: {fusion_weights}")
    
    # 设置随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 打印基础信息
    print("="*70)
    print("Knowledge Base Evaluation for Multi-Hop Reasoning")
    print("="*70)
    run_mode = 'demo' if args.demo else ('generation' if args.generation else 'retrieval_evaluation')
    print(f"Run Mode: {run_mode}")
    print(f"Data Source: {'Validation Set' if args.validation else 'Train Set'}")
    print(f"KB Directory: {args.kb_dir} | Output Directory: {args.output_dir}")
    print(f"Top-K Values: {top_k_values} | Fusion Weights: {fusion_weights}")
    if args.verbose:
        print(f"Rerank Top-K: {args.rerank_top_k} | Graph Top-K: {args.graph_top_k}")
        print(f"Batch Size: {args.batch_size} | Random Seed: {args.seed}")
    print("="*70)
    
    # 加载知识库
    try:
        kb_loader = KBLoader(args.kb_dir)
        print(f"[KB] Loaded successfully - Docs: {len(kb_loader.docs)}, KG Nodes: {kb_loader.kg.number_of_nodes()}, KG Edges: {kb_loader.kg.number_of_edges()}")
    except FileNotFoundError as e:
        print(f"[Fatal Error] KB directory not found: {e}")
        print(f"Please build the knowledge base first before evaluation.")
        sys.exit(1)
    except Exception as e:
        print(f"[Fatal Error] Failed to load KB: {e}")
        sys.exit(1)
    
    # 加载测试样本
    try:
        test_data = load_test_samples(args.kb_dir, args.samples, args.validation, args.data_dir, args.validation_file)
        if not test_data:
            print(f"[Fatal Error] No valid test samples loaded")
            sys.exit(1)
        print(f"[Data] Loaded {len(test_data)} valid test samples")
    except Exception as e:
        print(f"[Fatal Error] Failed to load test data: {e}")
        sys.exit(1)
    
    # 初始化结果容器
    all_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'data_source': 'validation-00000-of-00001.parquet' if args.validation else 'train docs.json',
        'num_samples': len(test_data),
        'kb_stats': {
            'docs': len(kb_loader.docs),
            'vector_dim': getattr(kb_loader.vector_index, 'd', 768),
            'graph_nodes': kb_loader.kg.number_of_nodes(),
            'graph_edges': kb_loader.kg.number_of_edges()
        },
        'config': {
            'top_k_values': top_k_values,
            'model': args.model if args.generation else None,
            'context_length': args.context_len
        }
    }
    
    # 执行对应模式
    if args.demo or not (args.eval or args.generation):
        all_results['mode'] = 'demo'
        demo(kb_loader, test_data)
        save_results(all_results, args.output_dir)
    elif args.eval:
        all_results['mode'] = 'retrieval_eval'
        # 执行检索评估
        eval_results = evaluate_retrieval(kb_loader, test_data, top_k_values)
        all_results['retrieval_eval'] = eval_results
        # 打印指标分析
        print_metric_analysis(eval_results, test_data)
        # 保存结果
        save_results(all_results, args.output_dir)
    elif args.generation:
        all_results['mode'] = 'generation_eval'
        # 执行生成评估
        gen_results = evaluate_generation(
            kb_loader, test_data,
            model_name=args.model,
            top_k=top_k_values[0] if top_k_values else 5,
            max_context_length=args.context_len
        )
        all_results['generation_eval'] = gen_results
        # 保存结果
        save_results(all_results, args.output_dir)
    
    print("\n" + "="*70)
    print("Evaluation Process Completed Successfully!")
    print("="*70)


if __name__ == "__main__":
    main()