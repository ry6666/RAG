#!/usr/bin/env python3
"""Metrics Calculator

评估 RAG 系统全流程指标：
- 检索层：Recall@k, Precision@k, MRR, Hit Rate@k
- 生成层：EM, F1 Score
- 全流程：End-to-End EM/F1, Overall Accuracy, Latency
"""

import json
import os
import re
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime


def normalize_answer(text: str) -> Set[str]:
    """标准化答案：转小写，移除标点，分词"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return set(words)


def exact_match(gold: str, pred: str) -> bool:
    """EM: 精确匹配（忽略大小写和标点）"""
    return normalize_answer(gold) == normalize_answer(pred)


def compute_f1(gold: str, pred: str) -> float:
    """计算 F1 值（单词级）"""
    gold_words = normalize_answer(gold)
    pred_words = normalize_answer(pred)
    
    if len(gold_words) == 0 and len(pred_words) == 0:
        return 1.0
    if len(gold_words) == 0 or len(pred_words) == 0:
        return 0.0
    
    overlap = gold_words & pred_words
    precision = len(overlap) / len(pred_words) if len(pred_words) > 0 else 0
    recall = len(overlap) / len(gold_words) if len(gold_words) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_em_f1(gold: str, pred: str) -> Tuple[float, float]:
    """计算 EM 和 F1"""
    em = 1.0 if exact_match(gold, pred) else 0.0
    f1 = compute_f1(gold, pred)
    return em, f1


def compute_retrieval_metrics(results: List[Dict], k_values: List[int] = [5, 10]) -> Dict:
    """计算检索层指标
    
    Args:
        results: 包含 retrieved_chunks 和 supporting_clues 的结果列表
        k_values: 要计算的 k 值列表
        
    Returns:
        检索层指标字典
    """
    metrics = {
        'recall': {},
        'precision': {},
        'mrr': 0.0,
        'hit_rate': {}
    }
    
    valid_results = []
    for r in results:
        retrieved = r.get('retrieved_chunks', [])
        gold_clues = r.get('gold_clues', r.get('supporting_clues', []))
        if retrieved and gold_clues:
            valid_results.append(r)
    
    if not valid_results:
        return metrics
    
    for k in k_values:
        recalls = []
        precisions = []
        hits = 0
        mrr_sum = 0.0
        
        for r in valid_results:
            retrieved = r.get('retrieved_chunks', [])[:k]
            gold_clues = r.get('gold_clues', [])
            
            if not gold_clues:
                continue
            
            gold_set = set(normalize_answer(c) for c in gold_clues)
            
            found_clues = set()
            for chunk in retrieved:
                chunk_text = chunk.get('core_text', '')
                chunk_norm = normalize_answer(chunk_text)
                for gold in gold_set:
                    if gold and gold in chunk_norm:
                        found_clues.add(gold)
            
            recall = len(found_clues) / len(gold_set) if gold_set else 0
            recalls.append(recall)
            
            precision = len(found_clues) / len(retrieved) if retrieved else 0
            precisions.append(precision)
            
            if len(found_clues) > 0:
                hits += 1
            
            for i, chunk in enumerate(retrieved):
                chunk_text = chunk.get('core_text', '')
                chunk_norm = normalize_answer(chunk_text)
                for gold in gold_set:
                    if gold and gold in chunk_norm:
                        mrr_sum += 1.0 / (i + 1)
                        break
                else:
                    continue
                break
        
        n = len(valid_results)
        metrics['recall'][f'recall@{k}'] = sum(recalls) / n if recalls else 0
        metrics['precision'][f'precision@{k}'] = sum(precisions) / n if precisions else 0
        metrics['hit_rate'][f'hit@{k}'] = hits / n if n > 0 else 0
    
    metrics['mrr'] = mrr_sum / len(valid_results) if valid_results else 0
    
    return metrics


def compute_generation_metrics(results: List[Dict]) -> Dict:
    """计算生成层指标
    
    Args:
        results: 包含 gold_answer 和 pred_answer 的结果列表
        
    Returns:
        生成层指标字典
    """
    em_scores = []
    f1_scores = []
    
    for r in results:
        gold = r.get('gold_answer', '')
        pred = r.get('pred_answer', '')
        em, f1 = compute_em_f1(gold, pred)
        em_scores.append(em)
        f1_scores.append(f1)
    
    return {
        'em': sum(em_scores) / len(em_scores) if em_scores else 0,
        'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        'em_scores': em_scores,
        'f1_scores': f1_scores
    }


def compute_full_pipeline_metrics(results: List[Dict]) -> Dict:
    """计算全流程指标
    
    Args:
        results: 完整流程结果列表
        
    Returns:
        全流程指标字典
    """
    if not results:
        return {'end_to_end_em': 0, 'end_to_end_f1': 0, 'overall_accuracy': 0, 'avg_latency_ms': 0}
    
    em_scores = []
    f1_scores = []
    overall_correct = 0
    latencies = []
    
    for r in results:
        gold = r.get('gold_answer', '')
        pred = r.get('pred_answer', '')
        em, f1 = compute_em_f1(gold, pred)
        em_scores.append(em)
        f1_scores.append(f1)
        
        retrieved = r.get('retrieved_chunks', [])
        has_valid_clues = len(retrieved) > 0
        if has_valid_clues and em == 1.0:
            overall_correct += 1
        
        latency = r.get('latency_ms', 0)
        if latency > 0:
            latencies.append(latency)
    
    return {
        'end_to_end_em': sum(em_scores) / len(em_scores) if em_scores else 0,
        'end_to_end_f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        'overall_accuracy': overall_correct / len(results) if results else 0,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0
    }


def evaluate_result(result: Dict) -> Dict:
    """评估单个结果"""
    gold = result.get('gold_answer', '')
    pred = result.get('pred_answer', '')
    
    em, f1 = compute_em_f1(gold, pred)
    
    result['is_correct_em'] = em == 1.0
    result['f1_score'] = f1
    
    return result


def evaluate_all_results(results: List[Dict]) -> List[Dict]:
    """评估所有结果"""
    return [evaluate_result(r) for r in results]


def generate_unified_report(results: List[Dict], output_dir: str) -> str:
    """生成简化评估报告

    Args:
        results: 评估结果列表
        output_dir: 输出目录

    Returns:
        报告文件路径
    """
    em_scores = []
    f1_scores = []

    simplified_results = []
    for r in results:
        gold = r.get('gold_answer', '')
        pred = r.get('pred_answer', '')
        em, f1 = compute_em_f1(gold, pred)
        em_scores.append(em)
        f1_scores.append(f1)

        context_ids = [c.get('chunk_id', '') for c in r.get('retrieved_chunks', [])]

        simplified_results.append({
            'question': r.get('question', ''),
            'gold_answer': gold,
            'pred_answer': pred,
            'is_correct': em == 1.0,
            'f1_score': round(f1, 4),
            'context_ids': context_ids,
            'question_type': r.get('question_type', '')
        })

    report_data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'em': round(sum(em_scores) / len(em_scores), 4) if em_scores else 0,
            'f1': round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
            'total': len(results),
            'correct': sum(1 for s in em_scores if s == 1.0)
        },
        'results': simplified_results
    }

    json_path = f"{output_dir}/eval_results.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    return json_path


if __name__ == "__main__":
    test_results = [
        {
            'idx': 0,
            'question': 'Test question 1',
            'question_type': 'bridge',
            'gold_answer': 'American',
            'pred_answer': 'American',
            'retrieved_chunks': [
                {'core_text': 'Scott Derrickson is an American film director.'},
                {'core_text': 'Ed Wood was an American filmmaker.'}
            ],
            'gold_clues': ['American', 'American'],
            'latency_ms': 150.5
        },
        {
            'idx': 1,
            'question': 'Test question 2',
            'question_type': 'comparison',
            'gold_answer': 'Paris',
            'pred_answer': 'London'
        }
    ]
    
    evaluated = evaluate_all_results(test_results)
    retrieval = compute_retrieval_metrics(evaluated)
    generation = compute_generation_metrics(evaluated)
    pipeline = compute_full_pipeline_metrics(evaluated)
    
    print("Retrieval Metrics:", json.dumps(retrieval, indent=2, ensure_ascii=False))
    print("\nGeneration Metrics:", json.dumps(generation, indent=2, ensure_ascii=False))
    print("\nPipeline Metrics:", json.dumps(pipeline, indent=2, ensure_ascii=False))
