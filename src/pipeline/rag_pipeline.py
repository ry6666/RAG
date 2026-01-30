#!/usr/bin/env python3
"""RAG Pipeline - 完整流程串联：分类 → 检索 → 生成 → 评估"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime

from src.retrieval.unified_retriever import get_unified_retriever
from src.generation.answer_generator import generate_answer_with_react
from src.test.metrics import evaluate_all_results, generate_unified_report


class RAGPipeline:
    """RAG 管道：串联分类、检索、生成、评估"""

    def __init__(self):
        self.retriever = get_unified_retriever()

    def run(self, question: str, gold_answer: str = None, gold_clues: list = None, latency_ms: float = None) -> dict:
        """运行完整管道

        Args:
            question: 输入问题
            gold_answer: 标准答案（可选，用于评估）
            gold_clues: 黄金线索列表（可选，用于检索层评估）
            latency_ms: 推理延迟（毫秒）

        Returns:
            包含所有步骤结果的字典
        """
        import time

        start_time = time.time()

        retrieved, analysis = self.retriever.retrieve(question)
        answer = self._generate(question, retrieved)

        latency_ms = (time.time() - start_time) * 1000

        result = {
            'question': question,
            'question_type': analysis['type'],
            'entities': analysis['entities'],
            'retrieved_chunks': retrieved,
            'pred_answer': answer
        }

        if gold_answer:
            result['gold_answer'] = gold_answer
            em, f1 = self._compute_metrics(gold_answer, answer)
            result['is_correct_em'] = em == 1.0
            result['f1_score'] = f1

        if gold_clues:
            result['gold_clues'] = gold_clues

        result['latency_ms'] = latency_ms

        return result

    def _generate(self, question: str, chunks: list) -> str:
        """生成答案"""
        from generation.answer_generator import generate_answer_with_fallback
        return generate_answer_with_fallback(question, chunks, self.retriever)

    def _compute_metrics(self, gold: str, pred: str) -> tuple:
        """计算 EM 和 F1"""
        import re
        gold_norm = set(re.sub(r'[^\w\s]', '', gold.lower()).split())
        pred_norm = set(re.sub(r'[^\w\s]', '', pred.lower()).split())

        em = 1.0 if gold_norm == pred_norm else 0.0

        overlap = gold_norm & pred_norm
        precision = len(overlap) / len(pred_norm) if pred_norm else 0
        recall = len(overlap) / len(gold_norm) if gold_norm else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return em, f1


def _check_retrieval_quality(retrieved_chunks: list, gold_answer: str, question: str) -> dict:
    """检查检索质量：检索结果是否包含答案线索"""
    gold_lower = gold_answer.lower() if gold_answer else ""
    question_lower = question.lower()
    
    keywords = [w for w in gold_lower.split() if len(w) > 2]
    
    found_in_chunks = []
    for i, chunk in enumerate(retrieved_chunks[:5]):
        core_text = chunk.get('core_text', '').lower()
        chunk_id = chunk.get('chunk_id', '')
        
        matches = []
        for kw in keywords:
            if kw in core_text:
                matches.append(kw)
        
        if matches:
            found_in_chunks.append({
                'chunk_id': chunk_id,
                'keywords_found': matches,
                'has_answer': any(gold_lower[:30] in core_text for _ in [1])
            })
    
    return {
        'keywords_found_count': len(found_in_chunks),
        'details': found_in_chunks[:3]
    }


def run_evaluation(samples_path: str, output_dir: str) -> dict:
    """批量评估（带诊断信息）
    
    Args:
        samples_path: 测试样本 JSON 文件路径
        output_dir: 输出目录
        
    Returns:
        评估报告
    """
    import os
    import time
    
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    if isinstance(samples, list):
        samples = samples[:10]
    else:
        samples = [samples]
    
    pipeline = RAGPipeline()
    
    results = []
    for idx, sample in enumerate(samples):
        question = sample.get('question', '')
        gold_answer = sample.get('answer', sample.get('gold_answer', ''))
        gold_clues = sample.get('gold_clues', extract_gold_clues(sample))
        
        print(f"\n[{idx+1}] Question: {question[:80]}...")
        print(f"    Gold: {gold_answer}")
        
        start_time = time.time()
        result = pipeline.run(question, gold_answer, gold_clues)
        latency_ms = (time.time() - start_time) * 1000
        
        result['idx'] = idx
        result['latency_ms'] = latency_ms
        
        retrieval_check = _check_retrieval_quality(
            result.get('retrieved_chunks', []), 
            gold_answer, 
            question
        )
        result['retrieval_diagnosis'] = retrieval_check
        
        print(f"    Type: {result.get('question_type', 'unknown')}")
        print(f"    Entities: {result.get('entities', [])[:3]}")
        print(f"    Retrieval: {retrieval_check['keywords_found_count']}/5 chunks have answer keywords")
        print(f"    Pred: {result.get('pred_answer', '')[:60]}...")
        
        results.append(result)

        status = "✓" if result.get('is_correct_em', False) else "✗"
        print(f"    [{status}] Final: Gold={gold_answer[:30]}... | Pred={result.get('pred_answer', '')[:30]}...")

    evaluated = evaluate_all_results(results)
    json_path = generate_unified_report(evaluated, output_dir)
    
    print(f"\n{'='*60}")
    print(f"评估完成！报告: {json_path}")
    print(f"EM: {evaluated[0].get('metrics', {}).get('em', 0) if evaluated else 0}")
    print(f"{'='*60}")


def extract_gold_clues(sample: dict) -> list:
    """从样本中提取黄金线索"""
    clues = sample.get('supporting_clues', [])
    if clues:
        return clues
    
    supporting_docs = sample.get('supporting_docs', '')
    if supporting_docs and isinstance(supporting_docs, str) and supporting_docs.strip():
        return [s.strip() for s in supporting_docs.split('. ') if s.strip()]
    
    return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Pipeline')
    parser.add_argument('--question', type=str, help='Single question')
    parser.add_argument('--gold', type=str, help='Gold answer (for evaluation)')
    parser.add_argument('--eval', type=str, help='Evaluate with sample file')
    parser.add_argument('--output', type=str, default='src/results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.question:
        pipeline = RAGPipeline()
        result = pipeline.run(args.question, args.gold)
        print(f"\nQuestion: {result['question']}")
        print(f"Type: {result['question_type']}")
        print(f"Answer: {result['pred_answer']}")
        if 'is_correct_em' in result:
            status = "✓" if result['is_correct_em'] else "✗"
            print(f"Result: {status} (F1: {result.get('f1_score', 0):.4f})")
    elif args.eval:
        run_evaluation(args.eval, args.output)
    else:
        print("Usage:")
        print("  python rag_pipeline.py --question 'Your question'")
        print("  python rag_pipeline.py --question 'Q' --gold 'Gold answer'")
        print("  python rag_pipeline.py --eval src/samples/validation_samples_10.json")
