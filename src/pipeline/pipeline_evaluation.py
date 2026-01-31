#!/usr/bin/env python3
"""Pipeline Evaluation - 使用主RAGPipeline

输出格式：
序号 | 问题 | 黄金答案 | 预测答案 | 是否正确 | 实体

生成结果报告：src/results/evaluation_report.json
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rag_pipeline import RAGPipeline


def run_pipeline_evaluation(samples_file: str, output_file: str = None, max_samples: int = 10):
    """运行pipeline评估

    Args:
        samples_file: 测试样本文件路径
        output_file: 结果报告输出路径
        max_samples: 最大样本数
    """
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if isinstance(samples, dict):
        samples = samples.get('samples', samples)
    samples = samples[:max_samples]

    pipeline = RAGPipeline()

    stats = {
        'total': len(samples),
        'correct': 0,
        'yes_no_correct': 0,
        'yes_no_total': 0,
        'bridge_correct': 0,
        'bridge_total': 0,
        'failed': [],
        'f1_scores': [],
        'em_scores': []
    }

    results_report = {
        'timestamp': datetime.now().isoformat(),
        'samples_file': samples_file,
        'max_samples': max_samples,
        'results': []
    }

    header = f"{'序号':^3} | {'正确':^2} | {'问题'}"
    separator = "=" * 90

    print(separator)
    print("Pipeline 评估结果")
    print(separator)

    for idx, sample in enumerate(samples):
        question = sample.get('question', '')
        gold_answer = sample.get('gold_answer', sample.get('answer', ''))
        q_type = sample.get('type', 'bridge')

        result = pipeline.run(question, gold_answer)
        entities = result.get('entities', [])
        pred_answer = result.get('pred_answer', '')
        is_correct = result.get('is_correct_em', False)
        f1_score = result.get('f1', 0.0)
        em_score = 1.0 if is_correct else 0.0

        stats['f1_scores'].append(f1_score)
        stats['em_scores'].append(em_score)

        if q_type == 'comparison':
            stats['yes_no_total'] += 1
        else:
            stats['bridge_total'] += 1

        if is_correct:
            stats['correct'] += 1
            if q_type == 'comparison':
                stats['yes_no_correct'] += 1
            else:
                stats['bridge_correct'] += 1
            correct_flag = '✓'
        else:
            correct_flag = '✗'
            stats['failed'].append({
                'idx': idx + 1,
                'question': question,
                'gold_answer': gold_answer,
                'pred_answer': pred_answer,
                'type': q_type,
                'entities': entities
            })

        result_entry = {
            'idx': idx + 1,
            'question': question,
            'gold_answer': gold_answer,
            'pred_answer': pred_answer,
            'is_correct': is_correct,
            'question_type': q_type,
            'entities': entities,
            'retrieved_count': len(result.get('retrieved_chunks', []))
        }
        results_report['results'].append(result_entry)

        row = f"{idx+1:^3} | {correct_flag:^2} | {question:<70}"
        print(row)
        print(f"    黄金答案: {gold_answer}")
        print(f"    预测答案: {pred_answer}")
        print(f"    提取实体: {entities}")
        print("-" * 90)

    print(separator)

    print(f"\n{'='*70}")
    print("评估结果")
    print(f"{'='*70}")
    print(f"总样本: {stats['total']}")
    print(f"正确: {stats['correct']} ({100*stats['correct']/stats['total']:.0f}%)")

    if 'f1_scores' in stats and stats['f1_scores']:
        avg_f1 = sum(stats['f1_scores']) / len(stats['f1_scores'])
        stats['avg_f1'] = round(avg_f1, 4)

    if 'em_scores' in stats and stats['em_scores']:
        avg_em = sum(stats['em_scores']) / len(stats['em_scores'])
        stats['avg_em'] = round(avg_em, 4)

    stats['accuracy'] = round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0

    print(f"\n{'='*70}")
    print("评估指标")
    print(f"{'='*70}")
    print(f"总样本: {stats['total']}")
    print(f"准确率: {stats['correct']} ({stats['accuracy']:.2%})")
    if stats['yes_no_total'] > 0:
        yes_no_acc = stats['yes_no_correct'] / stats['yes_no_total']
        stats['yes_no_accuracy'] = round(yes_no_acc, 4)
        print(f"Yes/No: {stats['yes_no_correct']}/{stats['yes_no_total']} ({stats['yes_no_accuracy']:.2%})")
    if stats['bridge_total'] > 0:
        bridge_acc = stats['bridge_correct'] / stats['bridge_total']
        stats['bridge_accuracy'] = round(bridge_acc, 4)
        print(f"桥接: {stats['bridge_correct']}/{stats['bridge_total']} ({stats['bridge_accuracy']:.2%})")
    if 'avg_f1' in stats:
        print(f"平均F1: {stats['avg_f1']:.2%}")
    if 'avg_em' in stats:
        print(f"平均EM: {stats['avg_em']:.2%}")

    if stats['failed']:
        print(f"\n失败样本:")
        for f in stats['failed'][:5]:
            print(f"  [{f['idx']}] {f['question']}")
            print(f"       黄金答案: {f['gold_answer']}")
            print(f"       预测答案: {f['pred_answer']}")
            print(f"       提取实体: {f['entities']}")

    if 'f1_scores' in stats:
        del stats['f1_scores']
    if 'em_scores' in stats:
        del stats['em_scores']

    results_report['stats'] = stats

    if output_file is None:
        output_dir = PROJECT_ROOT / 'src' / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'evaluation_report_{timestamp}.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_report, f, ensure_ascii=False, indent=2)

    print(f"\n结果报告已生成: {output_file}")

    return stats, results_report


def main():
    samples_file = 'src/samples/validation_samples_100.json'
    stats, report = run_pipeline_evaluation(samples_file, max_samples=100)
    return stats, report


if __name__ == "__main__":
    main()
