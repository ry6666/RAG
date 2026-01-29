#!/usr/bin/env python3
"""Generate validation test samples from parquet file"""

import os
import sys
import json
import pandas as pd

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


def load_from_parquet(parquet_path: str, num: int = None) -> list:
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
            supporting_facts = [(str(sf_titles[i]) if i < len(sf_titles) else '', 
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
            'type': str(row.get('type', 'bridge')),
            'level': str(row.get('level', 'medium')),
            'supporting_facts': supporting_facts,
            'doc_id': q_id
        })
    
    print(f"[Data] Loaded {len(test_data)} valid samples from parquet")
    return test_data


def main():
    validation_path = os.path.join(proj_root, "dataset", "validation-00000-of-00001.parquet")
    output_dir = os.path.join(proj_root, "test", "validation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    test_data = load_from_parquet(validation_path, num=10)
    
    output_path = os.path.join(output_dir, "validation_samples_10.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"[Output] Saved to {output_path}")
    print(f"[Info] Total samples: {len(test_data)}")
    
    for i, item in enumerate(test_data):
        print(f"  [{i+1}] {item['question'][:60]}...")
        print(f"       Answer: {item['answer'][:40]}")
        print(f"       Type: {item['type']} | Level: {item['level']}")


if __name__ == "__main__":
    main()
