#!/usr/bin/env python3
"""Multi-hop Reasoning Prompts and Pipeline for HotpotQA"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.local_llm import get_local_llm


class MultiHopReasoningPipeline:
    """多跳推理Pipeline - 支持问题分解和分步检索"""

    def __init__(self):
        self.llm = get_local_llm()
        from src.retrieval.unified_retriever import UnifiedRetriever
        self.retriever = UnifiedRetriever()

    def decompose_question(self, question: str) -> Dict:
        """使用LLM分解桥接问题为子问题"""
        
        prompt = f"""You are a high-precision multi-hop reasoning expert for HotpotQA.
Analyze the following question and decompose it into 2-3 logical sub-questions.

Question: {question}

Instructions:
1. Identify the bridge entity and reasoning chain
2. Break down into sub-questions that lead to the final answer
3. Each sub-question should be answerable by retrieving specific evidence

Output format (JSON only, no extra text):
{{
    "sub_questions": [
        {{
            "id": 1,
            "question": "first sub-question",
            "purpose": "what this helps find"
        }},
        {{
            "id": 2,
            "question": "second sub-question", 
            "purpose": "what this helps find"
        }}
    ],
    "bridge_entity": "the connecting entity between sub-questions",
    "reasoning_chain": "brief description of reasoning path"
}}
"""

        try:
            response = self.llm.generate(prompt, max_tokens=500, temperature=0.1)
            return self._parse_decomposition(response, question)
        except Exception as e:
            return self._fallback_decompose(question)

    def _parse_decomposition(self, response: str, original_question: str) -> Dict:
        """解析LLM返回的分解结果"""
        import json
        import re
        
        try:
            json_str = response.strip()
            json_str = re.sub(r'^```json\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
            data = json.loads(json_str)
            return data
        except Exception:
            return self._fallback_decompose(original_question)

    def _fallback_decompose(self, question: str) -> Dict:
        """回退分解策略：基于规则的问题分解"""
        import re
        
        question_lower = question.lower()
        
        patterns = [
            (r'who (?:was|is) .+ and .+', [
                {"id": 1, "question": question, "purpose": "find entity and related information"}
            ]),
            (r'what .+ position .+ (?:who|whom) .+', [
                {"id": 1, "question": "Who is the person mentioned?", "purpose": "identify the person"},
                {"id": 2, "question": question, "purpose": "find the position they held"}
            ]),
            (r'the .+ (?:where|when|who) .+', [
                {"id": 1, "question": f"Find information about: {question}", "purpose": "extract key entity"}
            ]),
            (r'.+ (?:is|are) .+ (?:from|of|in) .+', [
                {"id": 1, "question": question, "purpose": "find relationship"}
            ])
        ]
        
        for pattern, default_subqs in patterns:
            if re.search(pattern, question_lower):
                return {
                    "sub_questions": default_subqs,
                    "bridge_entity": "",
                    "reasoning_chain": "Rule-based decomposition"
                }
        
        return {
            "sub_questions": [{"id": 1, "question": question, "purpose": "direct answer"}],
            "bridge_entity": "",
            "reasoning_chain": "Single question"
        }

    def retrieve_with_sub_questions(self, question: str, sub_questions: List[Dict]) -> List[Dict]:
        """对每个子问题分别检索，然后合并结果"""
        all_results = []
        seen_chunks = set()
        
        for sq in sub_questions:
            sq_text = sq.get("question", "")
            results, analysis = self.retriever.retrieve(sq_text)
            
            for r in results:
                chunk_id = r.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    r["sub_question_id"] = sq.get("id", 0)
                    r["sub_question_purpose"] = sq.get("purpose", "")
                    all_results.append(r)
        
        all_results = sorted(all_results, key=lambda x: x.get("fused_score", 0), reverse=True)
        return all_results[:20]

    def synthesize_answer(self, question: str, sub_questions: List[Dict], 
                         evidences: List[str]) -> str:
        """基于检索到的证据综合答案（严格基于证据，不使用外部知识）"""
        
        if not evidences:
            return "Cannot get the answer from the given context"
        
        evidence_text = "\n".join([f"[Evidence {i+1}] {e}" for i, e in enumerate(evidences)])
        
        prompt = f"""You must answer ONLY based on the provided evidence. Do NOT use any external knowledge.

Original Question: {question}

Sub-questions to answer:
{chr(10).join([f"{sq['id']}. {sq['question']}" for sq in sub_questions])}

Supporting Evidence (ONLY use these, nothing else):
{evidence_text}

IMPORTANT RULES:
1. If the evidence does NOT contain information to answer the question, you MUST say: "Cannot get the answer from the given context"
2. Do NOT guess or use external knowledge
3. Extract the answer ONLY from the evidence provided above

Step-by-step reasoning (internal use only):
1. Check each piece of evidence for relevant information
2. If no evidence contains the answer, output "Cannot get the answer from the given context"
3. If evidence contains the answer, provide it

Final Answer (must be extractable from evidence):"""

        try:
            answer = self.llm.generate(prompt, max_tokens=200, temperature=0.0)
            answer = answer.strip()
            
            if not answer or len(answer) < 2:
                return "Cannot get the answer from the given context"
            
            if "cannot get the answer" in answer.lower() or "无法" in answer:
                return "Cannot get the answer from the given context"
            
            return answer
        except Exception:
            return "Cannot get the answer from the given context"

    def run(self, question: str, gold_answer: str = None, gold_clues: list = None) -> Dict:
        """运行多跳推理Pipeline"""
        import time
        from src.test.hotpotqa_evaluation import compute_em, compute_f1
        
        start_time = time.time()
        
        question_type = "bridge" if " and " in question or " or " in question else "bridge"
        
        print(f"  [Step 1] Decomposing question...")
        decomposition = self.decompose_question(question)
        sub_questions = decomposition.get("sub_questions", [])
        
        print(f"  [Step 2] Multi-hop retrieval for {len(sub_questions)} sub-questions...")
        retrieved_chunks = self.retrieve_with_sub_questions(question, sub_questions)
        
        print(f"  [Step 3] Synthesizing answer...")
        evidences = [r.get("core_text", "") for r in retrieved_chunks[:5] if r.get("core_text")]
        answer = self.synthesize_answer(question, sub_questions, evidences)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            "question": question,
            "question_type": question_type,
            "sub_questions": sub_questions,
            "retrieved_chunks": retrieved_chunks,
            "pred_answer": answer,
            "latency_ms": latency_ms
        }
        
        if gold_answer:
            em = compute_em(gold_answer, answer)
            f1 = compute_f1(gold_answer, answer)
            result["gold_answer"] = gold_answer
            result["em"] = em
            result["f1"] = f1
        
        return result


def run_comparison():
    """比较传统RAG和多跳推理RAG的效果"""
    from src.pipeline.rag_pipeline import RAGPipeline
    
    print("="*70)
    print("COMPARISON: Traditional RAG vs Multi-hop Reasoning RAG")
    print("="*70)
    
    df = pd.read_parquet('data/validation/validation-00000-of-00001.parquet')
    samples = df.to_dict('records')[:10]
    
    traditional_pipeline = RAGPipeline()
    multihop_pipeline = MultiHopReasoningPipeline()
    
    traditional_results = []
    multihop_results = []
    
    for i, sample in enumerate(samples):
        question = sample.get('question', '')
        gold_answer = sample.get('answer', '')
        q_type = sample.get('type', 'bridge')
        
        print(f"\n[{i+1}/10] {question[:50]}...")
        
        trad_result = traditional_pipeline.run(question, gold_answer)
        trad_em = trad_result.get('em', 0)
        print(f"    Traditional RAG: EM={trad_em:.2f} | Answer: {trad_result.get('pred_answer', '')[:30]}...")
        
        if q_type == "bridge":
            mh_result = multihop_pipeline.run(question, gold_answer)
            mh_em = mh_result.get('em', 0)
            print(f"    Multi-hop RAG:    EM={mh_em:.2f} | Answer: {mh_result.get('pred_answer', '')[:30]}...")
        else:
            mh_result = trad_result
            mh_em = trad_em
        
        traditional_results.append(trad_result)
        multihop_results.append(mh_result)
    
    trad_em_avg = sum(r.get('em', 0) for r in traditional_results) / len(traditional_results)
    mh_em_avg = sum(r.get('em', 0) for r in multihop_results) / len(multihop_results)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Traditional RAG EM: {trad_em_avg:.4f}")
    print(f"Multi-hop RAG EM:   {mh_em_avg:.4f}")
    print(f"Improvement:        {(mh_em_avg - trad_em_avg)*100:+.2f}%")


if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_parquet('data/validation/validation-00000-of-00001.parquet')
    samples = df.to_dict('records')[:10]
    
    pipeline = MultiHopReasoningPipeline()
    
    for i, sample in enumerate(samples[:3]):
        question = sample.get('question', '')
        gold_answer = sample.get('answer', '')
        
        print(f"\n{'='*70}")
        print(f"Question {i+1}: {question}")
        print(f"{'='*70}")
        
        result = pipeline.run(question, gold_answer)
        
        print(f"\nSub-questions:")
        for sq in result.get('sub_questions', []):
            print(f"  [{sq.get('id')}] {sq.get('question')}")
            print(f"      Purpose: {sq.get('purpose', '')}")
        
        print(f"\nRetrieved {len(result.get('retrieved_chunks', []))} chunks")
        print(f"Gold Answer: {gold_answer}")
        print(f"Pred Answer: {result.get('pred_answer', '')}")
        print(f"EM: {result.get('em', 0):.2f}")
