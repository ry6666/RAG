#!/usr/bin/env python3
"""Answer Generator

Uses Ollama to generate answers based on retrieval results
Constraint: Answer strictly based on clues, short answers only
"""

import os
import sys
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.ollama_model import get_ollama_model


def generate_answer_with_retrieval(question: str, max_chunks: int = 5) -> str:
    """Generate answer based on retrieval (integrated retrieval and generation)

    Args:
        question: Question
        max_chunks: Maximum chunks to use

    Returns:
        Generated answer
    """
    from src.retrieval.unified_retriever import get_unified_retriever

    retriever = get_unified_retriever()
    context = retriever.get_context(question, max_chunks=max_chunks)

    if not context.strip():
        return ""

    return _generate_from_context(question, context)


def _generate_from_context(question: str, context: str) -> str:
    """Generate answer from context string

    Args:
        question: Question
        context: Retrieved context

    Returns:
        Generated answer
    """
    ollama_model = get_ollama_model()

    if not ollama_model.is_available():
        return _fallback_from_context(context)

    chunks = context.split("\n\n")
    clues = []
    for i, chunk in enumerate(chunks[:5]):
        chunk = chunk.strip()
        if chunk and len(chunk) > 10:
            if chunk.startswith("["):
                clues.append(chunk)
            else:
                clues.append(f"[{i+1}] {chunk}")

    if not clues:
        return _fallback_from_context(context)

    clues_text = "\n".join(clues)

    prompt = f'''Based on the clues, give ONLY the answer. NO explanation, NO preamble.

Question: {question}

Clues:
{clues_text}

Answer:'''

    try:
        answer = ollama_model.generate(
            prompt,
            temperature=0.0,
            max_tokens=32
        )
        if answer and len(answer.strip()) > 0:
            answer = answer.strip()
            lines = answer.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 0 and not line.lower().startswith('based') and not line.lower().startswith('the') and not line.lower().startswith('answer'):
                    return line
            first_line = lines[0].strip() if lines else ''
            if first_line:
                return first_line
            return _fallback_from_context(context)
    except Exception as e:
        pass

    return _fallback_from_context(context)


def _fallback_from_context(context: str) -> str:
    """Fallback: extract answer from context string"""
    if not context:
        return ""

    sentences = re.split(r'[.!?]', context)
    for sent in sentences:
        sent = sent.strip()
        if sent and len(sent) > 10 and len(sent) < 200:
            return sent

    return context[:100] if len(context) > 100 else context


def generate_answer(question: str, chunks: list) -> str:
    """Generate answer based on retrieved chunks

    Args:
        question: Question
        chunks: Retrieved document chunks

    Returns:
        Generated answer
    """
    ollama_model = get_ollama_model()

    if not ollama_model.is_available():
        return _fallback_extract(chunks)

    clues = []
    for i, chunk in enumerate(chunks[:5]):
        core_text = chunk.get('core_text', '')
        if core_text and len(core_text.strip()) > 10:
            clues.append(f"[{i+1}] {core_text.strip()}")

    if not clues:
        return _fallback_extract(chunks)

    clues_text = "\n".join(clues)

    prompt = f'''Based on the clues, give ONLY the answer. NO explanation, NO preamble.

Question: {question}

Clues:
{clues_text}

Answer:'''

    try:
        answer = ollama_model.generate(
            prompt,
            temperature=0.0,
            max_tokens=32
        )
        if answer and len(answer.strip()) > 0:
            answer = answer.strip()
            lines = answer.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 0 and not line.lower().startswith('based') and not line.lower().startswith('the') and not line.lower().startswith('answer'):
                    return line
            first_line = lines[0].strip() if lines else ''
            if first_line:
                return first_line
            return _fallback_extract(chunks)
    except Exception as e:
        pass

    return _fallback_extract(chunks)


def _fallback_extract(chunks: list) -> str:
    """Fallback: extract answer directly from chunks"""
    if not chunks:
        return ""

    question = ""
    for chunk in chunks:
        if chunk.get('question'):
            question = chunk.get('question', '')
            break

    entities = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", question)
    entities = list(set(entities))

    for chunk in chunks[:3]:
        core_text = chunk.get('core_text', '')
        if not core_text:
            continue

        core_lower = core_text.lower()

        for entity in entities:
            if entity.lower() in core_lower:
                sentences = re.split(r'[.!?]', core_text)
                for sent in sentences:
                    if entity.lower() in sent.lower() and len(sent.strip()) > 5:
                        return sent.strip()

        if core_text:
            return (core_text[:80] + "...") if len(core_text) > 80 else core_text

    first_chunk = chunks[0]
    core = first_chunk.get('core_text', '')
    return (core[:80] + "...") if core and len(core) > 80 else core


def generate_answer_with_fallback(question: str, chunks: list, retriever) -> str:
    """ReAct-style answer generation with fallback retrieval

    Args:
        question: Question
        chunks: Retrieved chunks (list of dicts with core_text, chunk_id, etc.)
        retriever: UnifiedRetriever instance for fallback retrieval

    Returns:
        Generated answer
    """
    ollama_model = get_ollama_model()
    if not ollama_model.is_available():
        return _fallback_extract(chunks)

    def get_clues_from_chunks(chunks_list, max_chunks=5):
        clues = []
        for i, chunk in enumerate(chunks_list[:max_chunks]):
            core_text = chunk.get('core_text', '')
            if core_text and len(core_text.strip()) > 10:
                if core_text.startswith("["):
                    clues.append(core_text)
                else:
                    clues.append(f"[{i+1}] {core_text.strip()}")
        return clues

    def extract_answer_from_clues(question: str, clues: list) -> str:
        """从线索中提取答案（增强版：支持简洁答案提取）"""
        if not clues:
            return None

        clues_text = "\n".join(clues)

        question_lower = question.lower().strip()
        is_what_is_question = (
            question_lower.startswith('what ') or
            question_lower.startswith('who ') or
            question_lower.startswith('which ')
        )

        if is_what_is_question:
            prompt = f'''Extract the entity name that answers the question. Give ONLY the entity name, NO description.

Question: {question}

Clues:
{clues_text}

Entity name:'''
        else:
            prompt = f'''Based on the clues, give ONLY the answer. NO explanation.

Question: {question}

Clues:
{clues_text}

Answer:'''

        try:
            answer = ollama_model.generate(
                prompt,
                temperature=0.0,
                max_tokens=32
            )
            if answer and len(answer.strip()) > 0:
                answer = answer.strip()

                answer = re.sub(r'^(The )?(answer|entity|name)[:：]\s*', '', answer, flags=re.IGNORECASE)

                lines = answer.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 0:
                        if not line.lower().startswith('based') and \
                           not line.lower().startswith('the') and \
                           not line.lower().startswith('answer') and \
                           not line.lower().startswith('clue') and \
                           not line.lower().startswith('it is') and \
                           not line.lower().startswith('this is') and \
                           not line.startswith('[') and \
                           len(line) < 100:
                            return line

                first_line = lines[0].strip() if lines else ''
                if first_line:
                    return first_line
        except Exception:
            pass
        return None

    def contains_relevant_info(clues: list, question: str) -> bool:
        """Check if clues contain potentially relevant information"""
        if not clues:
            return False

        clues_text = " ".join(clues).lower()

        reject_patterns = [
            "don't know", "cannot answer", "no information",
            "not enough information", "cannot determine",
            "not provide", "not mention", "not related"
        ]

        for pattern in reject_patterns:
            if pattern in clues_text:
                return False

        return True

    clues = get_clues_from_chunks(chunks)

    answer = extract_answer_from_clues(question, clues)

    if answer and contains_relevant_info(clues, question):
        return answer

    if not contains_relevant_info(clues, question) and retriever:
        try:
            fallback_results, analysis = retriever.retrieve(question)

            if fallback_results:
                fallback_chunks = [{"core_text": r.get("core_text", ""),
                                   "chunk_id": r.get("chunk_id", "")}
                                  for r in fallback_results[:10]]

                fallback_clues = get_clues_from_chunks(fallback_chunks)

                answer = extract_answer_from_clues(question, fallback_clues)

                if answer:
                    return answer
        except Exception:
            pass

    return _fallback_extract(chunks)


def generate_answer_with_react(question: str, chunks: list, retriever, max_iterations: int = 3) -> dict:
    """ReAct模式生成答案：多轮推理和检索

    流程：
    1. 初始检索并重排上下文
    2. 大模型判断是否有足够信息回答问题
    3. 如果没有，生成下一个查询
    4. 重新检索并融合结果
    5. 重复直到有足够信息或达到最大轮次

    Args:
        question: 输入问题
        chunks: 初始检索的文档块
        retriever: 检索器实例（用于补充检索）
        max_iterations: 最大推理轮次（默认3）

    Returns:
        dict: 包含答案、推理过程、检索历史的结果
    """
    ollama_model = get_ollama_model()

    react_result = {
        'question': question,
        'final_answer': None,
        'iterations': [],
        'all_retrieved_chunks': [],
        'reasoning_trace': []
    }

    def format_chunks_for_llm(chunks_list, max_chunks=8):
        """将文档块格式化为LLM可读格式"""
        clues = []
        for i, chunk in enumerate(chunks_list[:max_chunks]):
            core_text = chunk.get('core_text', '')
            chunk_id = chunk.get('chunk_id', '')
            metadata = chunk.get('metadata', {})

            if core_text and len(core_text.strip()) > 10:
                clue_type = metadata.get('clue_type', '')
                source_entity = metadata.get('source_entity', '')

                clue_info = f"[线索{i+1}] "
                if source_entity:
                    clue_info += f"来源实体: {source_entity} | "
                if clue_type:
                    clue_info += f"类型: {clue_type} | "
                clue_info += f"内容: {core_text.strip()[:300]}"
                if len(core_text) > 300:
                    clue_info += "..."

                clues.append(clue_info)
        return "\n\n".join(clues) if clues else "无相关线索"

    def judge_can_answer(question: str, clues: str) -> tuple:
        """让LLM判断当前线索是否足够回答问题

        Returns:
            tuple: (can_answer: bool, reasoning: str)
        """
        if not ollama_model.is_available():
            return True, "模型不可用，直接生成答案"

        prompt = f'''你是一个问答系统。你的任务是判断当前收集的线索是否足够回答问题。

问题：{question}

当前收集的线索：
{clues}

请仔细分析：
1. 线索中是否包含回答问题所需的所有关键信息？
2. 是否有明确的实体、日期、数值、关系等答案要素？

如果可以回答，请输出：
[判断] 可以回答
[原因] 简要说明为什么线索足够

如果不能回答，请输出：
[判断] 无法回答
[原因] 简要说明缺少什么信息
[建议查询] 建议下一步查询什么内容（用一句话描述）

输出格式示例：
[判断] 可以回答
[原因] 线索中明确提到A和B的国籍都是美国

或：
[判断] 无法回答
[原因] 线索中只提到A的国籍，未提及B的国籍
[建议查询] B的国籍信息'''

        try:
            response = ollama_model.generate(
                prompt,
                temperature=0.0,
                max_tokens=256
            )

            response = response.strip() if response else ""

            if '[判断] 可以回答' in response:
                reason_start = response.find('[原因]')
                reason = response[reason_start+3:].strip() if reason_start != -1 else response
                return True, reason
            elif '[判断] 无法回答' in response:
                reason_start = response.find('[原因]')
                suggestion_start = response.find('[建议查询]')

                reason = ""
                suggestion = ""

                if reason_start != -1:
                    reason_end = suggestion_start if suggestion_start != -1 else len(response)
                    reason = response[reason_start+3:reason_end].strip()

                if suggestion_start != -1:
                    suggestion = response[suggestion_start+6:].strip()

                return False, f"{reason}\n建议查询: {suggestion}"

            return False, response

        except Exception as e:
            return False, f"判断失败: {str(e)}"

    def generate_next_query(question: str, clues: str, iteration: int) -> str:
        """根据当前线索生成下一步查询"""
        if not ollama_model.is_available():
            return question

        prompt = f'''你是一个智能检索系统。根据当前问题和已收集的线索，生成一个更精确的查询来补充缺失信息。

原始问题：{question}

当前轮次：{iteration + 1}/{max_iterations}

已收集的线索：
{clues}

请分析：
1. 当前线索缺少什么关键信息？
2. 应该用什么关键词进行下一步检索？

请直接输出下一步查询语句（用英文，简洁明了）：
'''

        try:
            response = ollama_model.generate(
                prompt,
                temperature=0.3,
                max_tokens=64
            )
            response = response.strip() if response else ""
            if response:
                return response
        except Exception:
            pass

        return question

    def extract_final_answer(question: str, clues: str) -> str:
        """从最终线索中提取答案（增强版）"""
        import re

        if not clues or clues == "无相关线索":
            return ""

        core_texts = re.findall(r'内容[:：] (.+?)(?:\.\.\.|$|\n|\[线索\d+\]|$)', clues)
        if not core_texts:
            core_texts = [c for c in clues.split('\n') if len(c) > 20 and not c.startswith('[')]

        combined_text = ' '.join(core_texts[:3])

        if not ollama_model.is_available():
            if combined_text:
                sentences = re.split(r'[.!?]', combined_text)
                for sent in sentences:
                    sent = sent.strip()
                    if 5 < len(sent) < 150:
                        return sent
            return combined_text[:100] if combined_text else ""

        prompt = f'''从以下线索中提取问题的答案。只输出答案，不要任何解释或格式。

问题：{question}

线索：
{clues}

答案：'''

        try:
            answer = ollama_model.generate(
                prompt,
                temperature=0.0,
                max_tokens=32
            )

            if answer and len(answer.strip()) > 0:
                answer = answer.strip()

                answer = re.sub(r'^答案[：:]\s*', '', answer)
                answer = re.sub(r'^\d+[.。]\s*', '', answer)

                lines = answer.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 1:
                        if not line.lower().startswith('based') and \
                           not line.lower().starts_with('the') and \
                           not line.lower().starts_with('answer') and \
                           not line.lower().starts_with('线索') and \
                           not line.startswith('[') and \
                           not line.startswith('问题') and \
                           not line.startswith('线索'):
                            return line

                first_line = lines[0].strip() if lines else ''
                if first_line:
                    return first_line
        except Exception:
            pass

        if combined_text:
            sentences = re.split(r'[.!?]', combined_text)
            for sent in sentences:
                sent = sent.strip()
                if 5 < len(sent) < 150 and not sent.startswith('['):
                    return sent
            return combined_text[:80] + "..."

        return ""

    all_chunks = []
    for chunk in chunks:
        if chunk.get('chunk_id'):
            all_chunks.append(chunk)

    for iteration in range(max_iterations):
        iteration_info = {
            'iteration': iteration + 1,
            'chunks_used': len(all_chunks),
            'judge_result': None,
            'new_query': None,
            'retrieved_new': 0
        }

        formatted_clues = format_chunks_for_llm(all_chunks)

        can_answer, reasoning = judge_can_answer(question, formatted_clues)

        iteration_info['judge_result'] = {
            'can_answer': can_answer,
            'reasoning': reasoning
        }
        react_result['reasoning_trace'].append({
            'iteration': iteration + 1,
            'clues_summary': formatted_clues[:200] + "...",
            'judgement': reasoning
        })

        if can_answer:
            final_answer = extract_final_answer(question, formatted_clues)
            react_result['final_answer'] = final_answer
            react_result['iterations'].append(iteration_info)
            break

        if iteration < max_iterations - 1:
            new_query = generate_next_query(
                question, formatted_clues, iteration
            )
            iteration_info['new_query'] = new_query

            if retriever and new_query:
                try:
                    if hasattr(retriever, 'retrieve'):
                        new_results, _ = retriever.retrieve(new_query)
                        new_chunks = [r for r in new_results if r.get('chunk_id') not in
                                     {c.get('chunk_id') for c in all_chunks}]

                        for chunk in new_chunks:
                            all_chunks.append(chunk)

                        iteration_info['retrieved_new'] = len(new_chunks)
                        react_result['all_retrieved_chunks'].extend(
                            [r.get('chunk_id') for r in new_chunks]
                        )
                except Exception:
                    pass

        react_result['iterations'].append(iteration_info)

    if react_result['final_answer'] is None:
        formatted_all = format_chunks_for_llm(all_chunks)
        react_result['final_answer'] = extract_final_answer(
            question, formatted_all
        )

    return react_result


if __name__ == "__main__":
    test_chunks = [
        {'core_text': 'Scott Derrickson is an American film director born in 1966.', 'question': ''},
        {'core_text': 'Ed Wood was an American filmmaker, widely regarded as one of the worst directors.', 'question': ''}
    ]
    result = generate_answer("Were Scott Derrickson and Ed Wood of the same nationality?", test_chunks)
    print(f"Generated answer: {result}")
