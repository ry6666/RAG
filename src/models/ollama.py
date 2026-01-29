#!/usr/bin/env python3
"""
Ollama Local Model Client
Use Ollama qwen2:7b-instruct for entity extraction and answer generation
"""

import subprocess
import json
import re
from typing import List, Dict, Any, Optional


class OllamaClient:
    """Ollama Local Model Client"""

    def __init__(self, model_name: str = "qwen2:7b-instruct"):
        self.model_name = model_name
        self._check_available()

    def _check_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return self.model_name in result.stdout
            return False
        except Exception:
            return False

    def _call_model(self, prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama model"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                output = re.sub(r'<\|.*?\|>', '', output)
                output = output.strip()
                return output
            return ""
        except Exception as e:
            print(f"Ollama call failed: {e}")
            return ""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from prompt"""
        return self._call_model(prompt, temperature=0.7)

    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        """Extract triplets (entity-relation-entity) from text"""
        prompt = f"""You are a knowledge graph expert. Extract knowledge triplets from the following text in JSON format.

Text: {text}

Triplet format: {{"subject": "...", "predicate": "...", "object": "..."}}

Relation types:
- establish/create/found (A establishes B)
- replace/overthrow/defeat (A replaces B)
- capital/move capital (A makes B capital)
- rule/rule during (A rules during period B)
- battle/fight/defeat (A battles/fights/defeats B)
- invent/create (A invents/creates B)
- belong/originate from (A belongs to B, A originates from B)
- location/situated at (A is located at B)
- inherit/succeed (A inherits from B)

Return only the JSON array, nothing else.
[
    {{"subject": "Xia Dynasty", "predicate": "established", "object": "Qi"}},
    {{"subject": "Qi", "predicate": "established", "object": "Xia Dynasty"}}
]"""

        result = self._call_model(prompt, temperature=0.3)

        triplets = []
        try:
            json_str = result.strip()
            if json_str.startswith('['):
                data = json.loads(json_str)
                if isinstance(data, list):
                    for item in data:
                        if all(k in item for k in ["subject", "predicate", "object"]):
                            triplets.append({
                                "subject": str(item["subject"]).strip(),
                                "predicate": str(item["predicate"]).strip(),
                                "object": str(item["object"]).strip()
                            })
            else:
                start = json_str.find('[')
                end = json_str.rfind(']') + 1
                if start != -1 and end > 0:
                    data = json.loads(json_str[start:end])
                    if isinstance(data, list):
                        for item in data:
                            if all(k in item for k in ["subject", "predicate", "object"]):
                                triplets.append({
                                    "subject": str(item["subject"]).strip(),
                                    "predicate": str(item["predicate"]).strip(),
                                    "object": str(item["object"]).strip()
                                })
        except json.JSONDecodeError:
            pass

        return triplets[:10]

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using LLM"""
        prompt = f"""You are an information extraction expert. Extract entities from the following text. Return only entity names, separated by commas.

Text: {text}

Entity types to extract:
- Person (historical figures, emperors, officials, etc.)
- Location (countries, cities, rivers, mountains, etc.)
- Organization (government agencies, armies, schools, etc.)
- Event (wars, reforms, inventions, etc.)
- Concept (books, ideas, religions, technologies, etc.)

Return only the list: entity1, entity2, entity3
Nothing else."""

        result = self._call_model(prompt, temperature=0.3)

        entities = []
        for item in result.split(','):
            item = item.strip().strip('"\'.')
            if item and len(item) >= 2:
                entities.append(item)

        return entities[:10]

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        prompt = f"""You are a keyword extraction expert. Extract 5-10 important keywords from the following text. Return only keywords, separated by commas.

Text: {text}

Include: important people, places, time, events, cultural achievements, etc.

Return only the list: keyword1, keyword2, keyword3
Nothing else."""

        result = self._call_model(prompt, temperature=0.3)

        keywords = []
        for item in result.split(','):
            item = item.strip().strip('"\'.')
            if item and len(item) >= 2:
                keywords.append(item)

        return keywords[:10]

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer based on contexts"""
        if not contexts:
            return "Cannot determine from the provided information"

        context_text = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(contexts)])

        prompt = f"""You are a precise information extraction assistant. Follow these rules:

1. **Direct Match Priority**: If the answer to the question appears directly and completely in the following Document Information, quote or briefly summarize that answer.
2. **No Free Interpretation**: Do NOT add, infer, summarize, or explain information not explicitly stated in the Document Information.
3. **Uncertainty**: Only answer "Cannot determine from the provided information" when NO relevant facts are found in the Document Information.

Document Information:
{context_text}

Question: {question}

Give the answer directly, no explanation needed."""

        answer = self._call_model(prompt, temperature=0.5)

        if not answer or "cannot determine" in answer.lower() or "don't know" in answer.lower():
            return "Cannot determine from the provided information"

        return answer.strip()

    def refine_context(self, question: str, contexts: List[str]) -> str:
        """Refine context based on question"""
        if not contexts:
            return ""

        context_text = "\n\n".join(contexts)

        prompt = f"""You are an information processing assistant. Extract the most relevant content from the documents for the given question.

Question: {question}

Documents:
{context_text}

Return only the relevant content fragments, separated by newlines if multiple. Nothing else."""

        return self._call_model(prompt, temperature=0.5)


def main():
    """Test entity extraction"""
    client = OllamaClient()

    test_texts = [
        "The Xia Dynasty was the first dynasty in Chinese history, established by Yu's son Qi.",
        "The Liangzhu Culture is known for its sophisticated jade civilization and jade cong craftsmanship.",
        "The Chinese civilization spanning five thousand years is the only ancient civilization in the world that has continued without interruption to this day."
    ]

    print("=== Test Triplet Extraction ===")
    for text in test_texts:
        triplets = client.extract_triplets(text)
        print(f"\nText: {text}")
        print(f"Triplets: {triplets}")

    print("\n=== Test Entity Extraction ===")
    for text in test_texts:
        entities = client.extract_entities(text)
        print(f"\nText: {text}")
        print(f"Entities: {entities}")

    print("\n=== Test Keyword Extraction ===")
    for text in test_texts:
        keywords = client.extract_keywords(text)
        print(f"\nText: {text}")
        print(f"Keywords: {keywords}")


if __name__ == "__main__":
    main()
