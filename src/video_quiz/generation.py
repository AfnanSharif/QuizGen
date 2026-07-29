from __future__ import annotations

import json
import os
import random
import re

from .models import Quiz, QuizQuestion
from .text import concepts, sentences, summarize


class OfflineQuizGenerator:
    name = "Offline grounded generator"

    def generate(self, transcript: str, title: str, count: int = 5, difficulty: str = "medium", seed: int = 42) -> Quiz:
        if count < 2 or count > 20:
            raise ValueError("Question count must be between 2 and 20")
        summary, key_concepts = summarize(transcript)
        rows = sentences(transcript)
        candidates = []
        for row in rows:
            lowered = row.lower()
            term = next((concept for concept in key_concepts if len(concept) >= 5 and re.search(rf"\b{re.escape(concept)}\b", lowered)), None)
            if term:
                candidates.append((row, term))
        if len(candidates) < 2:
            raise ValueError("Not enough distinct concepts were found to build a useful quiz")
        rng = random.Random(seed)
        rng.shuffle(candidates)
        selected = candidates[: min(count, len(candidates))]
        questions = []
        distractor_pool = list(dict.fromkeys(key_concepts))
        for index, (evidence, answer) in enumerate(selected, 1):
            blanked = re.sub(rf"\b{re.escape(answer)}\b", "_____", evidence, count=1, flags=re.I)
            distractors = [word for word in distractor_pool if word != answer and word not in evidence.lower()]
            rng.shuffle(distractors)
            options = [answer.title()] + [word.title() for word in distractors[:3]]
            if len(options) < 4:
                options.extend([f"Unrelated concept {number}" for number in range(1, 5 - len(options))])
            rng.shuffle(options)
            questions.append(
                QuizQuestion(
                    id=index,
                    prompt=f"Which term best completes this idea? “{blanked}”",
                    options=options,
                    answer_index=options.index(answer.title()),
                    explanation=f"The transcript states: {evidence}",
                    evidence=evidence,
                    difficulty=difficulty,
                )
            )
        return Quiz(title, summary, [item.title() for item in key_concepts[:8]], transcript, questions, provider=self.name)


class OpenAIQuizGenerator:
    name = "OpenAI structured generator"

    def __init__(self, client=None, model: str | None = None) -> None:
        self.client = client
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, transcript: str, title: str, count: int = 5, difficulty: str = "medium", seed: int = 42) -> Quiz:
        if self.client is None and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install openai to use GPT quiz generation") from exc
            self.client = OpenAI()
        instruction = f"""Use only the transcript. Return JSON with summary, key_concepts, and questions.
Each of exactly {count} questions has prompt, four options, answer_index (0-3), explanation, evidence (an exact supporting transcript sentence). Difficulty: {difficulty}. Never invent facts.
Transcript:\n{transcript}"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You create grounded educational assessments and output valid JSON only."}, {"role": "user", "content": instruction}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(response.choices[0].message.content)
        return _quiz_from_payload(data, transcript, title, count, difficulty, self.name)


class LangChainQuizGenerator:
    """Optional LangChain structured-output path with an injectable runnable."""

    name = "LangChain + OpenAI structured generator"

    def __init__(self, runnable=None, model: str | None = None) -> None:
        self.runnable = runnable
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _build_runnable(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        try:
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field
        except ImportError as exc:
            raise RuntimeError("Install langchain-openai and pydantic to use LangChain structured parsing") from exc

        class QuestionPayload(BaseModel):
            prompt: str
            options: list[str] = Field(min_length=4, max_length=4)
            answer_index: int = Field(ge=0, le=3)
            explanation: str
            evidence: str

        class QuizPayload(BaseModel):
            summary: str
            key_concepts: list[str]
            questions: list[QuestionPayload]

        QuizPayload.model_rebuild(_types_namespace={"QuestionPayload": QuestionPayload})
        return ChatOpenAI(model=self.model, temperature=0.2).with_structured_output(QuizPayload)

    def generate(self, transcript: str, title: str, count: int = 5, difficulty: str = "medium", seed: int = 42) -> Quiz:
        runnable = self.runnable or self._build_runnable()
        instruction = f"""Create exactly {count} {difficulty} multiple-choice questions from only this transcript.
Return a concise summary, key_concepts, and questions. Every question must have exactly four options, a 0-based answer_index, explanation, and an exact evidence sentence copied from the transcript. Never add outside facts.\n\n{transcript}"""
        response = runnable.invoke(instruction)
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif isinstance(response, dict):
            data = response
        else:
            raise RuntimeError("LangChain returned an unsupported structured response")
        return _quiz_from_payload(data, transcript, title, count, difficulty, self.name)


def _quiz_from_payload(data: dict, transcript: str, title: str, count: int, difficulty: str, provider: str) -> Quiz:
    rows = []
    for index, item in enumerate(data.get("questions", []), 1):
        options = [str(option) for option in item["options"]]
        answer_index = int(item["answer_index"])
        if len(options) != 4 or not 0 <= answer_index < 4:
            raise RuntimeError("Provider returned an invalid question shape")
        rows.append(
            QuizQuestion(
                index,
                str(item["prompt"]),
                options,
                answer_index,
                str(item["explanation"]),
                str(item["evidence"]),
                difficulty,
                float(item["timestamp"]) if item.get("timestamp") is not None else None,
            )
        )
    if len(rows) != count:
        raise RuntimeError(f"Provider returned {len(rows)} questions; expected {count}")
    return Quiz(title, str(data["summary"]), [str(x) for x in data.get("key_concepts", [])], transcript, rows, provider=provider)
