from __future__ import annotations

import re

from .evaluation import OpenAIGEvalEvaluator, evaluate_quiz
from .generation import LangChainQuizGenerator, OfflineQuizGenerator, OpenAIQuizGenerator
from .models import Quiz, Transcription, TranscriptSegment
from .text import clean_transcript


class QuizService:
    def generate(
        self,
        transcript: str,
        title: str = "Learning check",
        count: int = 5,
        difficulty: str = "medium",
        provider: str = "offline",
        seed: int = 42,
        *,
        transcript_segments: list[TranscriptSegment] | list[dict] | None = None,
        evaluation_provider: str = "deterministic",
        generator=None,
        evaluator=None,
    ) -> Quiz:
        cleaned = clean_transcript(transcript)
        providers = {
            "offline": OfflineQuizGenerator,
            "openai": OpenAIQuizGenerator,
            "langchain": LangChainQuizGenerator,
        }
        provider_key = provider.lower().strip()
        if provider_key not in providers:
            raise ValueError(f"Unknown quiz provider: {provider}")
        evaluation_key = evaluation_provider.lower().strip()
        if evaluation_key not in {"deterministic", "offline", "none", "openai", "geval", "g-eval"}:
            raise ValueError(f"Unknown evaluation provider: {evaluation_provider}")
        engine = generator or providers[provider_key]()
        quiz = engine.generate(cleaned, title.strip() or "Learning check", count, difficulty, seed)
        try:
            quiz.transcript_segments = [
                row if isinstance(row, TranscriptSegment) else TranscriptSegment(**row)
                for row in (transcript_segments or [])
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError("Transcript segments require start, end, and text fields") from exc
        for question in quiz.questions:
            if question.timestamp is None:
                question.timestamp = _evidence_timestamp(question.evidence, quiz.transcript_segments)
        quiz.quality = evaluate_quiz(quiz)
        if evaluation_key in {"openai", "geval", "g-eval"}:
            quiz.geval = (evaluator or OpenAIGEvalEvaluator()).evaluate(quiz)
        return quiz

    def generate_transcription(self, transcription: Transcription, **kwargs) -> Quiz:
        return self.generate(transcription.text, transcript_segments=transcription.segments, **kwargs)


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _evidence_timestamp(evidence: str, segments: list[TranscriptSegment]) -> float | None:
    target = _normalized(evidence)
    if not target:
        return None
    for segment in segments:
        candidate = _normalized(segment.text)
        if target in candidate or candidate in target:
            return segment.start
    target_words = set(target.split())
    ranked = [
        (len(target_words & set(_normalized(segment.text).split())) / max(1, len(target_words)), segment.start)
        for segment in segments
    ]
    best = max(ranked, default=(0.0, None))
    return best[1] if best[0] >= 0.6 else None
