from __future__ import annotations

import json
import os
import re

from .models import GEvalReport, QualityReport, Quiz


def evaluate_quiz(quiz: Quiz) -> QualityReport:
    """Transparent local equivalent of core G-Eval checks: grounding and answerability."""
    if not quiz.questions:
        return QualityReport(0, 0, 0, ["Quiz contains no questions"])
    transcript = " ".join(quiz.transcript.lower().split())
    grounded = sum(" ".join(question.evidence.lower().split()) in transcript for question in quiz.questions)
    answerable = sum(len(question.options) == 4 and 0 <= question.answer_index < 4 and bool(question.explanation) for question in quiz.questions)
    transcript_terms = set(re.findall(r"[a-z]{5,}", transcript))
    assessed = set(re.findall(r"[a-z]{5,}", " ".join(question.evidence.lower() for question in quiz.questions)))
    coverage = min(1.0, len(assessed & transcript_terms) / max(1, min(20, len(transcript_terms))))
    warnings = []
    if grounded < len(quiz.questions):
        warnings.append("Some evidence snippets are not verbatim in the transcript")
    if answerable < len(quiz.questions):
        warnings.append("Some questions have invalid options or answers")
    if coverage < 0.25:
        warnings.append("The quiz covers a narrow portion of the transcript")
    return QualityReport(round(grounded / len(quiz.questions), 2), round(answerable / len(quiz.questions), 2), round(coverage, 2), warnings)


class OpenAIGEvalEvaluator:
    """Optional rubric evaluator using OpenAI structured outputs.

    This is deliberately separate from the deterministic gates above: model scores
    are useful review signals, not objective measurements or a replacement for a
    teacher's review.
    """

    rubric = (
        "Score accuracy, relevance, clarity, and transcript grounding from 1 (poor) "
        "to 5 (excellent). Judge only the supplied transcript and quiz."
    )

    def __init__(self, client=None, model: str | None = None) -> None:
        self.client = client
        self.model = model or os.getenv("OPENAI_EVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    def evaluate(self, quiz: Quiz) -> GEvalReport:
        if self.client is None and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install openai to use model-based G-Eval") from exc
            self.client = OpenAI()

        questions = [
            {
                "prompt": row.prompt,
                "options": row.options,
                "answer_index": row.answer_index,
                "explanation": row.explanation,
                "evidence": row.evidence,
            }
            for row in quiz.questions
        ]
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are an educational assessment reviewer. Apply the supplied rubric. "
                "Return only the requested scores and short actionable feedback; do not "
                "return hidden reasoning or chain-of-thought."
            ),
            input=json.dumps({"rubric": self.rubric, "transcript": quiz.transcript, "questions": questions}),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "quiz_geval",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            name: {"type": "number", "minimum": 1, "maximum": 5}
                            for name in ("accuracy", "relevance", "clarity", "grounding")
                        }
                        | {
                            "feedback": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 5,
                            }
                        },
                        "required": ["accuracy", "relevance", "clarity", "grounding", "feedback"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        try:
            data = json.loads(response.output_text)
            scores = [float(data[name]) for name in ("accuracy", "relevance", "clarity", "grounding")]
            if any(score < 1 or score > 5 for score in scores):
                raise ValueError
            feedback = [str(item) for item in data.get("feedback", [])]
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("The evaluator returned an invalid structured report") from exc
        return GEvalReport(*scores, round(sum(scores) / len(scores), 2), feedback, self.model, self.rubric)
