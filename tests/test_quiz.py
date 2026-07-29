import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_quiz.evaluation import OpenAIGEvalEvaluator
from video_quiz.generation import LangChainQuizGenerator
from video_quiz.media import parse_transcript, transcribe
from video_quiz.models import Quiz, Transcription, TranscriptSegment
from video_quiz.presentation import escape_html
from video_quiz.service import QuizService
from video_quiz.text import clean_transcript


class QuizTests(unittest.TestCase):
    def setUp(self):
        self.transcript = (ROOT / "sample_data" / "lesson.txt").read_text(encoding="utf-8")
        self.quiz = QuizService().generate(self.transcript, count=5, seed=9)

    def test_builds_valid_questions(self):
        self.assertEqual(len(self.quiz.questions), 5)
        for question in self.quiz.questions:
            self.assertEqual(len(question.options), 4)
            self.assertIn(question.options[question.answer_index].lower(), question.evidence.lower())

    def test_quality_is_grounded(self):
        self.assertEqual(self.quiz.quality.grounding, 1.0)
        self.assertEqual(self.quiz.quality.answerability, 1.0)

    def test_deterministic_with_seed(self):
        again = QuizService().generate(self.transcript, count=5, seed=9)
        self.assertEqual(self.quiz.to_dict(), again.to_dict())

    def test_short_transcript_rejected(self):
        with self.assertRaises(ValueError):
            clean_transcript("Too short")

    def test_html_boundary_escapes_model_prompt(self):
        self.assertEqual(escape_html("<script>alert('x')</script>"), "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;")

    def test_serializes(self):
        data = json.loads(json.dumps(self.quiz.to_dict()))
        self.assertEqual(data["provider"], "Offline grounded generator")
        self.assertEqual(Quiz.from_dict(data).to_dict(), data)

    def test_srt_cues_and_question_timestamps_are_preserved(self):
        srt = """1
00:00:12,500 --> 00:00:18,000
Neural networks learn useful representations from examples.

2
00:00:18,000 --> 00:00:25,250
Optimization adjusts parameters to reduce a defined loss function.
"""
        parsed = parse_transcript(srt, "lesson.srt")
        self.assertEqual(parsed.segments[0].start, 12.5)
        self.assertEqual(parse_transcript(self.transcript, "lesson.txt").segments, [])
        timed = Transcription(self.transcript, [TranscriptSegment(37.5, 90.0, self.transcript)], "test")
        quiz = QuizService().generate_transcription(timed, count=2)
        self.assertTrue(all(question.timestamp == 37.5 for question in quiz.questions))

    def test_chunked_transcription_offsets_each_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "audio.wav"
            source.write_bytes(b"test")

            def chunker(_source, output, _minutes):
                return [Path(output) / "chunk-000.wav", Path(output) / "chunk-001.wav"]

            def transcribe_one(path, _provider, _model):
                index = int(path.stem.rsplit("-", 1)[1])
                return Transcription(f"part {index}", [TranscriptSegment(1.0, 4.0, f"part {index}")], "fake")

            result = transcribe(source, chunk_minutes=2, chunker=chunker, transcribe_one=transcribe_one)
        self.assertEqual([row.start for row in result.segments], [1.0, 121.0])
        self.assertIn("2 chunks", result.provider)

    def test_langchain_structured_runnable_is_injectable(self):
        evidence = self.transcript.split(". ", 1)[0] + "."

        class Runnable:
            def invoke(self, _prompt):
                return {
                    "summary": "A grounded summary.",
                    "key_concepts": ["learning"],
                    "questions": [
                        {
                            "prompt": f"Question {index}?",
                            "options": ["A", "B", "C", "D"],
                            "answer_index": 0,
                            "explanation": evidence,
                            "evidence": evidence,
                        }
                        for index in range(2)
                    ],
                }

        quiz = LangChainQuizGenerator(runnable=Runnable()).generate(self.transcript, "Test", count=2)
        self.assertEqual(len(quiz.questions), 2)
        self.assertIn("LangChain", quiz.provider)

    def test_openai_geval_uses_structured_scores(self):
        report_json = json.dumps(
            {"accuracy": 5, "relevance": 4, "clarity": 4.5, "grounding": 5, "feedback": ["Review distractor difficulty."]}
        )

        class Responses:
            def create(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(output_text=report_json)

        responses = Responses()
        report = OpenAIGEvalEvaluator(SimpleNamespace(responses=responses), model="test-evaluator").evaluate(self.quiz)
        self.assertEqual(report.overall, 4.62)
        self.assertEqual(responses.request["text"]["format"]["type"], "json_schema")
        self.assertEqual(report.model, "test-evaluator")

    def test_unknown_provider_is_not_silently_offline(self):
        with self.assertRaisesRegex(ValueError, "Unknown quiz provider"):
            QuizService().generate(self.transcript, provider="typo")


if __name__ == "__main__":
    unittest.main()
