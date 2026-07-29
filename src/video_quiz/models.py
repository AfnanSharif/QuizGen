from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Transcript segment timestamps must be ordered and non-negative")
        if not self.text.strip():
            raise ValueError("Transcript segment text cannot be empty")


@dataclass(frozen=True)
class Transcription:
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    provider: str = "text"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QuizQuestion:
    id: int
    prompt: str
    options: list[str]
    answer_index: int
    explanation: str
    evidence: str
    difficulty: str = "medium"
    timestamp: float | None = None


@dataclass
class QualityReport:
    grounding: float
    answerability: float
    coverage: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GEvalReport:
    accuracy: float
    relevance: float
    clarity: float
    grounding: float
    overall: float
    feedback: list[str]
    model: str
    rubric: str = "G-Eval-style model rubric; scores are 1–5 and require human review"


@dataclass
class Quiz:
    title: str
    summary: str
    key_concepts: list[str]
    transcript: str
    questions: list[QuizQuestion]
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)
    quality: QualityReport | None = None
    geval: GEvalReport | None = None
    provider: str = "Offline grounded generator"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Quiz":
        quality = QualityReport(**data["quality"]) if data.get("quality") else None
        geval = GEvalReport(**data["geval"]) if data.get("geval") else None
        questions = [QuizQuestion(**item) for item in data.get("questions", [])]
        segments = [TranscriptSegment(**item) for item in data.get("transcript_segments", [])]
        return cls(
            title=data["title"],
            summary=data["summary"],
            key_concepts=list(data.get("key_concepts", [])),
            transcript=data["transcript"],
            questions=questions,
            transcript_segments=segments,
            quality=quality,
            geval=geval,
            provider=data.get("provider", "Unknown provider"),
        )
