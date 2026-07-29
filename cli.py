from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from video_quiz.media import extract_audio, parse_transcript, transcribe
from video_quiz.service import QuizService


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded quiz from a transcript, audio file, or video")
    parser.add_argument("input", type=Path, help="TXT/VTT/SRT transcript, audio, or video file")
    parser.add_argument("--title", default="Learning check")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--provider", choices=["offline", "openai", "langchain"], default="offline")
    parser.add_argument("--evaluation", choices=["deterministic", "openai"], default="deterministic")
    parser.add_argument("--transcriber", choices=["faster-whisper", "openai"], default="faster-whisper")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--chunk-minutes", type=int, default=int(os.getenv("AUDIO_CHUNK_MINUTES", "10")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = args.input.suffix.lower()
    if suffix in {".txt", ".vtt", ".srt"}:
        transcription = parse_transcript(args.input.read_text(encoding="utf-8"), args.input.name)
    else:
        with tempfile.TemporaryDirectory(prefix="framequest-cli-") as folder:
            media = args.input
            if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
                media = extract_audio(media, Path(folder) / "audio.wav")
            transcription = transcribe(media, args.transcriber, args.whisper_model, chunk_minutes=args.chunk_minutes)
    quiz = QuizService().generate_transcription(
        transcription,
        title=args.title,
        count=args.count,
        difficulty=args.difficulty,
        provider=args.provider,
        evaluation_provider=args.evaluation,
    )
    payload = json.dumps(quiz.to_dict(), indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"Created {args.output.resolve()}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
