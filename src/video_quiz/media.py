from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Callable

from .models import Transcription, TranscriptSegment


TIMING = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)"
)


def timestamp_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        hours, minutes, seconds = 0, int(parts[0]), float(parts[1])
    elif len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    else:
        raise ValueError(f"Invalid timestamp: {value}")
    return hours * 3600 + minutes * 60 + seconds


def parse_transcript(content: str, filename: str = "transcript.txt") -> Transcription:
    """Preserve real subtitle cues; plain text intentionally has no fake timing."""
    content = content.strip()
    if not content:
        raise ValueError("Transcript is empty")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".srt", ".vtt"}:
        return Transcription(content, [], "supplied text")
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines) if TIMING.search(line)), None)
        if timing_index is None:
            continue
        match = TIMING.search(lines[timing_index])
        text = " ".join(lines[timing_index + 1 :]).strip()
        if text:
            segments.append(TranscriptSegment(timestamp_seconds(match.group("start")), timestamp_seconds(match.group("end")), text))
    if not segments:
        raise ValueError("No valid subtitle cues were found")
    return Transcription(" ".join(segment.text for segment in segments), segments, suffix.lstrip(".").upper())


def split_audio(path: str | Path, output_dir: str | Path, chunk_minutes: int = 10) -> list[Path]:
    if not 1 <= chunk_minutes <= 60:
        raise ValueError("chunk_minutes must be between 1 and 60")
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("Install pydub and FFmpeg to split large audio files") from exc
    audio = AudioSegment.from_file(path)
    if len(audio) <= 0:
        raise ValueError("The audio file contains no decodable samples")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chunk_ms = chunk_minutes * 60 * 1000
    rows = []
    for index, start in enumerate(range(0, len(audio), chunk_ms)):
        destination = output / f"chunk-{index:03d}.wav"
        audio[start : start + chunk_ms].export(destination, format="wav")
        rows.append(destination)
    if not rows:
        raise ValueError("The audio file contains no decodable samples")
    return rows


def extract_audio(video: str | Path, destination: str | Path) -> Path:
    try:
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip
    except ImportError as exc:
        raise RuntimeError("Install moviepy and FFmpeg to extract video audio") from exc
    with VideoFileClip(str(video)) as clip:
        if clip.audio is None:
            raise ValueError("The uploaded video has no audio track")
        clip.audio.write_audiofile(str(destination), logger=None)
    return Path(destination)


def _value(row, name: str, default=None):
    return row.get(name, default) if isinstance(row, dict) else getattr(row, name, default)


def _transcribe_one(path: Path, provider: str, model: str) -> Transcription:
    if provider == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper for local transcription") from exc
        engine = WhisperModel(model, device=os.getenv("WHISPER_DEVICE", "cpu"), compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
        rows, _ = engine.transcribe(str(path), vad_filter=True)
        segments = [TranscriptSegment(float(row.start), float(row.end), row.text.strip()) for row in rows if row.text.strip()]
        return Transcription(" ".join(row.text for row in segments), segments, f"faster-whisper:{model}")
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use hosted Whisper") from exc
        with path.open("rb") as handle:
            response = OpenAI().audio.transcriptions.create(
                model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"),
                file=handle,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        segments = [
            TranscriptSegment(float(_value(row, "start", 0)), float(_value(row, "end", 0)), str(_value(row, "text", "")).strip())
            for row in (_value(response, "segments", []) or [])
            if str(_value(row, "text", "")).strip()
        ]
        text = str(_value(response, "text", "")).strip() or " ".join(row.text for row in segments)
        return Transcription(text, segments, f"openai:{os.getenv('OPENAI_TRANSCRIBE_MODEL', 'whisper-1')}")
    raise ValueError(f"Unknown transcription provider: {provider}")


def merge_transcriptions(parts: list[Transcription], chunk_seconds: float) -> Transcription:
    if not parts:
        raise ValueError("At least one transcription part is required")
    segments = []
    for index, part in enumerate(parts):
        offset = index * chunk_seconds
        segments.extend(TranscriptSegment(row.start + offset, row.end + offset, row.text) for row in part.segments)
    provider = parts[0].provider + (f" · {len(parts)} chunks" if len(parts) > 1 else " · 1 chunk")
    return Transcription(" ".join(part.text.strip() for part in parts if part.text.strip()), segments, provider)


def transcribe(
    path: str | Path,
    provider: str = "faster-whisper",
    model: str = "base",
    *,
    chunk_minutes: int = 10,
    chunker: Callable[[str | Path, str | Path, int], list[Path]] | None = None,
    transcribe_one: Callable[[Path, str, str], Transcription] | None = None,
) -> Transcription:
    """Chunk media, transcribe each part, and preserve absolute segment timestamps."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not 1 <= chunk_minutes <= 60:
        raise ValueError("chunk_minutes must be between 1 and 60")
    chunker = chunker or split_audio
    transcribe_one = transcribe_one or _transcribe_one
    with tempfile.TemporaryDirectory(prefix="framequest-chunks-") as folder:
        chunks = chunker(path, folder, chunk_minutes)
        parts = [transcribe_one(Path(chunk), provider, model) for chunk in chunks]
    result = merge_transcriptions(parts, chunk_minutes * 60)
    if not result.text.strip():
        raise ValueError("Transcription produced no text")
    return result
