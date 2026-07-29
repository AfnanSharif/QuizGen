from __future__ import annotations

import os
import tempfile
import sys
from pathlib import Path

from flask import Flask, jsonify, request

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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
service = QuizService()


@app.get("/health")
def health():
    return {"status": "ok", "offline_ready": True}


@app.post("/api/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    try:
        quiz = service.generate(
            payload.get("transcript", ""),
            payload.get("title", "Learning check"),
            int(payload.get("count", 5)),
            payload.get("difficulty", "medium"),
            payload.get("provider", "offline"),
            int(payload.get("seed", 42)),
            transcript_segments=payload.get("transcript_segments", []),
            evaluation_provider=payload.get("evaluation_provider", "deterministic"),
        )
        return jsonify(quiz.to_dict())
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 422


@app.post("/api/upload")
def upload():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "A file is required"}), 400
    suffix = Path(uploaded.filename).suffix.lower()
    try:
        if suffix in {".txt", ".vtt", ".srt"}:
            transcription = parse_transcript(uploaded.read().decode("utf-8"), uploaded.filename)
        else:
            with tempfile.TemporaryDirectory() as folder:
                media_path = Path(folder) / ("input" + suffix)
                uploaded.save(media_path)
                if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
                    media_path = extract_audio(media_path, Path(folder) / "audio.wav")
                transcription = transcribe(
                    media_path,
                    request.form.get("transcriber", "faster-whisper"),
                    request.form.get("whisper_model", "base"),
                    chunk_minutes=int(request.form.get("chunk_minutes", os.getenv("AUDIO_CHUNK_MINUTES", "10"))),
                )
        quiz = service.generate_transcription(
            transcription,
            title=request.form.get("title", Path(uploaded.filename).stem),
            count=int(request.form.get("count", 5)),
            difficulty=request.form.get("difficulty", "medium"),
            provider=request.form.get("provider", "offline"),
            evaluation_provider=request.form.get("evaluation_provider", "deterministic"),
        )
        return jsonify(quiz.to_dict())
    except (UnicodeDecodeError, ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 422


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
