from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from video_quiz.media import extract_audio, parse_transcript, transcribe
from video_quiz.models import Quiz, Transcription
from video_quiz.presentation import escape_html
from video_quiz.service import QuizService


def generate_via_api(
    api_url: str,
    transcription: Transcription,
    title: str,
    count: int,
    difficulty: str,
    provider: str,
    evaluation_provider: str,
) -> Quiz:
    payload = json.dumps(
        {
            "transcript": transcription.text,
            "transcript_segments": [segment.__dict__ for segment in transcription.segments],
            "title": title,
            "count": count,
            "difficulty": difficulty,
            "provider": provider,
            "evaluation_provider": evaluation_provider,
        }
    ).encode()
    request = urllib.request.Request(api_url.rstrip("/") + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return Quiz.from_dict(json.load(response))
    except urllib.error.HTTPError as exc:
        detail = json.loads(exc.read().decode()).get("error", str(exc))
        raise RuntimeError(f"Quiz API rejected the request: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Quiz API is unavailable at {api_url}: {exc.reason}") from exc

st.set_page_config(page_title="FrameQuest", page_icon="▶", layout="wide")
st.markdown("""
<style>
.stApp{background:#071018;color:#e5f8ff}.hero{background:linear-gradient(120deg,#0f172a,#0e7490,#164e63);background-size:180% 180%;border:1px solid #22d3ee55;border-radius:26px;padding:2rem;margin-bottom:1rem;animation:frame-enter .55s ease-out both,frame-drift 10s ease-in-out infinite}.hero h1{font-size:3.5rem;margin:.1rem 0;color:#ecfeff}.signal{color:#67e8f9;letter-spacing:.14em;font-size:.8rem}[data-testid="stMetric"]{background:#102431;border:1px solid #1d4f61;border-radius:15px;padding:1rem}.quiz-card{background:#0d202c;border:1px solid #1d4f61;border-radius:18px;padding:1.2rem;margin:.8rem 0}
@keyframes frame-enter{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes frame-drift{50%{background-position:100% 50%;box-shadow:0 18px 52px #0891b229}}
@media (prefers-reduced-motion: reduce){.hero{animation:none!important;background-position:0 50%}}
</style><div class="hero"><div class="signal">WATCH LESS PASSIVELY. REMEMBER MORE.</div><h1>FrameQuest</h1><p>Turn any lesson transcript—or an optionally transcribed video—into an evidence-linked interactive quiz.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Quiz blueprint")
    title = st.text_input("Title", "Understanding the lesson")
    count = st.slider("Questions", 2, 12, 5)
    difficulty = st.select_slider("Difficulty", ["easy", "medium", "hard"], value="medium")
    question_engines = {"Offline grounded": "offline"}
    if os.getenv("OPENAI_API_KEY"):
        question_engines |= {"OpenAI direct": "openai", "LangChain + OpenAI": "langchain"}
    quiz_provider_label = st.selectbox("Question engine", list(question_engines))
    evaluation_engines = {"Deterministic gates": "deterministic"}
    if os.getenv("OPENAI_API_KEY"):
        evaluation_engines["OpenAI G-Eval"] = "openai"
    evaluation_label = st.selectbox("Quality evaluation", list(evaluation_engines))
    transcriber = st.selectbox("Audio transcription", ["faster-whisper", "openai"])
    whisper_model = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1)
    chunk_minutes = st.number_input(
        "Audio chunk size (minutes)",
        min_value=1,
        max_value=60,
        value=int(os.getenv("AUDIO_CHUNK_MINUTES", "10")),
    )
    backend = st.selectbox("Processing backend", ["In-process", "Flask API"])
    api_url = st.text_input("Flask API URL", "http://127.0.0.1:5000", disabled=backend != "Flask API")
    st.caption("Paste/upload a transcript for a zero-key workflow. Video and audio require an optional transcription stack.")

input_mode = st.radio("Start from", ["Included sample", "Paste transcript", "Upload transcript / media"], horizontal=True)
transcript_text = ""
uploaded = None
if input_mode == "Included sample":
    transcript_text = (ROOT / "sample_data" / "lesson.txt").read_text(encoding="utf-8")
    st.text_area("Sample transcript", transcript_text, height=180, disabled=True)
elif input_mode == "Paste transcript":
    transcript_text = st.text_area("Transcript", placeholder="Paste at least a few instructional paragraphs…", height=230)
else:
    uploaded = st.file_uploader("Lesson file", type=["txt", "vtt", "srt", "mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm"])

if st.button("Build the quiz", type="primary", use_container_width=True):
    try:
        if uploaded:
            suffix = Path(uploaded.name).suffix.lower()
            if suffix in {".txt", ".vtt", ".srt"}:
                transcription = parse_transcript(uploaded.getvalue().decode("utf-8"), uploaded.name)
            else:
                with tempfile.TemporaryDirectory() as folder:
                    media = Path(folder) / ("input" + suffix)
                    media.write_bytes(uploaded.getvalue())
                    if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
                        media = extract_audio(media, Path(folder) / "audio.wav")
                    transcription = transcribe(media, transcriber, whisper_model, chunk_minutes=int(chunk_minutes))
        else:
            transcription = parse_transcript(transcript_text, "transcript.txt")
        provider = question_engines[quiz_provider_label]
        evaluation_provider = evaluation_engines[evaluation_label]
        quiz = (
            generate_via_api(api_url, transcription, title, count, difficulty, provider, evaluation_provider)
            if backend == "Flask API"
            else QuizService().generate_transcription(
                transcription,
                title=title,
                count=count,
                difficulty=difficulty,
                provider=provider,
                evaluation_provider=evaluation_provider,
            )
        )
        st.session_state["quiz"] = quiz
        st.session_state["submitted"] = False
    except Exception as exc:
        st.error(str(exc))

if quiz := st.session_state.get("quiz"):
    q = quiz.quality
    metrics = st.columns(4)
    metrics[0].metric("Questions", len(quiz.questions))
    metrics[1].metric("Grounded", f"{q.grounding:.0%}")
    metrics[2].metric("Answerable", f"{q.answerability:.0%}")
    metrics[3].metric("Coverage", f"{q.coverage:.0%}")
    if quiz.geval:
        st.caption(f"Optional model review · {quiz.geval.model} · 1–5 rubric scores")
        review = st.columns(5)
        review[0].metric("Accuracy", f"{quiz.geval.accuracy:.1f}")
        review[1].metric("Relevance", f"{quiz.geval.relevance:.1f}")
        review[2].metric("Clarity", f"{quiz.geval.clarity:.1f}")
        review[3].metric("Grounding", f"{quiz.geval.grounding:.1f}")
        review[4].metric("G-Eval avg", f"{quiz.geval.overall:.1f}")
        for note in quiz.geval.feedback:
            st.info(note)
    st.subheader("Lesson in a minute")
    st.write(quiz.summary)
    st.caption("Key concepts · " + " · ".join(quiz.key_concepts))
    answers = []
    for question in quiz.questions:
        st.markdown(f"<div class='quiz-card'><strong>{escape_html(question.id)}. {escape_html(question.prompt)}</strong></div>", unsafe_allow_html=True)
        if question.timestamp is not None:
            minutes, seconds = divmod(int(question.timestamp), 60)
            st.caption(f"Evidence starts near {minutes:02d}:{seconds:02d}")
        choice = st.radio("Choose one", question.options, key=f"answer-{question.id}", index=None, label_visibility="collapsed")
        answers.append(question.options.index(choice) if choice in question.options else None)
        if st.session_state.get("submitted"):
            if answers[-1] == question.answer_index:
                st.success("Correct — " + question.explanation)
            else:
                st.error(f"Answer: {question.options[question.answer_index]} — {question.explanation}")
    if st.button("Check my answers", use_container_width=True):
        st.session_state["submitted"] = True
        st.rerun()
    if st.session_state.get("submitted"):
        score = sum(given == question.answer_index for given, question in zip(answers, quiz.questions))
        st.metric("Your score", f"{score} / {len(quiz.questions)}")
    st.download_button("Download quiz JSON", json.dumps(quiz.to_dict(), indent=2), "quiz.json", "application/json", use_container_width=True)
    with st.expander("Full transcript and grounding evidence"):
        if quiz.transcript_segments:
            st.dataframe(
                [
                    {"start_s": row.start, "end_s": row.end, "text": row.text}
                    for row in quiz.transcript_segments
                ],
                use_container_width=True,
                hide_index=True,
            )
        st.write(quiz.transcript)
