<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=00c6ff,0072ff&height=200&section=header&text=QuizGen&fontSize=70&fontColor=ffffff&animation=twinkling" width="100%" />

<img src="https://img.icons8.com/?id=44010&format=png&size=100" width="90" />

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=2500&pause=1000&color=00c6ff&center=true&vCenter=true&width=600&height=50&lines=Lecture+Video+to+Quiz;Whisper+Transcription+%2B+GPT;Streamlit+%2B+Flask+%2B+CLI" alt="Typing SVG" />

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=for-the-badge&logo=flask&logoColor=white)](#)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o+%2F+Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)](#)
[![License](https://img.shields.io/github/license/AfnanSharif/QuizGen?style=for-the-badge&color=yellow)](LICENSE)

</div>

---

## 📖 Overview

QuizGen turns a lecture video, audio file, or transcript into a quiz. It extracts audio,
transcribes it (Whisper, local via `faster-whisper` or OpenAI's API), then generates
questions with an LLM. Ships three ways to use it: a **Streamlit** web app, a **Flask**
API, and a **CLI**.

Works with transcript files with no API key at all; audio/video transcription and
AI-generated questions need an `OPENAI_API_KEY`.

## 🏗️ Project Layout

```
QuizGen/
├── app.py                 # Streamlit web UI
├── api.py                 # Flask API
├── cli.py                 # Command-line interface
├── src/video_quiz/         # Core: media extraction, transcription, quiz generation
├── config/settings.yaml    # App configuration
├── sample_data/lesson.txt  # Example transcript to try things out
├── requirements.txt         # Core deps (streamlit, flask, python-dotenv)
├── requirements-ai.txt      # + openai, faster-whisper, moviepy, langchain
└── scripts/setup.sh         # venv + install helper (macOS/Linux)
```

## ⚡ Setup & Run

### 🪟 Windows (PowerShell / CMD)
```cmd
git clone https://github.com/AfnanSharif/QuizGen.git
cd QuizGen

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-ai.txt

copy .env.example .env
:: edit .env and add OPENAI_API_KEY to enable transcription + AI question generation

streamlit run app.py
```

### 🍎 macOS / 🐧 Linux
```bash
git clone https://github.com/AfnanSharif/QuizGen.git
cd QuizGen

./scripts/setup.sh                 # creates .venv and installs requirements.txt
source .venv/bin/activate
pip install -r requirements-ai.txt  # optional: adds Whisper/OpenAI/moviepy support

cp .env.example .env
# edit .env and add OPENAI_API_KEY to enable transcription + AI question generation

streamlit run app.py
```

Open **http://localhost:8501**.

### Alternative entry points
```bash
python api.py                              # Flask API instead of the Streamlit UI
python cli.py sample_data/lesson.txt        # generate a quiz straight from the CLI
make test                                   # run the unittest suite
```

---

<div align="center">

**Created by [AfnanSharif](https://github.com/AfnanSharif)** · ⭐ star this repo if it helped you

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=00c6ff,0072ff&height=80&section=footer" width="100%" />

</div>
