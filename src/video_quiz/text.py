from __future__ import annotations

import html
import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOP = {"about", "after", "again", "also", "and", "are", "because", "before", "being", "between", "both", "but", "can", "could", "does", "each", "for", "from", "have", "into", "just", "more", "most", "other", "over", "such", "than", "that", "their", "then", "there", "these", "they", "this", "through", "under", "very", "was", "were", "when", "where", "which", "while", "will", "with", "would", "your"}


def clean_transcript(text: str) -> str:
    text = html.unescape(text.replace("\ufeff", ""))
    text = re.sub(r"WEBVTT|\d+\s*\n", " ", text)
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?[.,]\d+\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?[.,]\d+", " ", text)
    text = re.sub(r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 80:
        raise ValueError("Transcript is too short; provide at least 80 characters of instructional content")
    return text


def sentences(text: str) -> list[str]:
    return [row.strip() for row in re.split(r"(?<=[.!?])\s+", text) if len(row.strip().split()) >= 5]


def concepts(text: str, limit: int = 12) -> list[str]:
    words = [word.lower() for word in _WORD.findall(text) if word.lower() not in _STOP and len(word) >= 5]
    counts = Counter(words)
    # Prefer repeated instructional terms, then longer distinct terms.
    ranked = sorted(counts, key=lambda word: (counts[word], len(word), word), reverse=True)
    return [word for word in ranked[:limit]]


def summarize(text: str, limit: int = 4) -> tuple[str, list[str]]:
    rows = sentences(text)
    if not rows:
        raise ValueError("Transcript does not contain complete instructional sentences")
    important = Counter(concepts(text, 30))
    scored = []
    for index, row in enumerate(rows):
        score = sum(important[word.lower()] for word in _WORD.findall(row)) / max(1, len(row.split()))
        if re.search(r"\b(key|important|means|defined|because|therefore|result)\b", row, re.I):
            score += 1
        scored.append((score, index, row))
    selected = sorted(sorted(scored, reverse=True)[:limit], key=lambda item: item[1])
    return " ".join(row for _, _, row in selected), concepts(text)
