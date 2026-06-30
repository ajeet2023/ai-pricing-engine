from __future__ import annotations

import os
from typing import Any

import pandas as pd

POSITIVE_TERMS = ["great", "excellent", "love", "good", "satisfied", "fast", "fair", "perfect", "amazing", "happy"]
NEGATIVE_TERMS = ["bad", "slow", "poor", "damaged", "late", "issue", "problem", "disappointed", "hate", "expensive"]


def _heuristic_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0

    lowered = text.lower()
    positive_hits = sum(1 for term in POSITIVE_TERMS if term in lowered)
    negative_hits = sum(1 for term in NEGATIVE_TERMS if term in lowered)
    score = positive_hits - negative_hits
    if "not" in lowered and positive_hits:
        score -= 0.5
    return max(-1.0, min(1.0, score / 4.0))


def analyze_reviews(reviews: Any, api_key: str | None = None) -> dict[str, Any]:
    if isinstance(reviews, pd.Series):
        reviews = reviews.tolist()
    elif isinstance(reviews, str):
        reviews = [reviews]

    if not reviews:
        return {"sentiment_score": 0.0, "sentiment_label": "neutral", "sentiment_vector": [0.0, 0.0, 0.0], "summary": "No review content was available to analyze."}

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = "Classify the following customer reviews as positive, neutral, or negative and return a JSON array with one label per review.\n" + "\n".join(str(r) for r in reviews[:10])
            response = client.responses.create(model="gpt-4o-mini", input=prompt)
            text = response.output_text.strip()
            labels = [label.strip().lower() for label in text.splitlines() if label.strip()]
            if labels:
                score = sum(1 if label == "positive" else -1 if label == "negative" else 0 for label in labels) / max(len(labels), 1)
                label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
                summary = "Customers are leaning positive about the offer" if label == "positive" else "Customers are leaning negative; pricing or service may need attention" if label == "negative" else "Customer sentiment is mixed and should be monitored"
                return {
                    "sentiment_score": round(score, 3),
                    "sentiment_label": label,
                    "sentiment_vector": [max(0.0, score), 1 - abs(score), max(0.0, -score)],
                    "summary": summary,
                }
        except Exception:
            pass

    scores = [_heuristic_sentiment(str(review)) for review in reviews]
    average_score = sum(scores) / len(scores)
    label = "positive" if average_score > 0.2 else "negative" if average_score < -0.2 else "neutral"
    summary = "Customers are leaning positive about the offer" if label == "positive" else "Customers are leaning negative; pricing or service may need attention" if label == "negative" else "Customer sentiment is mixed and should be monitored"
    return {
        "sentiment_score": round(average_score, 3),
        "sentiment_label": label,
        "sentiment_vector": [max(0.0, average_score), 1 - abs(average_score), max(0.0, -average_score)],
        "summary": summary,
    }
