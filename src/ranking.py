from __future__ import annotations

import math
import os
from statistics import median
from typing import Any

from dotenv import load_dotenv

from src.embedding_similarity import (
    OllamaEmbeddingSimilarity,
    cosine_similarity,
)
from src.telegram_client import ENV_PATH


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def join_messages(messages: list[str]) -> str:
    return "\n".join(
        str(message).strip()
        for message in messages
        if str(message).strip()
    )


def message_text(message: dict[str, Any]) -> str:
    return str(message.get("text", "")).strip()


def incoming_block_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        message_text(message)
        for message in messages
        if message_text(message)
    )


def outgoing_style_examples(
    history_messages: list[dict[str, Any]],
    limit: int = 40,
) -> list[str]:
    examples = [
        message_text(message)
        for message in history_messages
        if bool(message.get("is_outgoing")) and message_text(message)
    ]

    return examples[-limit:]


def average_top_k(values: list[float], k: int = 5) -> float:
    if not values:
        return 0.5

    top = sorted(values, reverse=True)[:k]

    return sum(top) / len(top)


class CandidateRanker:
    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        self.style_weight = float(os.getenv("RANK_STYLE_WEIGHT", "0.35"))
        self.relevance_weight = float(os.getenv("RANK_RELEVANCE_WEIGHT", "0.35"))
        self.length_weight = float(os.getenv("RANK_LENGTH_WEIGHT", "0.10"))
        self.naturalness_weight = float(os.getenv("RANK_NATURALNESS_WEIGHT", "0.20"))

        self.embeddings = OllamaEmbeddingSimilarity()

    def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        history_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        candidate_texts = [
            join_messages(candidate["messages"])
            for candidate in candidates
        ]

        incoming_text = incoming_block_text(incoming_messages)
        style_examples = outgoing_style_examples(history_messages)

        texts_to_embed = candidate_texts + [incoming_text] + style_examples
        vectors = self.embeddings.embed_texts(texts_to_embed)

        candidate_vectors = vectors[:len(candidate_texts)]
        incoming_vector = vectors[len(candidate_texts)]
        style_vectors = vectors[len(candidate_texts) + 1:]

        usual_lengths = [
            len(example)
            for example in style_examples
            if example
        ]

        usual_length = median(usual_lengths) if usual_lengths else 35

        ranked: list[dict[str, Any]] = []

        for candidate, text, vector in zip(
            candidates,
            candidate_texts,
            candidate_vectors,
        ):
            relevance_score = clamp(
                (cosine_similarity(vector, incoming_vector) + 1.0) / 2.0
            )

            if style_vectors:
                style_similarities = [
                    clamp((cosine_similarity(vector, style_vector) + 1.0) / 2.0)
                    for style_vector in style_vectors
                ]
                style_score = average_top_k(style_similarities, k=5)
            else:
                style_score = 0.5

            length_score = self._length_score(
                candidate_length=len(text),
                usual_length=float(usual_length),
            )

            naturalness_score = self._naturalness_score(
                candidate["messages"]
            )

            final_score = (
                self.style_weight * style_score
                + self.relevance_weight * relevance_score
                + self.length_weight * length_score
                + self.naturalness_weight * naturalness_score
            )

            enriched = dict(candidate)
            enriched["style_score"] = round(style_score, 4)
            enriched["relevance_score"] = round(relevance_score, 4)
            enriched["length_score"] = round(length_score, 4)
            enriched["naturalness_score"] = round(naturalness_score, 4)
            enriched["final_score"] = round(clamp(final_score), 4)

            ranked.append(enriched)

        return sorted(
            ranked,
            key=lambda candidate: candidate["final_score"],
            reverse=True,
        )

    def _length_score(
        self,
        candidate_length: int,
        usual_length: float,
    ) -> float:
        if candidate_length <= 0:
            return 0.0

        usual_length = max(1.0, usual_length)
        ratio = (candidate_length + 1.0) / (usual_length + 1.0)

        return clamp(math.exp(-abs(math.log(ratio))))

    def _naturalness_score(
        self,
        messages: list[str],
    ) -> float:
        clean_messages = [
            str(message).strip()
            for message in messages
            if str(message).strip()
        ]

        if not clean_messages:
            return 0.0

        score = 1.0

        if len(set(clean_messages)) != len(clean_messages):
            score -= 0.25

        joined = " ".join(clean_messages)

        if len(joined) <= 1:
            score -= 0.35

        if len(clean_messages) >= 3:
            very_short = [
                message
                for message in clean_messages
                if len(message) <= 3
            ]

            if len(very_short) == len(clean_messages):
                score -= 0.25

        if any(len(message) > 500 for message in clean_messages):
            score -= 0.20

        return clamp(score)
