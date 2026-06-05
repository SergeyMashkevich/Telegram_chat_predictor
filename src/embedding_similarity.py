from __future__ import annotations

import math
import os
from typing import Any

from dotenv import load_dotenv
from ollama import Client

from src.telegram_client import ENV_PATH


def require_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"Переменная {name} не указана в .env")

    return value.strip()


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding-векторы имеют разную длину.")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


class OllamaEmbeddingSimilarity:
    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        self.model = require_env("OLLAMA_EMBED_MODEL")
        self.client = Client(host=require_env("OLLAMA_HOST"))

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        clean_texts = [
            str(text).strip()
            for text in texts
        ]

        if not clean_texts:
            return []

        try:
            response = self.client.embed(
                model=self.model,
                input=clean_texts,
            )

            embeddings = getattr(response, "embeddings", None)

            if embeddings is None and isinstance(response, dict):
                embeddings = response.get("embeddings")

            if embeddings is not None:
                return [
                    [float(value) for value in vector]
                    for vector in embeddings
                ]

        except Exception:
            pass

        # Совместимость со старым endpoint /api/embeddings.
        vectors: list[list[float]] = []

        for text in clean_texts:
            response = self.client.embeddings(
                model=self.model,
                prompt=text,
            )

            embedding = getattr(response, "embedding", None)

            if embedding is None and isinstance(response, dict):
                embedding = response.get("embedding")

            if embedding is None:
                raise RuntimeError(
                    "Ollama не вернула embedding-вектор."
                )

            vectors.append(
                [float(value) for value in embedding]
            )

        return vectors

    def similarity(
        self,
        left_text: str,
        right_text: str,
    ) -> float:
        left_vector, right_vector = self.embed_texts(
            [left_text, right_text]
        )

        return cosine_similarity(
            left_vector,
            right_vector,
        )
