from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.telegram_client import ENV_PATH


def bool_env(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def create_raw_predictor(provider: str) -> Any:
    provider = provider.strip().lower()

    if provider == "mlx":
        from src.mlx_predictor import MLXPredictor

        return MLXPredictor()

    if provider == "ollama":
        from src.predictor import OllamaPredictor

        return OllamaPredictor()

    raise RuntimeError(
        f"Неизвестный generation provider: {provider!r}. "
        "Используйте ollama или mlx."
    )


def create_predictor() -> Any:
    load_dotenv(ENV_PATH)

    provider = os.getenv("CHAT_GENERATION_PROVIDER", "ollama").strip().lower()
    primary = create_raw_predictor(provider)

    fallback_provider = os.getenv(
        "GENERATION_FALLBACK_PROVIDER",
        "",
    ).strip().lower()

    fallback_enabled = bool_env(
        "GENERATION_FALLBACK_ON_ERROR",
        "true",
    )

    if (
        fallback_enabled
        and fallback_provider
        and fallback_provider != provider
    ):
        from src.fallback_predictor import FallbackPredictor

        fallback = create_raw_predictor(fallback_provider)

        return FallbackPredictor(
            primary=primary,
            fallback=fallback,
            fallback_enabled=True,
        )

    return primary
