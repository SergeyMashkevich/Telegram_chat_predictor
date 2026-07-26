from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.telegram_client import ENV_PATH


SUPPORTED_PROVIDERS = {"ollama", "mlx"}


def bool_env(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def configured_generation_provider() -> str:
    provider = os.getenv(
        "CHAT_GENERATION_PROVIDER",
        "auto",
    ).strip().lower()

    return provider or "auto"


def mlx_adapter_available() -> bool:
    from src.adapter_metadata import active_adapter_path

    try:
        return active_adapter_path().is_file()
    except RuntimeError:
        return False


def resolve_generation_provider(
    configured_provider: str | None = None,
) -> tuple[str, str]:
    provider = (
        configured_provider
        if configured_provider is not None
        else configured_generation_provider()
    ).strip().lower()

    if provider == "auto":
        if mlx_adapter_available():
            return "mlx", "active chat has a trained MLX adapter"

        return "ollama", "active chat has no trained MLX adapter"

    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"Unknown generation provider: {provider!r}. "
            "Use auto, ollama, or mlx."
        )

    return provider, "explicit configuration"


def create_raw_predictor(provider: str) -> Any:
    provider = provider.strip().lower()

    if provider == "mlx":
        from src.mlx_predictor import MLXPredictor

        return MLXPredictor()

    if provider == "ollama":
        from src.predictor import OllamaPredictor

        return OllamaPredictor()

    raise RuntimeError(
        f"Unknown generation provider: {provider!r}. "
        "Use ollama or mlx."
    )


def create_predictor() -> Any:
    load_dotenv(ENV_PATH)

    configured_provider = configured_generation_provider()
    provider, _ = resolve_generation_provider(configured_provider)

    fallback_provider = os.getenv(
        "GENERATION_FALLBACK_PROVIDER",
        "",
    ).strip().lower()

    if (
        fallback_provider
        and fallback_provider not in SUPPORTED_PROVIDERS
    ):
        raise RuntimeError(
            f"Unknown fallback provider: {fallback_provider!r}. "
            "Use ollama or mlx."
        )

    fallback_enabled = bool_env(
        "GENERATION_FALLBACK_ON_ERROR",
        "true",
    )

    fallback_available = bool(
        fallback_enabled
        and fallback_provider
        and fallback_provider != provider
    )

    try:
        primary = create_raw_predictor(provider)
    except Exception as primary_error:
        if not fallback_available:
            raise

        print(
            f"[generation fallback] Could not start {provider}: {primary_error}",
            flush=True,
        )
        print(
            f"[generation fallback] Starting {fallback_provider} instead...",
            flush=True,
        )

        return create_raw_predictor(fallback_provider)

    if fallback_available:
        from src.fallback_predictor import FallbackPredictor

        fallback = create_raw_predictor(fallback_provider)

        return FallbackPredictor(
            primary=primary,
            fallback=fallback,
            fallback_enabled=True,
        )

    return primary
