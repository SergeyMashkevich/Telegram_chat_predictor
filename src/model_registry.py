from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from dotenv import load_dotenv
from ollama import Client

from src.chat_context import get_active_chat
from src.storage import get_app_state, set_app_state
from src.telegram_client import ENV_PATH


def load_env() -> None:
    load_dotenv(ENV_PATH)


def ollama_host() -> str:
    load_env()
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()


def default_chat_model() -> str:
    load_env()

    return (
        os.getenv("DEFAULT_OLLAMA_CHAT_MODEL")
        or os.getenv("OLLAMA_CHAT_MODEL")
        or "qwen3:8b"
    ).strip()


def slugify_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return slug


def suggested_trained_model_name() -> str:
    chat = get_active_chat()

    if chat is None:
        return "telegram-chat-unknown"

    chat_id = int(chat["chat_id"])
    title = str(chat.get("title", "")).strip()
    slug = slugify_title(title)

    if slug:
        return f"telegram-chat-{slug}-{chat_id}"

    return f"telegram-chat-{chat_id}"


def ollama_client() -> Client:
    return Client(host=ollama_host())


def _model_name_from_item(item: Any) -> str | None:
    if isinstance(item, dict):
        value = item.get("model") or item.get("name")
        return str(value) if value else None

    value = getattr(item, "model", None) or getattr(item, "name", None)

    if value:
        return str(value)

    return None


def list_ollama_models() -> list[str]:
    response = ollama_client().list()

    models = getattr(response, "models", None)

    if models is None and isinstance(response, dict):
        models = response.get("models", [])

    result: list[str] = []

    for item in models or []:
        name = _model_name_from_item(item)

        if name:
            result.append(name)

    return sorted(set(result))


def model_exists(model_name: str) -> bool:
    try:
        return model_name in set(list_ollama_models())
    except Exception:
        return False


def get_chat_model_override() -> str | None:
    value = get_app_state("chat_model")

    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    return value


def set_chat_model_override(model_name: str) -> None:
    model_name = model_name.strip()

    if not model_name:
        raise ValueError("Имя модели не может быть пустым.")

    available = set(list_ollama_models())

    if model_name not in available:
        raise ValueError(
            f"Модель {model_name!r} не найдена в Ollama. "
            "Проверьте командой models или выполните ollama pull."
        )

    set_app_state("chat_model", model_name)


def clear_chat_model_override() -> None:
    set_app_state("chat_model", "")


def resolve_chat_model() -> str:
    """
    Приоритет:
    1. Ручная модель текущего чата из app_state.chat_model.
    2. Автообученная модель telegram-chat-... если она существует в Ollama.
    3. DEFAULT_OLLAMA_CHAT_MODEL.
    """
    override = get_chat_model_override()

    if override and model_exists(override):
        return override

    trained_name = suggested_trained_model_name()

    if model_exists(trained_name):
        return trained_name

    return default_chat_model()


def describe_model_selection() -> dict[str, Any]:
    chat = get_active_chat()
    override = get_chat_model_override()
    trained_name = suggested_trained_model_name()
    default_model = default_chat_model()

    try:
        available_models = list_ollama_models()
        available_error = None
    except Exception as error:
        available_models = []
        available_error = str(error)

    available_set = set(available_models)

    if override and override in available_set:
        resolved = override
        reason = "per-chat selected model"
    elif trained_name in available_set:
        resolved = trained_name
        reason = "per-chat trained model"
    else:
        resolved = default_model
        reason = "default fallback model"

    return {
        "chat": chat,
        "selected_model": override,
        "suggested_trained_model": trained_name,
        "trained_model_exists": trained_name in available_set,
        "default_model": default_model,
        "resolved_model": resolved,
        "reason": reason,
        "available_models": available_models,
        "available_error": available_error,
    }
