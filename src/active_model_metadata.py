from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.chat_context import get_active_chat
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


def active_lora_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    return ADAPTERS_DIR / "chats" / str(active_chat["chat_id"]) / "lora"


def active_metadata_path() -> Path:
    return active_lora_dir() / "active_metadata.json"


def load_active_model_metadata() -> dict[str, Any] | None:
    path = active_metadata_path()

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_active_model_metadata(
    *,
    base_model: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    path = active_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "chat_title": active_chat["title"],
        "chat_id": int(active_chat["chat_id"]),
        "user_id": int(active_chat["user_id"]),
        "base_model": base_model,
        "source": source,
        "saved_at": time.strftime("%Y%m%d_%H%M%S"),
    }

    if extra:
        payload.update(extra)

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return path


def resolve_active_mlx_base_model() -> str:
    load_dotenv(ENV_PATH)

    metadata = load_active_model_metadata()

    if metadata:
        base_model = str(metadata.get("base_model", "")).strip()

        if base_model:
            return base_model

    return (
        os.getenv("MLX_CHAT_MODEL")
        or os.getenv("TRAIN_BASE_MODEL")
        or "mlx-community/Qwen2.5-7B-Instruct-4bit"
    ).strip()
