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
    chat = get_active_chat()

    if chat is None:
        raise RuntimeError(
            "No chat selected. Run make run or make select-chat first."
        )

    return ADAPTERS_DIR / "chats" / str(chat["chat_id"]) / "lora"


def active_adapter_path() -> Path:
    return active_lora_dir() / "adapters.safetensors"


def active_metadata_path() -> Path:
    return active_lora_dir() / "active_metadata.json"


def read_active_adapter_metadata() -> dict[str, Any] | None:
    path = active_metadata_path()

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_active_adapter_metadata(
    *,
    base_model: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    chat = get_active_chat()

    if chat is None:
        raise RuntimeError(
            "No chat selected. Run make run or make select-chat first."
        )

    adapter_path = active_adapter_path()

    if not adapter_path.exists():
        raise RuntimeError(f"Active adapter not found: {adapter_path}")

    payload: dict[str, Any] = {
        "chat_title": chat.get("title"),
        "chat_id": int(chat["chat_id"]),
        "user_id": int(chat["user_id"]),
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "source": source,
        "marked_at": time.strftime("%Y%m%d_%H%M%S"),
    }

    if extra:
        payload.update(extra)

    metadata_path = active_metadata_path()
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata_path


def resolve_active_mlx_model() -> str:
    load_dotenv(ENV_PATH)

    metadata = read_active_adapter_metadata()

    if metadata:
        base_model = str(metadata.get("base_model", "")).strip()

        if base_model:
            return base_model

    return (
        os.getenv("MLX_CHAT_MODEL")
        or os.getenv("TRAIN_BASE_MODEL")
        or "mlx-community/Qwen2.5-7B-Instruct-4bit"
    ).strip()
