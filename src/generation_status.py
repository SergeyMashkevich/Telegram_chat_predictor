from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.chat_context import get_active_chat
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


def load_env() -> None:
    load_dotenv(ENV_PATH)


def active_chat_id() -> int | None:
    chat = get_active_chat()

    if chat is None:
        return None

    return int(chat["chat_id"])


def active_adapter_path() -> Path | None:
    chat_id = active_chat_id()

    if chat_id is None:
        return None

    return ADAPTERS_DIR / "chats" / str(chat_id) / "lora" / "adapters.safetensors"


def active_metadata_path() -> Path | None:
    chat_id = active_chat_id()

    if chat_id is None:
        return None

    return ADAPTERS_DIR / "chats" / str(chat_id) / "lora" / "active_metadata.json"


def load_active_metadata() -> dict[str, Any] | None:
    path = active_metadata_path()

    if path is None or not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def describe_generation_status() -> dict[str, Any]:
    load_env()

    chat = get_active_chat()
    provider = os.getenv("CHAT_GENERATION_PROVIDER", "ollama").strip().lower()

    adapter_path = active_adapter_path()
    adapter_exists = bool(adapter_path and adapter_path.exists())

    metadata = load_active_metadata() or {}

    active_base_model = str(metadata.get("base_model") or "").strip()
    mlx_model = (
        active_base_model
        or os.getenv("MLX_CHAT_MODEL", "").strip()
        or os.getenv("TRAIN_BASE_MODEL", "").strip()
    )

    return {
        "chat": chat,
        "generation_provider": provider,
        "fallback_provider": os.getenv("GENERATION_FALLBACK_PROVIDER", ""),
        "fallback_on_error": os.getenv("GENERATION_FALLBACK_ON_ERROR", "true"),

        "mlx_model": mlx_model,
        "mlx_adapter_path": str(adapter_path) if adapter_path else "",
        "mlx_adapter_exists": adapter_exists,

        "ollama_embed_model": os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
        "ollama_chat_model": os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"),
    }


def print_generation_status() -> None:
    info = describe_generation_status()
    chat = info["chat"]

    print()
    print("[CLI] Generation status")
    print("=======================")

    if chat is None:
        print("chat:                not selected")
    else:
        print(f"chat:                {chat.get('title')}")
        print(f"chat_id:             {chat.get('chat_id')}")

    print()
    print(f"provider:            {info['generation_provider']}")
    print(f"fallback:            {info.get('fallback_provider') or 'not set'}")

    print()
    print("Generation")
    print("----------")

    if info["generation_provider"] == "mlx":
        print(f"base_model:          {info['mlx_model'] or 'not set'}")
        print(f"adapter_exists:      {info['mlx_adapter_exists']}")
        print(f"adapter_path:        {info['mlx_adapter_path'] or 'not available'}")

    elif info["generation_provider"] == "ollama":
        print(f"ollama_model:        {info['ollama_chat_model']}")

    else:
        print("provider_status:     unknown provider")

    print()
    print("Ranking")
    print("-------")
    print(f"embedding_model:     {info['ollama_embed_model']}")

    print()
    print("Status")
    print("------")

    if info["generation_provider"] == "mlx":
        if info["mlx_adapter_exists"]:
            print("OK: MLX base model + LoRA adapter will generate candidates")
        else:
            print("ERROR: provider=mlx but adapter is missing")

    elif info["generation_provider"] == "ollama":
        print("OK: Ollama chat model will generate candidates")

    else:
        print("ERROR: unknown generation provider")

    print()
