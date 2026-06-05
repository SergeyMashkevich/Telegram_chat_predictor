from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "state"
DATA_DIR = PROJECT_ROOT / "data"
ACTIVE_CHAT_PATH = STATE_DIR / "active_chat.json"


def safe_chat_id(chat_id: int | str) -> str:
    value = str(chat_id).strip()
    return re.sub(r"[^0-9A-Za-z_-]", "_", value)


def get_active_chat() -> dict[str, Any] | None:
    if not ACTIVE_CHAT_PATH.exists():
        return None

    return json.loads(
        ACTIVE_CHAT_PATH.read_text(encoding="utf-8")
    )


def set_active_chat(chat: dict[str, Any]) -> None:
    required = {"title", "user_id", "chat_id"}
    missing = required - set(chat)

    if missing:
        raise ValueError(
            "Не хватает полей active chat: "
            + ", ".join(sorted(missing))
        )

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": str(chat["title"]),
        "user_id": int(chat["user_id"]),
        "chat_id": int(chat["chat_id"]),
    }

    ACTIVE_CHAT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def require_active_chat() -> dict[str, Any]:
    chat = get_active_chat()

    if chat is None:
        raise RuntimeError(
            "Чат не выбран. Запустите make run или make select-chat."
        )

    return chat


def chat_state_dir() -> Path:
    chat = require_active_chat()
    return STATE_DIR / "chats" / safe_chat_id(chat["chat_id"])


def chat_data_dir() -> Path:
    chat = require_active_chat()
    return DATA_DIR / "chats" / safe_chat_id(chat["chat_id"])


def chat_db_path() -> Path:
    return chat_state_dir() / "app.db"


def chat_result_path() -> Path:
    return chat_data_dir() / "result.json"
