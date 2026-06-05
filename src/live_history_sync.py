from __future__ import annotations

import json
from typing import Any

from src.history_sync import get_chat_history
from src.message_normalizer import normalize_text_message
from src.result_store import write_result
from src.storage import get_app_state, initialize_database, set_app_state, upsert_messages
from src.telegram_client import TdlibClient


def full_name(user: dict[str, Any]) -> str:
    first_name = str(user.get("first_name", "")).strip()
    last_name = str(user.get("last_name", "")).strip()
    return f"{first_name} {last_name}".strip()


def sync_current_chat(
    client: TdlibClient,
    chat_id: int,
    limit: int,
) -> dict[str, Any]:
    target_user_id_raw = get_app_state("target_user_id")

    if target_user_id_raw is None:
        raise RuntimeError(
            "target_user_id не найден. Сначала выполните make select-chat."
        )

    target_user_id = int(target_user_id_raw)

    me = client.request({"@type": "getMe"})
    own_user_id = int(me["id"])
    own_name = full_name(me)

    chat = client.request(
        {
            "@type": "getChat",
            "chat_id": int(chat_id),
        }
    )

    other_name = str(chat.get("title", "")).strip() or str(target_user_id)

    raw_messages = get_chat_history(
        client=client,
        chat_id=int(chat_id),
        max_messages=int(limit),
    )

    normalized_messages: list[dict[str, Any]] = []

    for raw_message in raw_messages:
        normalized = normalize_text_message(
            message=raw_message,
            tdlib_chat_id=int(chat_id),
            own_user_id=own_user_id,
            own_name=own_name,
            other_user_id=target_user_id,
            other_name=other_name,
        )

        if normalized is not None:
            normalized_messages.append(normalized)

    initialize_database()
    upsert_messages(normalized_messages)

    set_app_state("target_user_id", str(target_user_id))
    set_app_state("target_tdlib_chat_id", str(chat_id))
    set_app_state("target_chat_title", other_name)

    result = {
        "name": other_name,
        "type": "personal_chat",
        "id": target_user_id,
        "tdlib_chat_id": int(chat_id),
        "messages": normalized_messages,
    }

    write_result(result)

    return {
        "chat_title": other_name,
        "raw_count": len(raw_messages),
        "normalized_count": len(normalized_messages),
    }
