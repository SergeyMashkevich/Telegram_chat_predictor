from __future__ import annotations

import json
from typing import Any, Iterable

from src.chat_context import chat_result_path


def load_result() -> dict[str, Any]:
    result_path = chat_result_path()

    if not result_path.exists():
        return {
            "name": "",
            "type": "personal_chat",
            "id": 0,
            "tdlib_chat_id": 0,
            "messages": [],
        }

    return json.loads(
        result_path.read_text(encoding="utf-8")
    )


def write_result(result: dict[str, Any]) -> None:
    result_path = chat_result_path()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = result_path.with_name(
        f"{result_path.name}.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(result_path)


def upsert_result_message(
    message: dict[str, Any],
    chat_name: str,
    target_user_id: int,
    tdlib_chat_id: int,
    max_messages: int,
) -> None:
    result = load_result()

    messages_by_id = {
        int(existing["id"]): existing
        for existing in result.get("messages", [])
    }

    messages_by_id[int(message["id"])] = message

    messages = sorted(
        messages_by_id.values(),
        key=lambda item: int(item["id"]),
    )

    if max_messages > 0:
        messages = messages[-max_messages:]

    result.update(
        {
            "name": chat_name,
            "type": "personal_chat",
            "id": target_user_id,
            "tdlib_chat_id": tdlib_chat_id,
            "messages": messages,
        }
    )

    write_result(result)


def delete_result_messages(message_ids: Iterable[int]) -> None:
    ids = {int(message_id) for message_id in message_ids}
    result_path = chat_result_path()

    if not ids or not result_path.exists():
        return

    result = load_result()

    result["messages"] = [
        message
        for message in result.get("messages", [])
        if int(message["id"]) not in ids
    ]

    write_result(result)
