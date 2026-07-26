from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.chat_context import chat_result_path

from src.message_normalizer import normalize_text_message
from src.storage import get_app_state, initialize_database, set_app_state, upsert_messages
from src.telegram_client import (
    ENV_PATH,
    PROJECT_ROOT,
    TdlibClient,
    TdlibRequestError,
)




def require_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"Variable {name} is not set in .env")

    return value.strip()


def full_name(user: dict[str, Any]) -> str:
    first_name = str(user.get("first_name", "")).strip()
    last_name = str(user.get("last_name", "")).strip()

    return f"{first_name} {last_name}".strip()


def ensure_authorized(client: TdlibClient) -> None:
    for _ in range(10):
        state = client.request(
            {
                "@type": "getAuthorizationState",
            }
        )

        state_type = state.get("@type")

        if state_type == "authorizationStateReady":
            return

        if state_type == "authorizationStateWaitTdlibParameters":
            client.set_tdlib_parameters()
            continue

        if state_type == "authorizationStateWaitEncryptionKey":
            client.request(client.build_database_encryption_key_request())
            continue

        raise RuntimeError(
            "The TDLib session is not ready. "
            f"Current state: {state_type}. "
            "Run python -m src.authorize first."
        )

    raise RuntimeError("Timed out waiting for the TDLib session to become ready")


def load_more_chats(
    client: TdlibClient,
    chat_list: dict[str, Any] | None,
) -> bool:
    try:
        client.request(
            {
                "@type": "loadChats",
                "chat_list": chat_list,
                "limit": 100,
            },
            timeout=60.0,
        )

        return True

    except TdlibRequestError as error:
        if error.code == 404:
            return False

        raise


def find_private_chat(
    client: TdlibClient,
    target_user_id: int,
) -> dict[str, Any]:
    chat_lists: list[tuple[str, dict[str, Any] | None]] = [
        ("main list", None),
        ("archive", {"@type": "chatListArchive"}),
    ]

    for list_name, chat_list in chat_lists:
        print(f"Searching for the chat in the {list_name}...")

        while True:
            loaded_more = load_more_chats(
                client=client,
                chat_list=chat_list,
            )

            chats = client.request(
                {
                    "@type": "getChats",
                    "chat_list": chat_list,
                    "limit": 10000,
                },
                timeout=60.0,
            )

            for chat_id in chats.get("chat_ids", []):
                chat = client.request(
                    {
                        "@type": "getChat",
                        "chat_id": int(chat_id),
                    }
                )

                chat_type = chat.get("type", {})

                if (
                    chat_type.get("@type") == "chatTypePrivate"
                    and int(chat_type.get("user_id", 0)) == target_user_id
                ):
                    return chat

            if not loaded_more:
                break

    raise RuntimeError(
        "A private chat with TARGET_USER_ID was not found in the main "
        "or archived Telegram chats."
    )


def get_chat_history(
    client: TdlibClient,
    chat_id: int,
    max_messages: int,
) -> list[dict[str, Any]]:
    collected: dict[int, dict[str, Any]] = {}
    from_message_id = 0

    while len(collected) < max_messages:
        remaining = max_messages - len(collected)
        request_limit = min(100, remaining)

        response = client.request(
            {
                "@type": "getChatHistory",
                "chat_id": chat_id,
                "from_message_id": from_message_id,
                "offset": 0,
                "limit": request_limit,
                "only_local": False,
            },
            timeout=60.0,
        )

        page = response.get("messages", [])

        if not page:
            break

        previous_count = len(collected)

        for message in page:
            collected[int(message["id"])] = message

        next_from_message_id = int(page[-1]["id"])

        if (
            next_from_message_id == from_message_id
            or len(collected) == previous_count
        ):
            break

        from_message_id = next_from_message_id

        print(f"Messages received: {len(collected)} / {max_messages}")

    return sorted(
        collected.values(),
        key=lambda message: int(message["id"]),
    )


def close_client(client: TdlibClient) -> None:
    client.send({"@type": "close"})

    deadline = time.monotonic() + 10.0

    while time.monotonic() < deadline:
        event = client.receive(timeout=1.0)

        if event is None:
            continue

        if event.get("@type") != "updateAuthorizationState":
            continue

        state = event.get("authorization_state", {})

        if state.get("@type") == "authorizationStateClosed":
            return


def main() -> None:
    load_dotenv(ENV_PATH)

    target_user_id_raw = get_app_state("target_user_id") or os.getenv("TARGET_USER_ID")

    if target_user_id_raw is None or not str(target_user_id_raw).strip():
        raise RuntimeError(
            "No chat selected. Run make select-chat or make run."
        )

    target_user_id = int(target_user_id_raw)

    target_chat_id_raw = (
        get_app_state("target_tdlib_chat_id")
        or os.getenv("TARGET_TDLIB_CHAT_ID")
    )

    history_limit = int(os.getenv("HISTORY_SYNC_LIMIT", "1000"))

    client = TdlibClient()

    try:
        ensure_authorized(client)

        me = client.request({"@type": "getMe"})
        own_user_id = int(me["id"])
        own_name = full_name(me)

        if target_chat_id_raw is not None and str(target_chat_id_raw).strip():
            tdlib_chat_id = int(target_chat_id_raw)

            chat = client.request(
                {
                    "@type": "getChat",
                    "chat_id": tdlib_chat_id,
                }
            )
        else:
            chat = find_private_chat(
                client=client,
                target_user_id=target_user_id,
            )

            tdlib_chat_id = int(chat["id"])
        other_name = str(chat.get("title", "")).strip() or str(target_user_id)

        print(f"Chat found: {other_name}")
        print(f"TDLib chat_id: {tdlib_chat_id}")

        raw_messages = get_chat_history(
            client=client,
            chat_id=tdlib_chat_id,
            max_messages=history_limit,
        )

        normalized_messages: list[dict[str, Any]] = []

        for raw_message in raw_messages:
            normalized = normalize_text_message(
                message=raw_message,
                tdlib_chat_id=tdlib_chat_id,
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
        set_app_state("target_tdlib_chat_id", str(tdlib_chat_id))
        set_app_state("target_chat_title", other_name)
        set_app_state("history_sync_limit", str(history_limit))

        result_path = chat_result_path()
        result_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "name": other_name,
            "type": "personal_chat",
            "id": target_user_id,
            "tdlib_chat_id": tdlib_chat_id,
            "messages": normalized_messages,
        }

        result_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        from src.chat_context import chat_db_path

        print()
        print(f"Total TDLib objects received: {len(raw_messages)}")
        print(f"Text messages saved: {len(normalized_messages)}")
        print(f"SQLite: {chat_db_path()}")
        print(f"JSON: {result_path}")

    finally:
        close_client(client)


if __name__ == "__main__":
    main()
