from __future__ import annotations

import os

from dotenv import load_dotenv

from src.live_history_sync import sync_current_chat
from src.storage import connect, initialize_database
from src.telegram_client import ENV_PATH, TdlibClient


def is_enabled() -> bool:
    value = os.getenv("STARTUP_SYNC_ON_RUN", "true").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def count_messages() -> int:
    initialize_database()

    with connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()

    return int(row[0] or 0)


def startup_sync(
    client: TdlibClient,
    chat_id: int,
) -> None:
    load_dotenv(ENV_PATH)

    if not is_enabled():
        print("[startup sync] отключён.")
        return

    existing_messages = count_messages()

    if existing_messages == 0:
        limit = int(os.getenv("HISTORY_SYNC_LIMIT", "1000"))

        print(
            f"[startup sync] База чата пустая. "
            f"Первичная синхронизация, limit={limit}...",
            flush=True,
        )
    else:
        limit = int(os.getenv("STARTUP_SYNC_LIMIT", "300"))

        print(
            f"[startup sync] В базе уже {existing_messages} сообщений. "
            f"Догоняющая синхронизация, limit={limit}...",
            flush=True,
        )

    result = sync_current_chat(
        client=client,
        chat_id=chat_id,
        limit=limit,
    )

    print(
        f"[startup sync] Готово: "
        f"{result['chat_title']}, "
        f"TDLib objects={result['raw_count']}, "
        f"saved={result['normalized_count']}.",
        flush=True,
    )
