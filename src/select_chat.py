from __future__ import annotations

from typing import Any

from dotenv import load_dotenv

from src.chat_context import get_active_chat, set_active_chat
from src.history_sync import close_client, ensure_authorized, load_more_chats
from src.storage import initialize_database, set_app_state
from src.telegram_client import ENV_PATH, TdlibClient


def load_private_chats(client: TdlibClient) -> list[dict[str, Any]]:
    chat_lists: list[tuple[str, dict[str, Any] | None]] = [
        ("main", None),
        ("archive", {"@type": "chatListArchive"}),
    ]

    seen_chat_ids: set[int] = set()
    private_chats: list[dict[str, Any]] = []

    for _, chat_list in chat_lists:
        while True:
            loaded_more = load_more_chats(client, chat_list)

            chats = client.request(
                {
                    "@type": "getChats",
                    "chat_list": chat_list,
                    "limit": 10000,
                },
                timeout=60.0,
            )

            for raw_chat_id in chats.get("chat_ids", []):
                chat_id = int(raw_chat_id)

                if chat_id in seen_chat_ids:
                    continue

                seen_chat_ids.add(chat_id)

                chat = client.request(
                    {
                        "@type": "getChat",
                        "chat_id": chat_id,
                    }
                )

                chat_type = chat.get("type", {})

                if chat_type.get("@type") != "chatTypePrivate":
                    continue

                user_id = int(chat_type.get("user_id", 0))

                if user_id <= 0:
                    continue

                private_chats.append(
                    {
                        "title": str(chat.get("title", "")).strip() or str(user_id),
                        "user_id": user_id,
                        "chat_id": chat_id,
                    }
                )

            if not loaded_more:
                break

    return sorted(
        private_chats,
        key=lambda chat: chat["title"].lower(),
    )


def choose_chat(chats: list[dict[str, Any]]) -> dict[str, Any]:
    if not chats:
        raise RuntimeError("Личные чаты не найдены.")

    while True:
        query = input(
            "Введите часть имени для поиска или Enter, чтобы показать первые 50: "
        ).strip().lower()

        if query:
            filtered = [
                chat
                for chat in chats
                if query in chat["title"].lower()
            ]
        else:
            filtered = chats[:50]

        if not filtered:
            print("По этому фильтру чаты не найдены.")
            continue

        print()
        print("Найденные личные чаты:")

        for index, chat in enumerate(filtered, start=1):
            print(
                f"{index:>3}. {chat['title']} "
                f"| user_id={chat['user_id']} "
                f"| chat_id={chat['chat_id']}"
            )

        print()

        raw_choice = input(
            "Выберите номер чата или Enter, чтобы изменить поиск: "
        ).strip()

        if not raw_choice:
            continue

        try:
            choice = int(raw_choice)
        except ValueError:
            print("Введите число.")
            continue

        if 1 <= choice <= len(filtered):
            return filtered[choice - 1]

        print("Номер вне диапазона.")


def save_selected_chat(chat: dict[str, Any]) -> None:
    set_active_chat(chat)

    initialize_database()

    set_app_state("target_user_id", str(chat["user_id"]))
    set_app_state("target_tdlib_chat_id", str(chat["chat_id"]))
    set_app_state("target_chat_title", str(chat["title"]))


def select_and_save_chat(
    client: TdlibClient,
    *,
    allow_keep_current: bool,
) -> dict[str, Any]:
    current = get_active_chat()

    if allow_keep_current and current is not None:
        print()
        print("Текущий выбранный чат:")
        print(f"  title:   {current['title']}")
        print(f"  user_id: {current['user_id']}")
        print(f"  chat_id: {current['chat_id']}")
        print()

        keep = input(
            "Enter оставить этот чат, search выбрать другой: "
        ).strip().lower()

        if keep == "":
            save_selected_chat(current)
            return current

    print()
    print("Загружаю список личных чатов...")

    chats = load_private_chats(client)
    selected = choose_chat(chats)
    save_selected_chat(selected)

    print()
    print("Выбран чат:")
    print(f"  title:   {selected['title']}")
    print(f"  user_id: {selected['user_id']}")
    print(f"  chat_id: {selected['chat_id']}")

    return selected


def main() -> None:
    load_dotenv(ENV_PATH)

    client = TdlibClient()

    try:
        ensure_authorized(client)

        select_and_save_chat(
            client,
            allow_keep_current=True,
        )

    finally:
        close_client(client)


if __name__ == "__main__":
    main()
