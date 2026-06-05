from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.auto_sender import AutoSender
from src.batching import IncomingBatcher
from src.outgoing_batching import OutgoingBatcher
from src.cli import (
    handle_cli_command,
    poll_cli_line,
    print_cli_help,
)
from src.manual_storage import (
    initialize_manual_storage,
    is_application_sent_message,
    replace_response_message_id,
)
from src.history_sync import close_client, ensure_authorized
from src.prediction_worker import PredictionWorker
from src.response_matcher import ResponseMatcherWorker
from src.message_normalizer import (
    extract_message_reactions,
    normalize_text_message,
)
from src.result_store import (
    delete_result_messages,
    upsert_result_message,
)
from src.startup_sync import startup_sync
from src.storage import (
    delete_messages,
    get_app_state,
    get_message,
    initialize_database,
    replace_message_id_in_batches,
    upsert_messages,
)
from src.telegram_client import ENV_PATH, TdlibClient


def require_int_setting(
    state_key: str,
    env_key: str,
) -> int:
    value = get_app_state(state_key) or os.getenv(env_key)

    if value is None or not value.strip():
        raise RuntimeError(
            f"Не найдено значение {state_key} или {env_key}. "
            "Сначала запустите python -m src.history_sync"
        )

    return int(value)


def full_name(user: dict[str, Any]) -> str:
    first_name = str(user.get("first_name", "")).strip()
    last_name = str(user.get("last_name", "")).strip()

    return f"{first_name} {last_name}".strip()


def preview_text(text: str, limit: int = 100) -> str:
    compact = " ".join(text.split())

    if len(compact) <= limit:
        return compact

    return f"{compact[:limit - 3]}..."


def format_reaction_target(message: dict[str, Any]) -> str:
    content_type = str(message.get("content_type", ""))
    text = preview_text(str(message.get("text", "")), limit=80)

    if content_type == "messageSticker":
        target_type = "стикеру"
    elif content_type in {"messageAnimatedEmoji", "messageDice"}:
        target_type = "эмодзи"
    elif content_type in {
        "messagePhoto",
        "messageVideo",
        "messageAnimation",
        "messageDocument",
        "messageAudio",
        "messageVoiceNote",
        "messageVideoNote",
    }:
        target_type = "медиа-сообщению"
    else:
        target_type = "сообщению"

    return f'{target_type} "{text}"'


def format_reactions(reactions: list[dict[str, Any]]) -> str:
    if not reactions:
        return "нет реакций"

    parts = []

    for reaction in reactions:
        reaction_type = reaction.get("type")

        if reaction_type == "emoji":
            label = reaction.get("emoji") or "emoji"
        elif reaction_type == "custom_emoji":
            label = "[custom emoji]"
        elif reaction_type == "paid":
            label = "[paid reaction]"
        else:
            label = "[unknown reaction]"

        count = int(reaction.get("total_count", 0))
        chosen = " ваша" if reaction.get("is_chosen") else ""

        parts.append(f"{label} × {count}{chosen}")

    return ", ".join(parts)


def save_message(
    message: dict[str, Any],
    *,
    target_chat_id: int,
    target_user_id: int,
    own_user_id: int,
    own_name: str,
    other_name: str,
    result_max_messages: int,
    remove_if_not_text: bool = False,
    log_message: bool = True,
) -> dict[str, Any] | None:
    chat_id = int(message.get("chat_id", 0))

    if chat_id != target_chat_id:
        return None

    message_id = int(message["id"])

    normalized = normalize_text_message(
        message=message,
        tdlib_chat_id=target_chat_id,
        own_user_id=own_user_id,
        own_name=own_name,
        other_user_id=target_user_id,
        other_name=other_name,
    )

    if normalized is None:
        if remove_if_not_text:
            delete_messages(target_chat_id, [message_id])
            delete_result_messages([message_id])

            print(
                f"Сообщение {message_id} удалено из локального снимка.",
                flush=True,
            )

        return None

    upsert_messages([normalized])

    upsert_result_message(
        message=normalized,
        chat_name=other_name,
        target_user_id=target_user_id,
        tdlib_chat_id=target_chat_id,
        max_messages=result_max_messages,
    )

    if log_message:
        direction = (
            "исходящее"
            if normalized["is_outgoing"]
            else "входящее"
        )

        print(
            f"[{direction}] "
            f"id={normalized['id']} "
            f"text={preview_text(normalized['text'])}",
            flush=True,
        )

    return normalized


def handle_message_content_update(
    client: TdlibClient,
    event: dict[str, Any],
    *,
    target_chat_id: int,
    target_user_id: int,
    own_user_id: int,
    own_name: str,
    other_name: str,
    result_max_messages: int,
) -> None:
    chat_id = int(event.get("chat_id", 0))

    if chat_id != target_chat_id:
        return

    message_id = int(event["message_id"])

    try:
        message = client.request(
            {
                "@type": "getMessage",
                "chat_id": target_chat_id,
                "message_id": message_id,
            }
        )

    except RuntimeError as error:
        if "404" in str(error):
            delete_messages(target_chat_id, [message_id])
            delete_result_messages([message_id])
            return

        raise

    save_message(
        message,
        target_chat_id=target_chat_id,
        target_user_id=target_user_id,
        own_user_id=own_user_id,
        own_name=own_name,
        other_name=other_name,
        result_max_messages=result_max_messages,
        remove_if_not_text=True,
    )


def handle_message_interaction_update(
    client: TdlibClient,
    event: dict[str, Any],
    *,
    target_chat_id: int,
    target_user_id: int,
    own_user_id: int,
    own_name: str,
    other_name: str,
    result_max_messages: int,
) -> None:
    chat_id = int(event.get("chat_id", 0))

    if chat_id != target_chat_id:
        return

    message_id = int(event["message_id"])
    saved_message = get_message(target_chat_id, message_id)

    if saved_message is None:
        try:
            message = client.request(
                {
                    "@type": "getMessage",
                    "chat_id": target_chat_id,
                    "message_id": message_id,
                }
            )

        except RuntimeError:
            return

        save_message(
            message,
            target_chat_id=target_chat_id,
            target_user_id=target_user_id,
            own_user_id=own_user_id,
            own_name=own_name,
            other_name=other_name,
            result_max_messages=result_max_messages,
            log_message=False,
        )

        saved_message = get_message(target_chat_id, message_id)

    if saved_message is None:
        return

    old_reactions = saved_message.get("reactions", [])

    new_reactions = extract_message_reactions(
        event.get("interaction_info")
    )

    if old_reactions == new_reactions:
        return

    saved_message["reactions"] = new_reactions

    upsert_messages([saved_message])

    upsert_result_message(
        message=saved_message,
        chat_name=other_name,
        target_user_id=target_user_id,
        tdlib_chat_id=target_chat_id,
        max_messages=result_max_messages,
    )

    print(
        f"[реакции] "
        f"к {format_reaction_target(saved_message)} "
        f"id={message_id}: "
        f"{format_reactions(new_reactions)}",
        flush=True,
    )


def main() -> None:
    load_dotenv(ENV_PATH)
    initialize_database()
    initialize_manual_storage()

    target_user_id = require_int_setting(
        "target_user_id",
        "TARGET_USER_ID",
    )

    target_chat_id = require_int_setting(
        "target_tdlib_chat_id",
        "TARGET_TDLIB_CHAT_ID",
    )

    result_max_messages = int(
        os.getenv(
            "RESULT_JSON_MAX_MESSAGES",
            os.getenv("HISTORY_SYNC_LIMIT", "1000"),
        )
    )

    batch_delay_seconds = float(
        os.getenv("INCOMING_BATCH_DELAY_SECONDS", "10")
    )

    outgoing_batch_delay_seconds = float(
        os.getenv("OUTGOING_BATCH_DELAY_SECONDS", "5")
    )

    batcher = IncomingBatcher(
        chat_id=target_chat_id,
        delay_seconds=batch_delay_seconds,
    )

    outgoing_batcher = OutgoingBatcher(
        chat_id=target_chat_id,
        delay_seconds=outgoing_batch_delay_seconds,
    )

    auto_sender = AutoSender(
        chat_id=target_chat_id,
    )

    prediction_worker = PredictionWorker(
        chat_id=target_chat_id,
    )

    response_matcher = ResponseMatcherWorker()

    client = TdlibClient()

    try:
        ensure_authorized(client)

        me = client.request({"@type": "getMe"})
        own_user_id = int(me["id"])
        own_name = full_name(me)

        chat = client.request(
            {
                "@type": "getChat",
                "chat_id": target_chat_id,
            }
        )

        other_name = (
            str(chat.get("title", "")).strip()
            or str(target_user_id)
        )

        startup_sync(
            client=client,
            chat_id=target_chat_id,
        )

        print(f"Live-синхронизация запущена для чата: {other_name}")
        print(f"TDLib chat_id: {target_chat_id}")
        print(
            f"Задержка входящего блока: "
            f"{batch_delay_seconds:g} секунд"
        )
        print(
            f"Задержка исходящего блока: "
            f"{outgoing_batch_delay_seconds:g} секунд"
        )
        print("Ожидание новых сообщений. Для остановки нажмите Ctrl + C.")

        prediction_worker.start()
        response_matcher.start()
        print_cli_help()

        while True:
            event = client.receive(timeout=1.0)

            batcher.close_due_batches()
            outgoing_batcher.close_due_batches()

            cli_line = poll_cli_line()

            if cli_line is not None:
                keep_running = handle_cli_command(
                    cli_line,
                    chat_id=target_chat_id,
                    client=client,
                )

                if not keep_running:
                    print("Завершение по CLI-команде.")
                    break

            auto_sender.tick(client)

            if event is None:
                continue

            event_type = event.get("@type")

            if event_type == "updateNewMessage":
                message = event["message"]

                if int(message.get("chat_id", 0)) != target_chat_id:
                    continue

                is_outgoing = bool(message.get("is_outgoing"))
                message_id = int(message["id"])

                app_sent = (
                    is_outgoing
                    and is_application_sent_message(
                        chat_id=target_chat_id,
                        message_id=message_id,
                    )
                )

                normalized = save_message(
                    message,
                    target_chat_id=target_chat_id,
                    target_user_id=target_user_id,
                    own_user_id=own_user_id,
                    own_name=own_name,
                    other_name=other_name,
                    result_max_messages=result_max_messages,
                    log_message=not app_sent,
                )

                if normalized is not None and not app_sent:
                    if bool(normalized.get("is_outgoing")):
                        batcher.handle_new_message(normalized)
                        outgoing_batcher.handle_new_message(normalized)
                    else:
                        batcher.handle_new_message(normalized)

            elif event_type == "updateMessageContent":
                handle_message_content_update(
                    client,
                    event,
                    target_chat_id=target_chat_id,
                    target_user_id=target_user_id,
                    own_user_id=own_user_id,
                    own_name=own_name,
                    other_name=other_name,
                    result_max_messages=result_max_messages,
                )

            elif event_type == "updateMessageInteractionInfo":
                handle_message_interaction_update(
                    client,
                    event,
                    target_chat_id=target_chat_id,
                    target_user_id=target_user_id,
                    own_user_id=own_user_id,
                    own_name=own_name,
                    other_name=other_name,
                    result_max_messages=result_max_messages,
                )

            elif event_type == "updateMessageSendSucceeded":
                message = event["message"]

                if int(message.get("chat_id", 0)) != target_chat_id:
                    continue

                old_message_id = int(event["old_message_id"])
                new_message_id = int(message["id"])

                replace_message_id_in_batches(
                    chat_id=target_chat_id,
                    old_message_id=old_message_id,
                    new_message_id=new_message_id,
                )

                replace_response_message_id(
                    chat_id=target_chat_id,
                    old_message_id=old_message_id,
                    new_message_id=new_message_id,
                )

                delete_messages(
                    target_chat_id,
                    [old_message_id],
                )

                delete_result_messages([old_message_id])

                app_sent = is_application_sent_message(
                    chat_id=target_chat_id,
                    message_id=new_message_id,
                )

                save_message(
                    message,
                    target_chat_id=target_chat_id,
                    target_user_id=target_user_id,
                    own_user_id=own_user_id,
                    own_name=own_name,
                    other_name=other_name,
                    result_max_messages=result_max_messages,
                    log_message=not app_sent,
                )

            elif event_type == "updateDeleteMessages":
                chat_id = int(event.get("chat_id", 0))

                if chat_id != target_chat_id:
                    continue

                if not bool(event.get("is_permanent")):
                    continue

                message_ids = [
                    int(message_id)
                    for message_id in event.get("message_ids", [])
                ]

                delete_messages(
                    target_chat_id,
                    message_ids,
                )

                delete_result_messages(message_ids)

                print(
                    f"Удалено сообщений из локального снимка: "
                    f"{len(message_ids)}",
                    flush=True,
                )

            elif event_type == "updateAuthorizationState":
                state = event.get("authorization_state", {})
                state_type = state.get("@type")

                if state_type == "authorizationStateClosed":
                    print("TDLib-соединение закрыто.")
                    break

    except KeyboardInterrupt:
        print("\nОстановка live-синхронизации...")

    finally:
        response_matcher.stop()
        prediction_worker.stop()
        close_client(client)


if __name__ == "__main__":
    main()
