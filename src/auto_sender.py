from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.manual_storage import (
    claim_best_candidate_for_auto_send,
    mark_response_failed,
    mark_response_sent,
)
from src.message_sender import MessageSendError, send_text_messages
from src.storage import get_app_state
from src.telegram_client import ENV_PATH, TdlibClient


load_dotenv(ENV_PATH)


def app_mode() -> str:
    value = get_app_state("app_mode") or os.getenv("APP_MODE", "manual")
    return value.strip().lower()


def reply_probability() -> float:
    try:
        return float(os.getenv("REPLY_USE_PROBABILITY", "0.25"))
    except ValueError:
        return 0.25


def auto_send_delay_seconds() -> float:
    try:
        return float(os.getenv("AUTO_SEND_DELAY_SECONDS", "2"))
    except ValueError:
        return 2.0


def auto_min_final_score() -> float:
    try:
        return float(os.getenv("AUTO_MIN_FINAL_SCORE", "0.0"))
    except ValueError:
        return 0.0


def message_ids(messages: list[dict[str, Any]]) -> list[int]:
    return [
        int(message["id"])
        for message in messages
        if "id" in message
    ]


class AutoSender:
    def __init__(self, chat_id: int) -> None:
        self.chat_id = int(chat_id)

    def tick(self, client: TdlibClient) -> None:
        if app_mode() != "auto":
            return

        claimed = claim_best_candidate_for_auto_send(
            chat_id=self.chat_id,
            min_final_score=auto_min_final_score(),
            delay_seconds=auto_send_delay_seconds(),
        )

        if claimed is None:
            return

        response_id = int(claimed["response_id"])
        batch_id = int(claimed["batch_id"])
        chat_id = int(claimed["chat_id"])
        candidate_position = int(claimed["candidate_position"])
        final_score = claimed.get("final_score")
        messages = list(claimed["messages"])
        reply_to_message_id = claimed.get("reply_to_message_id")

        score_text = (
            "n/a"
            if final_score is None
            else f"{float(final_score):.3f}"
        )

        print(
            f"[AUTO] Отправляю кандидат [{candidate_position}] "
            f"для блока #{batch_id}, score={score_text}.",
            flush=True,
        )

        try:
            result = send_text_messages(
                client=client,
                chat_id=chat_id,
                messages=messages,
                reply_to_message_id=reply_to_message_id,
                reply_probability=reply_probability(),
            )

        except MessageSendError as error:
            mark_response_failed(
                response_id=response_id,
                chat_id=chat_id,
                message_ids=message_ids(error.sent_messages),
                error_text=str(error),
            )

            print(
                f"[AUTO] Ошибка отправки для блока #{batch_id}: {error}",
                flush=True,
            )
            return

        mark_response_sent(
            response_id=response_id,
            chat_id=chat_id,
            message_ids=message_ids(result.sent_messages),
            reply_used=result.used_reply,
        )

        if result.used_reply:
            print(
                f"[AUTO] Ответ для блока #{batch_id} отправлен с reply.",
                flush=True,
            )
        else:
            print(
                f"[AUTO] Ответ для блока #{batch_id} отправлен без reply.",
                flush=True,
            )
