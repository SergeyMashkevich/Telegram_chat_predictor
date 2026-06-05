from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.telegram_client import TdlibClient


@dataclass
class SendTextMessagesResult:
    sent_messages: list[dict[str, Any]]
    used_reply: bool


class MessageSendError(RuntimeError):
    def __init__(
        self,
        message: str,
        sent_messages: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.sent_messages = sent_messages


def build_reply_to(
    reply_to_message_id: int | None,
) -> dict[str, Any] | None:
    if reply_to_message_id is None:
        return None

    return {
        "@type": "inputMessageReplyToMessage",
        "message_id": int(reply_to_message_id),
        "quote": None,
    }


def build_text_message_request(
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
) -> dict[str, Any]:
    return {
        "@type": "sendMessage",
        "chat_id": int(chat_id),
        "topic_id": None,
        "reply_to": build_reply_to(reply_to_message_id),
        "options": None,
        "reply_markup": None,
        "input_message_content": {
            "@type": "inputMessageText",
            "text": {
                "@type": "formattedText",
                "text": text,
                "entities": [],
            },
            "link_preview_options": None,
            "clear_draft": False,
        },
    }


def send_text_messages(
    client: TdlibClient,
    chat_id: int,
    messages: list[str],
    reply_to_message_id: int | None = None,
    reply_probability: float = 0.0,
) -> SendTextMessagesResult:
    clean_messages = [
        str(message).strip()
        for message in messages
        if str(message).strip()
    ]

    if not clean_messages:
        raise ValueError("Нет сообщений для отправки.")

    should_try_reply = (
        reply_to_message_id is not None
        and random.random() < max(0.0, min(1.0, reply_probability))
    )

    sent_messages: list[dict[str, Any]] = []
    used_reply = False

    for index, text in enumerate(clean_messages):
        reply_target = (
            int(reply_to_message_id)
            if index == 0 and should_try_reply and reply_to_message_id is not None
            else None
        )

        try:
            sent_message = client.request(
                build_text_message_request(
                    chat_id=chat_id,
                    text=text,
                    reply_to_message_id=reply_target,
                ),
                timeout=60.0,
            )

        except Exception as first_error:
            if reply_target is not None:
                try:
                    sent_message = client.request(
                        build_text_message_request(
                            chat_id=chat_id,
                            text=text,
                            reply_to_message_id=None,
                        ),
                        timeout=60.0,
                    )

                except Exception as second_error:
                    raise MessageSendError(
                        "Не удалось отправить сообщение "
                        f"даже без reply: {second_error}",
                        sent_messages=sent_messages,
                    ) from second_error

            else:
                raise MessageSendError(
                    f"Не удалось отправить сообщение: {first_error}",
                    sent_messages=sent_messages,
                ) from first_error

        else:
            if reply_target is not None:
                used_reply = True

        sent_messages.append(sent_message)

    return SendTextMessagesResult(
        sent_messages=sent_messages,
        used_reply=used_reply,
    )
