from __future__ import annotations

import time
from typing import Any

from src.storage import (
    append_message_to_batch,
    close_batch,
    close_due_incoming_batches,
    create_message_batch,
    get_batch_messages,
    get_collecting_batch,
    mark_latest_incoming_batch_answered,
    message_has_batch,
)


class IncomingBatcher:
    def __init__(
        self,
        chat_id: int,
        delay_seconds: float,
    ) -> None:
        self.chat_id = int(chat_id)
        self.delay_seconds = float(delay_seconds)

    def handle_new_message(
        self,
        message: dict[str, Any],
    ) -> None:
        message_id = int(message["id"])

        if message_has_batch(self.chat_id, message_id):
            return

        if bool(message.get("is_outgoing")):
            batch_id = mark_latest_incoming_batch_answered(
                self.chat_id
            )

            if batch_id is not None:
                print(
                    f"[incoming batch #{batch_id}] "
                    "marked as answered manually.",
                    flush=True,
                )

            return

        now = int(time.time())

        collecting_batch = get_collecting_batch(
            self.chat_id,
            "incoming",
        )

        if collecting_batch is not None:
            elapsed = (
                now
                - int(collecting_batch["last_activity_at"])
            )

            if elapsed > self.delay_seconds:
                old_batch_id = int(
                    collecting_batch["batch_id"]
                )

                close_batch(
                    old_batch_id,
                    "ready_for_prediction",
                )

                self.print_ready_batch(old_batch_id)
                collecting_batch = None

        if collecting_batch is None:
            batch_id = create_message_batch(
                chat_id=self.chat_id,
                direction="incoming",
                message_id=message_id,
                activity_time=now,
            )

            print(
                f"[incoming batch #{batch_id}] "
                "started collecting messages.",
                flush=True,
            )

            return

        batch_id = int(collecting_batch["batch_id"])

        appended = append_message_to_batch(
            batch_id=batch_id,
            chat_id=self.chat_id,
            message_id=message_id,
            activity_time=now,
        )

        if appended:
            print(
                f"[incoming batch #{batch_id}] "
                "added another message.",
                flush=True,
            )

    def close_due_batches(self) -> None:
        cutoff_time = int(
            time.time() - self.delay_seconds
        )

        batch_ids = close_due_incoming_batches(
            chat_id=self.chat_id,
            cutoff_time=cutoff_time,
        )

        for batch_id in batch_ids:
            self.print_ready_batch(batch_id)

    def print_ready_batch(self, batch_id: int) -> None:
        messages = get_batch_messages(batch_id)

        print()
        print(
            f"[incoming batch #{batch_id} ready]",
            flush=True,
        )

        for index, message in enumerate(messages, start=1):
            text = str(message.get("text", "")).strip()

            print(
                f"{index}. {text}",
                flush=True,
            )

        print()
