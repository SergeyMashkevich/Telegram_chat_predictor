from __future__ import annotations

import time
from typing import Any

from src.manual_storage import record_external_telegram_response
from src.storage import (
    append_message_to_batch,
    close_due_outgoing_batches,
    create_message_batch,
    get_batch_messages,
    get_collecting_batch,
    message_has_batch,
)


class OutgoingBatcher:
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
        if not bool(message.get("is_outgoing")):
            return

        message_id = int(message["id"])

        if message_has_batch(self.chat_id, message_id):
            return

        now = int(time.time())

        collecting_batch = get_collecting_batch(
            self.chat_id,
            "outgoing",
        )

        if collecting_batch is not None:
            elapsed = now - int(collecting_batch["last_activity_at"])

            if elapsed > self.delay_seconds:
                old_batch_id = int(collecting_batch["batch_id"])
                self._close_and_record(old_batch_id)
                collecting_batch = None

        if collecting_batch is None:
            batch_id = create_message_batch(
                chat_id=self.chat_id,
                direction="outgoing",
                message_id=message_id,
                activity_time=now,
            )

            print(
                f"[outgoing batch #{batch_id}] started collecting messages.",
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
                f"[outgoing batch #{batch_id}] "
                "added another message.",
                flush=True,
            )

    def close_due_batches(self) -> None:
        cutoff_time = int(time.time() - self.delay_seconds)

        batch_ids = close_due_outgoing_batches(
            chat_id=self.chat_id,
            cutoff_time=cutoff_time,
        )

        for batch_id in batch_ids:
            self._record_closed_batch(batch_id)

    def _close_and_record(self, batch_id: int) -> None:
        from src.storage import close_batch

        close_batch(
            batch_id=batch_id,
            status="outgoing_ready",
        )

        self._record_closed_batch(batch_id)

    def _record_closed_batch(self, batch_id: int) -> None:
        messages = get_batch_messages(batch_id)

        print()
        print(f"[outgoing batch #{batch_id} ready]")

        for index, message in enumerate(messages, start=1):
            print(f"{index}. {message.get('text', '')}")

        linked_incoming_batch_id = record_external_telegram_response(
            outgoing_batch_id=batch_id,
        )

        if linked_incoming_batch_id is None:
            print(
                f"[outgoing batch #{batch_id}] "
                "saved but not linked to an incoming batch.",
                flush=True,
            )
        else:
            print(
                f"[outgoing batch #{batch_id}] "
                f"linked to incoming batch #{linked_incoming_batch_id}.",
                flush=True,
            )

        print()
