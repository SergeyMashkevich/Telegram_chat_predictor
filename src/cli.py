from __future__ import annotations

import os
import select
import sys
from typing import Any

from dotenv import load_dotenv

from src.app_status import print_status
from src.live_history_sync import sync_current_chat
from src.generation_status import print_generation_status
from src.manual_storage import (
    claim_candidate_for_send,
    claim_custom_response_for_send,
    get_oldest_pending_batch_id,
    get_pending_candidates,
    mark_response_failed,
    mark_response_sent,
    skip_batch,
)
from src.message_sender import MessageSendError, send_text_messages
from src.storage import get_app_state, set_app_state, get_batch_messages
from src.telegram_client import ENV_PATH, TdlibClient


load_dotenv(ENV_PATH)


def poll_cli_line() -> str | None:
    readable, _, _ = select.select(
        [sys.stdin],
        [],
        [],
        0,
    )

    if not readable:
        return None

    line = sys.stdin.readline()

    if line == "":
        return "quit"

    return line.strip()


def print_cli_help() -> None:
    print()
    print("[CLI manual mode]")
    print("  list                    show pending candidates")
    print("  1, 2, 3                 send a candidate for the oldest batch")
    print("  send <batch> <number>    send a specific candidate")
    print("  e <text>                 send custom text for the oldest batch")
    print("  write <batch> <text>     send custom text for a batch")
    print("  s                        skip the oldest batch")
    print("  skip <batch>             skip a specific batch")
    print("  help                     show commands")
    print("  status                   show application status")
    print("  sync [limit]             update current chat history")
    print("  model                    show generation provider and adapter")
    print("  mode                     show current mode")
    print("  auto                     enable automatic sending")
    print("  manual                   enable manual mode")
    print("  quit                     exit the application")
    print()


def _preview(text: str, limit: int = 70) -> str:
    compact = " ".join(text.split())

    if len(compact) <= limit:
        return compact

    return f"{compact[:limit - 3]}..."


def _format_reply_hint(
    batch_id: int,
    reply_to_incoming_index: int,
) -> str:
    if reply_to_incoming_index <= 0:
        return "(no reply)"

    messages = get_batch_messages(batch_id)
    index = reply_to_incoming_index - 1

    if index < 0 or index >= len(messages):
        return "(invalid reply value)"

    text = str(messages[index].get("text", ""))

    return f'(reply to [{reply_to_incoming_index}] "{_preview(text)}")'


def print_pending_candidates(chat_id: int) -> None:
    batches = get_pending_candidates(chat_id)

    if not batches:
        print("[CLI] No batches are waiting for a selection.", flush=True)
        return

    for batch in batches:
        batch_id = int(batch["batch_id"])

        print()
        print(f"[batch #{batch_id} awaiting selection]")

        for candidate in batch["candidates"]:
            print()
            print(f"[{candidate['position']}]")
            print(
                _format_reply_hint(
                    batch_id=batch_id,
                    reply_to_incoming_index=int(
                        candidate.get("reply_to_incoming_index", 0)
                    ),
                )
            )

            for message in candidate["messages"]:
                print(message)

    print()


def _message_ids(
    sent_messages: list[dict[str, Any]],
) -> list[int]:
    return [
        int(message["id"])
        for message in sent_messages
        if "id" in message
    ]


def _reply_probability() -> float:
    try:
        return float(os.getenv("REPLY_USE_PROBABILITY", "0.25"))
    except ValueError:
        return 0.25


def _send_claimed_response(
    client: TdlibClient,
    claimed: dict[str, Any],
) -> None:
    response_id = int(claimed["response_id"])
    batch_id = int(claimed["batch_id"])
    chat_id = int(claimed["chat_id"])
    messages = list(claimed["messages"])
    reply_to_message_id = claimed.get("reply_to_message_id")

    try:
        result = send_text_messages(
            client=client,
            chat_id=chat_id,
            messages=messages,
            reply_to_message_id=reply_to_message_id,
            reply_probability=_reply_probability(),
        )

    except MessageSendError as error:
        message_ids = _message_ids(error.sent_messages)

        mark_response_failed(
            response_id=response_id,
            chat_id=chat_id,
            message_ids=message_ids,
            error_text=str(error),
        )

        print(
            f"[CLI] Send error for batch #{batch_id}: {error}",
            flush=True,
        )
        return

    message_ids = _message_ids(result.sent_messages)

    mark_response_sent(
        response_id=response_id,
        chat_id=chat_id,
        message_ids=message_ids,
        reply_used=result.used_reply,
    )

    if result.used_reply:
        print(
            f"[CLI] Response for batch #{batch_id} sent with a reply.",
            flush=True,
        )
    else:
        print(
            f"[CLI] Response for batch #{batch_id} sent without a reply.",
            flush=True,
        )


def _send_candidate(
    client: TdlibClient,
    batch_id: int,
    position: int,
) -> None:
    claimed = claim_candidate_for_send(
        batch_id=batch_id,
        position=position,
    )

    if claimed is None:
        print(
            "[CLI] Candidate not found or batch already handled.",
            flush=True,
        )
        return

    _send_claimed_response(
        client=client,
        claimed=claimed,
    )


def _send_custom_text(
    client: TdlibClient,
    batch_id: int,
    text: str,
) -> None:
    claimed = claim_custom_response_for_send(
        batch_id=batch_id,
        text=text,
    )

    if claimed is None:
        print(
            "[CLI] Text is empty or batch already handled.",
            flush=True,
        )
        return

    _send_claimed_response(
        client=client,
        claimed=claimed,
    )


def _oldest_pending_batch(chat_id: int) -> int | None:
    batch_id = get_oldest_pending_batch_id(chat_id)

    if batch_id is None:
        print(
            "[CLI] No batches are waiting for a selection.",
            flush=True,
        )

    return batch_id


def handle_cli_command(
    line: str,
    *,
    chat_id: int,
    client: TdlibClient,
) -> bool:
    command_line = line.strip()

    if not command_line:
        return True

    lower = command_line.lower()

    if lower in {"help", "h", "?"}:
        print_cli_help()
        return True

    if lower in {"model", "provider", "generation"}:
        print_generation_status()
        return True

    if lower == "model":
        info = describe_model_selection()

        print()
        print("[CLI] Model selection")
        print(f"  resolved:        {info['resolved_model']}")
        print(f"  reason:          {info['reason']}")
        print(f"  selected_model:  {info['selected_model'] or 'not set'}")
        print(f"  trained_model:   {info['suggested_trained_model']}")
        print(f"  trained_exists:  {info['trained_model_exists']}")
        print(f"  default_model:   {info['default_model']}")

        if info["available_error"]:
            print(f"  ollama_error:    {info['available_error']}")

        print()
        return True

        if not models:
            print("[CLI] No Ollama models found.", flush=True)
            return True

        print()
        print("[CLI] Available Ollama models:")

        for model_name in models:
            print(f"  {model_name}")

        print()
        return True

        print(
            f"[CLI] Selected model for the current chat: {model_name}",
            flush=True,
        )
        return True

    if lower in {"status", "st"}:
        print_status()
        return True

    if lower == "sync" or lower.startswith("sync "):
        parts = command_line.split(maxsplit=1)

        if len(parts) == 2:
            try:
                limit = int(parts[1])
            except ValueError:
                print("[CLI] limit must be a number.", flush=True)
                return True
        else:
            try:
                limit = int(os.getenv("HISTORY_SYNC_LIMIT", "1000"))
            except ValueError:
                limit = 1000

        print(
            f"[CLI] Updating current chat history, limit={limit}...",
            flush=True,
        )

        try:
            result = sync_current_chat(
                client=client,
                chat_id=chat_id,
                limit=limit,
            )
        except Exception as error:
            print(f"[CLI] Sync error: {error}", flush=True)
            return True

        print(
            f"[CLI] Sync complete: "
            f"{result['chat_title']}, "
            f"TDLib objects={result['raw_count']}, "
            f"saved={result['normalized_count']}.",
            flush=True,
        )
        return True

    if lower == "mode":
        mode = get_app_state("app_mode") or os.getenv("APP_MODE", "manual")
        print(f"[CLI] Current mode: {mode}", flush=True)
        return True

    if lower == "auto":
        set_app_state("app_mode", "auto")
        print("[CLI] Auto mode enabled.", flush=True)
        return True

    if lower == "manual":
        set_app_state("app_mode", "manual")
        print("[CLI] Manual mode enabled.", flush=True)
        return True

    if lower in {"list", "l"}:
        print_pending_candidates(chat_id)
        return True

    if lower in {"quit", "q", "exit"}:
        return False

    if command_line.isdigit():
        batch_id = _oldest_pending_batch(chat_id)

        if batch_id is not None:
            _send_candidate(
                client=client,
                batch_id=batch_id,
                position=int(command_line),
            )

        return True

    if lower == "s":
        batch_id = _oldest_pending_batch(chat_id)

        if batch_id is not None:
            if skip_batch(batch_id):
                print(
                    f"[CLI] Batch #{batch_id} skipped.",
                    flush=True,
                )
            else:
                print(
                    "[CLI] Batch already handled.",
                    flush=True,
                )

        return True

    if lower.startswith("e "):
        batch_id = _oldest_pending_batch(chat_id)

        if batch_id is not None:
            text = command_line[2:].strip()

            _send_custom_text(
                client=client,
                batch_id=batch_id,
                text=text,
            )

        return True

    parts = command_line.split(maxsplit=2)
    command = parts[0].lower()

    try:
        if command == "send" and len(parts) == 3:
            _send_candidate(
                client=client,
                batch_id=int(parts[1]),
                position=int(parts[2]),
            )
            return True

        if command == "write" and len(parts) == 3:
            _send_custom_text(
                client=client,
                batch_id=int(parts[1]),
                text=parts[2],
            )
            return True

        if command == "skip" and len(parts) == 2:
            batch_id = int(parts[1])

            if skip_batch(batch_id):
                print(
                    f"[CLI] Batch #{batch_id} skipped.",
                    flush=True,
                )
            else:
                print(
                    "[CLI] Batch not found or already handled.",
                    flush=True,
                )

            return True

    except ValueError:
        print("[CLI] Invalid numeric argument.", flush=True)
        return True

    print(
        "[CLI] Unknown command. Enter help.",
        flush=True,
    )

    return True
