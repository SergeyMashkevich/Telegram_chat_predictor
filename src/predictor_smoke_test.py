from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.predictor import OllamaPredictor
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = PROJECT_ROOT / "data" / "result.json"


def main() -> None:
    load_dotenv(ENV_PATH)

    if not RESULT_PATH.exists():
        raise RuntimeError(
            "data/result.json not found. "
            "Run python -m src.history_sync first."
        )

    result = json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )

    context_limit = int(
        os.getenv("RECENT_CONTEXT_MAX_MESSAGES", "100")
    )

    history_messages = result.get("messages", [])[-context_limit:]

    print("Enter incoming messages from the chat partner.")
    print("Enter each message on a new line.")
    print("An empty line finishes the input.")
    print()

    incoming_messages = []
    position = 1

    while True:
        text = input(f"[{position}] ").strip()

        if not text:
            break

        incoming_messages.append(
            {
                "id": position,
                "is_outgoing": False,
                "text": text,
                "reactions": [],
            }
        )

        position += 1

    if not incoming_messages:
        raise RuntimeError(
            "Enter at least one incoming message."
        )

    predictor = OllamaPredictor()

    print()
    print(f"Model: {predictor.model}")
    print("Generating candidates...")

    started_at = time.monotonic()

    candidates, raw_response = predictor.generate_candidate_specs(
        history_messages=history_messages,
        incoming_messages=incoming_messages,
    )

    elapsed = time.monotonic() - started_at

    print()
    print(f"Completed in {elapsed:.2f} seconds.")

    for candidate_position, candidate in enumerate(
        candidates,
        start=1,
    ):
        reply_index = int(
            candidate["reply_to_incoming_index"]
        )

        print()
        print(f"[{candidate_position}]")

        if reply_index == 0:
            print("(no reply)")
        else:
            target_text = incoming_messages[
                reply_index - 1
            ]["text"]

            print(
                f'(recommended reply to [{reply_index}] "{target_text}")'
            )

        for message in candidate["messages"]:
            print(message)

    print()
    print("Raw Ollama JSON response:")
    print(raw_response)


if __name__ == "__main__":
    main()
