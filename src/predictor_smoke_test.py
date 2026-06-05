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
            "data/result.json не найден. "
            "Сначала запустите python -m src.history_sync"
        )

    result = json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )

    context_limit = int(
        os.getenv("RECENT_CONTEXT_MAX_MESSAGES", "100")
    )

    history_messages = result.get("messages", [])[-context_limit:]

    print("Введите входящие сообщения собеседника.")
    print("Каждое сообщение вводите с новой строки.")
    print("Пустая строка завершает ввод.")
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
            "Нужно ввести хотя бы одно входящее сообщение."
        )

    predictor = OllamaPredictor()

    print()
    print(f"Модель: {predictor.model}")
    print("Генерация кандидатов...")

    started_at = time.monotonic()

    candidates, raw_response = predictor.generate_candidate_specs(
        history_messages=history_messages,
        incoming_messages=incoming_messages,
    )

    elapsed = time.monotonic() - started_at

    print()
    print(f"Готово за {elapsed:.2f} секунд.")

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
            print("(без reply)")
        else:
            target_text = incoming_messages[
                reply_index - 1
            ]["text"]

            print(
                f'(рекомендуемый reply на [{reply_index}] "{target_text}")'
            )

        for message in candidate["messages"]:
            print(message)

    print()
    print("Сырой JSON-ответ Ollama:")
    print(raw_response)


if __name__ == "__main__":
    main()
