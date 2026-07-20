from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.chat_context import get_active_chat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def dataset_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    return DATASETS_DIR / "chats" / str(active_chat["chat_id"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(
            f"Dataset не найден: {path}\n"
            "Сначала выполните make export-training."
        )

    examples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Некорректный JSONL на строке {line_number}: {error}"
                ) from error

    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def review_path() -> Path:
    return dataset_dir() / "review.json"


def load_review() -> dict[str, Any]:
    path = review_path()

    if not path.exists():
        return {
            "dropped_indexes": [],
            "notes": {},
        }

    return json.loads(path.read_text(encoding="utf-8"))


def save_review(review: dict[str, Any]) -> None:
    path = review_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_message_content(example: dict[str, Any], role: str) -> str:
    for message in example.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))

    return ""


def print_example(
    examples: list[dict[str, Any]],
    index: int,
    dropped_indexes: set[int],
) -> None:
    example = examples[index]

    user_content = get_message_content(example, "user")
    assistant_content = get_message_content(example, "assistant")
    metadata = example.get("metadata", {})

    status = "DROP" if index in dropped_indexes else "KEEP"

    print()
    print("=" * 80)
    print(f"Example {index + 1}/{len(examples)} | status={status}")
    print("=" * 80)

    if metadata:
        target_count = metadata.get("target_message_count", "n/a")
        target_chars = metadata.get("target_chars", "n/a")
        print(f"target_message_count: {target_count}")
        print(f"target_chars:         {target_chars}")
        print()

    print("USER PROMPT")
    print("-" * 80)
    print(user_content[:3000])

    if len(user_content) > 3000:
        print("\n...[truncated]")

    print()
    print("TARGET ASSISTANT ANSWER")
    print("-" * 80)
    print(assistant_content[:1500])

    if len(assistant_content) > 1500:
        print("\n...[truncated]")

    print()
    print("Commands:")
    print("  k / Enter  keep and next")
    print("  d          drop and next")
    print("  u          undo drop / keep again")
    print("  n          next")
    print("  p          previous")
    print("  r          random")
    print("  j <num>    jump to example number")
    print("  f          write filtered dataset")
    print("  q          quit")
    print()


def write_filtered_dataset(
    examples: list[dict[str, Any]],
    dropped_indexes: set[int],
) -> Path:
    filtered = [
        example
        for index, example in enumerate(examples)
        if index not in dropped_indexes
    ]

    output_path = dataset_dir() / "sft.filtered.jsonl"
    write_jsonl(output_path, filtered)

    print()
    print("Filtered dataset written")
    print("========================")
    print(f"kept:    {len(filtered)}")
    print(f"dropped: {len(dropped_indexes)}")
    print(f"output:  {output_path}")
    print()

    return output_path


def main() -> None:
    input_path = dataset_dir() / "sft.jsonl"
    examples = load_jsonl(input_path)

    if not examples:
        raise RuntimeError("Dataset пустой.")

    review = load_review()
    dropped_indexes = {
        int(index)
        for index in review.get("dropped_indexes", [])
    }

    print()
    print("Training dataset inspector")
    print("==========================")
    print(f"input:    {input_path}")
    print(f"examples: {len(examples)}")
    print(f"dropped:  {len(dropped_indexes)}")
    print()

    index = 0

    while True:
        index = max(0, min(index, len(examples) - 1))
        print_example(examples, index, dropped_indexes)

        command = input("> ").strip()

        if command == "":
            command = "k"

        lower = command.lower()

        if lower in {"k", "keep"}:
            dropped_indexes.discard(index)
            index += 1

        elif lower in {"d", "drop"}:
            dropped_indexes.add(index)
            index += 1

        elif lower in {"u", "undo"}:
            dropped_indexes.discard(index)

        elif lower in {"n", "next"}:
            index += 1

        elif lower in {"p", "prev", "previous"}:
            index -= 1

        elif lower in {"r", "random"}:
            index = random.randint(0, len(examples) - 1)

        elif lower.startswith("j "):
            raw_number = lower.split(maxsplit=1)[1]

            try:
                number = int(raw_number)
            except ValueError:
                print("Номер должен быть числом.")
                continue

            index = number - 1

        elif lower in {"f", "filter", "write"}:
            review["dropped_indexes"] = sorted(dropped_indexes)
            save_review(review)
            write_filtered_dataset(examples, dropped_indexes)

        elif lower in {"q", "quit", "exit"}:
            review["dropped_indexes"] = sorted(dropped_indexes)
            save_review(review)

            print()
            print("Review saved.")
            print(f"dropped: {len(dropped_indexes)}")
            print(f"review:  {review_path()}")
            print()
            break

        else:
            print("Неизвестная команда.")

        if index >= len(examples):
            index = len(examples) - 1
            print()
            print("Вы дошли до конца dataset. Введите f, чтобы записать filtered dataset.")


if __name__ == "__main__":
    main()
