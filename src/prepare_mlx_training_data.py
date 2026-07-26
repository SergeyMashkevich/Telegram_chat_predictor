from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.chat_context import get_active_chat
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def dataset_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "No chat selected. Run make run or make select-chat first."
        )

    return DATASETS_DIR / "chats" / str(active_chat["chat_id"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(
            f"Dataset not found: {path}\n"
            "Run make export-training and make inspect-training first."
        )

    examples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv(ENV_PATH)

    ratio = float(os.getenv("TRAIN_VALID_RATIO", "0.10"))

    base_dir = dataset_dir()
    filtered_path = base_dir / "sft.filtered.jsonl"
    source_path = filtered_path if filtered_path.exists() else base_dir / "sft.jsonl"

    examples = load_jsonl(source_path)

    if len(examples) < 5:
        raise RuntimeError(
            f"Too few examples for a train/valid split: {len(examples)}"
        )

    random.seed(42)
    shuffled = list(examples)
    random.shuffle(shuffled)

    valid_count = max(1, int(len(shuffled) * ratio))
    valid_examples = shuffled[:valid_count]
    train_examples = shuffled[valid_count:]

    output_dir = base_dir / "mlx"

    write_jsonl(output_dir / "train.jsonl", train_examples)
    write_jsonl(output_dir / "valid.jsonl", valid_examples)

    print()
    print("MLX training data prepared")
    print("==========================")
    print(f"source: {source_path}")
    print(f"train:  {output_dir / 'train.jsonl'} ({len(train_examples)})")
    print(f"valid:  {output_dir / 'valid.jsonl'} ({len(valid_examples)})")


if __name__ == "__main__":
    main()
