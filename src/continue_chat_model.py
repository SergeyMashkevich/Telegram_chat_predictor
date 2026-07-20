from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from src.chat_context import get_active_chat
from src.model_registry import suggested_trained_model_name
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


def dataset_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    return DATASETS_DIR / "chats" / str(active_chat["chat_id"])


def adapter_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    return ADAPTERS_DIR / "chats" / str(active_chat["chat_id"]) / "lora"


def backup_adapter(adapter_path: Path) -> Path:
    source = adapter_path / "adapters.safetensors"

    if not source.exists():
        raise RuntimeError(
            f"Нет существующего adapter для продолжения обучения: {source}"
        )

    backups_dir = adapter_path / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    destination = backups_dir / f"adapters.before_continue.{timestamp}.safetensors"

    shutil.copy2(source, destination)

    return destination


def main() -> None:
    load_dotenv(ENV_PATH)

    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    model = os.getenv(
        "TRAIN_BASE_MODEL",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
    ).strip()

    iters = os.getenv("CONTINUE_TRAIN_ITERS", "50").strip()
    batch_size = os.getenv("TRAIN_BATCH_SIZE", "1").strip()
    num_layers = os.getenv("TRAIN_NUM_LAYERS", "8").strip()

    data_path = dataset_dir() / "mlx"
    adapter_path = adapter_dir()
    resume_file = adapter_path / "adapters.safetensors"

    if not (data_path / "train.jsonl").exists():
        raise RuntimeError(
            f"Нет {data_path / 'train.jsonl'}. "
            "Сначала выполните make prepare-training."
        )

    if not resume_file.exists():
        raise RuntimeError(
            f"Нет adapter для продолжения обучения: {resume_file}. "
            "Сначала выполните make train-chat-model."
        )

    backup_path = backup_adapter(adapter_path)

    print()
    print("Continuing MLX LoRA training")
    print("============================")
    print(f"chat:          {active_chat['title']}")
    print(f"chat_id:       {active_chat['chat_id']}")
    print(f"base model:    {model}")
    print(f"adapter path:  {adapter_path}")
    print(f"resume file:   {resume_file}")
    print(f"backup:        {backup_path}")
    print(f"target name:   {suggested_trained_model_name()}")
    print(f"data:          {data_path}")
    print(f"iters:         {iters}")
    print(f"batch size:    {batch_size}")
    print(f"num layers:    {num_layers}")
    print()

    command = [
        "mlx_lm.lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_path),
        "--adapter-path",
        str(adapter_path),
        "--resume-adapter-file",
        str(resume_file),
        "--iters",
        iters,
        "--batch-size",
        batch_size,
        "--num-layers",
        num_layers,
        "--mask-prompt",
    ]

    subprocess.run(command, check=True)

    print()
    print("Continue training finished")
    print("==========================")
    print(f"adapter: {adapter_path}")
    print(f"backup:  {backup_path}")
    print()
    print("Теперь выполните make test-chat-model и сравните качество.")


if __name__ == "__main__":
    main()
